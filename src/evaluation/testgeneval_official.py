"""Official TestGenEval bridge.

Generates official-compatible prediction JSONL and runs the official
TestGenEval evaluation/report scripts as subprocesses.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from .benchmarks.testgeneval import DATASET_IDS
from .models import BenchmarkCase

log = logging.getLogger(__name__)

_MAX_LOG_CHARS = 8192


def _truncate(text: str, limit: int = _MAX_LOG_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-half:]


def _official_path_arg(path: Path) -> str:
    """Return a path string compatible with TestGenEval's POSIX-style parsing.

    TestGenEval's report utilities split log paths on "/" even on Windows.
    Passing forward-slash paths keeps the official parser from treating the
    whole absolute Windows path as the instance id.
    """
    return path.resolve().as_posix()


def _official_dir_arg(path: Path) -> str:
    """Return a forward-slash directory path that keeps Windows glob output parseable."""
    return _official_path_arg(path).rstrip("/") + "/"


def _needs_docker_safe_path(path: Path) -> bool:
    """Return True when TestGenEval's Docker command is likely to mishandle a path."""
    return " " in str(path.resolve())


def _staging_logs_dir_for(logs_dir: Path, benchmark: str) -> Path:
    """Create a deterministic no-space staging path for official Docker logs."""
    key = hashlib.sha1(str(logs_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "llm_agent_testgeneval" / key / benchmark / "official_logs"


def _copy_tree_contents(src: Path, dst: Path) -> None:
    """Copy files from a staging directory into the requested artifact directory."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target)


@dataclass
class OfficialBridgeResult:
    """Result of running the official TestGenEval bridge."""

    predictions_path: Path
    tasks_path: Path
    manifest_path: Path
    official_logs_dir: Path
    official_reports_dir: Path
    staging_logs_dir: Optional[Path] = None
    summary_copied: Optional[Path] = None
    report_copied: Optional[Path] = None
    commands_run: List[List[str]] = field(default_factory=list)
    returncodes: List[int] = field(default_factory=list)
    stdout_logs: List[str] = field(default_factory=list)
    stderr_logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "predictions_path": str(self.predictions_path),
            "tasks_path": str(self.tasks_path),
            "manifest_path": str(self.manifest_path),
            "official_logs_dir": str(self.official_logs_dir),
            "official_reports_dir": str(self.official_reports_dir),
            "staging_logs_dir": str(self.staging_logs_dir) if self.staging_logs_dir else None,
            "summary_copied": str(self.summary_copied) if self.summary_copied else None,
            "report_copied": str(self.report_copied) if self.report_copied else None,
            "commands_run": self.commands_run,
            "returncodes": self.returncodes,
            "stdout_logs": self.stdout_logs,
            "stderr_logs": self.stderr_logs,
            "errors": self.errors,
            "counts": self.counts,
        }


def _validate_official_repo(official_repo_dir: Path) -> None:
    required = ["run_evaluation.py", "generate_report.py"]
    missing = [f for f in required if not (official_repo_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Official repo dir {official_repo_dir} missing required files: {missing}"
        )


def _is_pytest_raises_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "raises"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def _is_dummy_assert(node: ast.Assert) -> bool:
    test = node.test
    if isinstance(test, ast.Constant):
        return test.value is True
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        left = test.left
        right = test.comparators[0]
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            return left.value == right.value
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return isinstance(test.operand, ast.Constant) and test.operand.value is False
    return False


def _is_testcase_subclass(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in {"SimpleTestCase", "TestCase"}:
            return True
        if isinstance(base, ast.Attribute) and base.attr in {"SimpleTestCase", "TestCase"}:
            return True
    return False


def _test_method_names(node: ast.ClassDef) -> list[str]:
    return [
        item.name
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name.startswith("test_")
    ]


def validate_official_prediction(test_code: str, *, allow_test_classes: bool = False) -> str:
    """Reject unusable official TestGenEval predictions before Docker scoring."""
    if not isinstance(test_code, str):
        raise TypeError(f"Generator returned {type(test_code)}, expected str")

    cleaned = test_code.strip()
    if not cleaned:
        raise ValueError("Generated test code is empty")

    try:
        tree = ast.parse(cleaned)
    except SyntaxError as exc:
        raise ValueError(f"Generated test code is not valid Python: {exc.msg}") from exc

    imports_pytest = any(
        (isinstance(node, ast.Import) and any(alias.name == "pytest" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "pytest")
        for node in tree.body
    )
    uses_pytest = any(isinstance(node, ast.Name) and node.id == "pytest" for node in ast.walk(tree))
    if imports_pytest or uses_pytest:
        raise ValueError(
            "Generated TestGenEval full-mode code must not import or use pytest; "
            "official repo containers may run with unittest only"
        )

    class_tests = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if class_tests:
        named_test_classes = [node.name for node in class_tests if node.name.startswith("Test")]
        if named_test_classes and not allow_test_classes:
            raise ValueError(
                "Generated TestGenEval full-mode code must use file-level test_* "
                f"functions, not test classes: {', '.join(named_test_classes)}"
            )
        if allow_test_classes:
            invalid_classes = [
                node.name
                for node in class_tests
                if _test_method_names(node) and not _is_testcase_subclass(node)
            ]
            if invalid_classes:
                raise ValueError(
                    "Generated TestGenEval class-based tests must inherit from "
                    f"SimpleTestCase or TestCase: {', '.join(invalid_classes)}"
                )

    has_pytest_test = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        for node in tree.body
    )
    has_class_test = allow_test_classes and any(_test_method_names(node) for node in class_tests)
    if not has_pytest_test and not has_class_test:
        raise ValueError("Generated test code must define at least one file-level pytest test")

    assert_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    has_meaningful_assert = any(not _is_dummy_assert(node) for node in assert_nodes)
    has_pytest_raises = any(
        any(_is_pytest_raises_call(item.context_expr) for item in node.items)
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
    )
    if not has_meaningful_assert and not has_pytest_raises:
        if assert_nodes:
            raise ValueError("Generated test code contains only dummy assertions")
        raise ValueError("Generated test code must contain meaningful assertions")

    return cleaned + "\n"


def _function_source(lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    start = node.lineno
    if node.decorator_list:
        start = min(decorator.lineno for decorator in node.decorator_list)
    return lines[start - 1 : node.end_lineno]


def _add_self_to_function_header(line: str) -> str:
    pattern = re.compile(r"^(\s*(?:async\s+def|def)\s+\w+\s*)\(([^)]*)\)(\s*(?:->[^:]+)?\s*:.*)$")
    match = pattern.match(line)
    if not match:
        return line
    args = match.group(2).strip()
    if args.startswith("self"):
        return line
    replacement_args = f"self, {args}" if args else "self"
    return f"{match.group(1)}({replacement_args}){match.group(3)}"


def wrap_django_official_prediction(test_code: str) -> str:
    """Wrap file-level TestGenEval Django tests in a SimpleTestCase class."""
    cleaned = validate_official_prediction(test_code)
    tree = ast.parse(cleaned)
    test_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if not test_functions:
        return cleaned

    lines = cleaned.splitlines()
    removed_lines: set[int] = set()
    method_blocks: list[list[str]] = []
    for node in test_functions:
        block = _function_source(lines, node)
        for i in range(
            (
                min(decorator.lineno for decorator in node.decorator_list)
                if node.decorator_list
                else node.lineno
            ),
            node.end_lineno + 1,
        ):
            removed_lines.add(i)

        method_block = []
        header_rewritten = False
        for line in block:
            stripped = line.lstrip()
            if not header_rewritten and (
                stripped.startswith("def ") or stripped.startswith("async def ")
            ):
                line = _add_self_to_function_header(line)
                header_rewritten = True
            method_block.append("    " + line)
        method_blocks.append(method_block)

    preamble_lines = [
        line.rstrip() for i, line in enumerate(lines, start=1) if i not in removed_lines
    ]
    while preamble_lines and not preamble_lines[-1].strip():
        preamble_lines.pop()

    has_simple_testcase = any("SimpleTestCase" in line for line in preamble_lines)
    if not has_simple_testcase:
        preamble_lines.insert(0, "from django.test import SimpleTestCase")

    class_lines = ["", "", "class TestsHarness(SimpleTestCase):"]
    for block in method_blocks:
        class_lines.extend(["", *block])

    wrapped = "\n".join(preamble_lines + class_lines).strip() + "\n"
    return validate_official_prediction(wrapped, allow_test_classes=True)


def _write_predictions_jsonl(
    predictions_path: Path,
    tasks_path: Path,
    cases: Iterable[BenchmarkCase],
    model_name: str,
    generate: Callable[[BenchmarkCase], str],
    max_cases: Optional[int] = None,
    validate: Optional[Callable[[str], str]] = validate_official_prediction,
) -> tuple[int, int, List[dict]]:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    manifest_entries: List[dict] = []

    with (
        predictions_path.open("w", encoding="utf-8") as pred_file,
        tasks_path.open("w", encoding="utf-8") as task_file,
    ):
        for i, case in enumerate(cases):
            if max_cases is not None and i >= max_cases:
                break
            official_id = case.metadata.get("id") or case.id
            instance_id = case.metadata.get("instance_id") or official_id
            task_file.write(
                json.dumps(_official_task_record(case, official_id, instance_id)) + "\n"
            )
            entry: dict = {
                "case_index": i,
                "case_id": case.id,
                "official_id": official_id,
                "status": "ok",
                "error": None,
            }
            try:
                test_code = generate(case)
                if validate is not None:
                    test_code = validate(test_code)
                record = {
                    "id": official_id,
                    "instance_id": instance_id,
                    "model_name_or_path": model_name,
                    "preds": {"full": [test_code]},
                }
                pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                failed += 1
                entry["status"] = "failed"
                entry["error"] = str(exc)
                log.warning("Generation failed for case %s: %s", case.id, exc)
            manifest_entries.append(entry)

    return written, failed, manifest_entries


def _official_task_record(case: BenchmarkCase, official_id: str, instance_id: str) -> dict:
    """Build a local task record accepted by the official TestGenEval scripts."""
    metadata = dict(case.metadata)
    preds_context = metadata.get("preds_context")
    if not isinstance(preds_context, dict):
        preds_context = {}
    preds_context.setdefault("code_src", case.code)
    preds_context.setdefault("last", "")

    return {
        "id": official_id,
        "instance_id": instance_id,
        "repo": metadata.get("repo", ""),
        "version": metadata.get("version", ""),
        "base_commit": metadata.get("base_commit", ""),
        "code_file": metadata.get("code_file", ""),
        "test_file": metadata.get("test_file", ""),
        "preds_context": preds_context,
        "test_patch": metadata.get("test_patch", ""),
        "patch": metadata.get("patch", ""),
        "baseline_covs": metadata.get("baseline_covs", {}),
    }


def _run_subprocess(
    cmd: List[str],
    cwd: Path,
    result: OfficialBridgeResult,
    label: str,
) -> int:
    log.info("Running %s: %s", label, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        result.errors.append(f"{label} subprocess failed to start: {exc}")
        result.returncodes.append(-1)
        result.stdout_logs.append("")
        result.stderr_logs.append(str(exc))
        result.commands_run.append(cmd)
        log.error("%s subprocess failed to start: %s", label, exc)
        return -1

    result.commands_run.append(cmd)
    result.returncodes.append(proc.returncode)
    result.stdout_logs.append(_truncate(proc.stdout))
    result.stderr_logs.append(_truncate(proc.stderr))

    if proc.returncode != 0:
        result.errors.append(f"{label} exited with code {proc.returncode}")
        log.warning("%s exited with code %d", label, proc.returncode)
    else:
        log.info("%s completed with code 0", label)

    return proc.returncode


def _copy_if_present(
    src_dir: Path, dst_dir: Path, src_names: List[str], dst_name: str
) -> Optional[Path]:
    for name in src_names:
        src = src_dir / name
        if src.is_file():
            dst = dst_dir / dst_name
            try:
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                log.info("Copied %s -> %s", src, dst)
                return dst
            except Exception as exc:
                log.warning("Failed to copy %s to %s: %s", src, dst, exc)
    return None


def _copy_report_outputs(
    src_dir: Path, dst_dir: Path, model_name: str
) -> tuple[Optional[Path], Optional[Path]]:
    summary_names = [
        f"{model_name}_summary.json",
        "summary.json",
        "report_summary.json",
    ]
    report_names = [
        f"{model_name}_report.json",
        f"{model_name}_full.json",
        "report.json",
        "full_report.json",
    ]
    summary = _copy_if_present(src_dir, dst_dir, summary_names, "official_summary.json")
    report = _copy_if_present(src_dir, dst_dir, report_names, "official_report.json")
    return summary, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _docker_image_for_task(task: dict[str, Any], namespace: str) -> str:
    """Return the official TestGenEval Docker image name for a task record."""
    repo = str(task.get("repo") or "").strip().replace("/", "_")
    version = str(task.get("version") or "").strip()
    repo_slug = repo or "unknown"
    tag = version or "unknown"
    return f"{namespace}/swe-bench-{repo_slug}-testbed:{tag}"


def _prediction_task_pairs(
    predictions_path: Path,
    tasks_path: Path,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair prediction records with matching official task records by id."""
    tasks_by_id = {task.get("id"): task for task in _read_jsonl(tasks_path)}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for prediction in _read_jsonl(predictions_path):
        task = tasks_by_id.get(prediction.get("id"))
        if task is not None:
            pairs.append((prediction, task))
    return pairs


def _group_pairs_by_image(
    pairs: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    namespace: str,
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for prediction, task in pairs:
        image = _docker_image_for_task(task, namespace)
        groups.setdefault(image, []).append((prediction, task))
    return groups


def _existing_generation_counts(predictions_path: Path, manifest_path: Path) -> dict[str, int]:
    """Summarize already-generated official predictions for scorer-only reruns."""
    written = len(_read_jsonl(predictions_path)) if predictions_path.is_file() else 0
    failed = 0
    total_attempted = written

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = []
        if isinstance(manifest, list):
            failed = sum(1 for entry in manifest if entry.get("status") != "ok")
            total_attempted = len(manifest)

    return {
        "written": written,
        "failed": failed,
        "total_attempted": total_attempted,
    }


def _count_methods(code_str: str) -> int:
    import re

    return len(re.findall(r"\bdef\b\s+\w+\s*\(", code_str))


def _lines_of_code(code_str: str) -> int:
    return len(code_str.strip().splitlines()) if code_str.strip() else 0


def _prediction_lexical_report(predictions_path: Path, tasks_by_id: dict[str, dict]) -> dict:
    preds = _read_jsonl(predictions_path)
    loc: list[int] = []
    methods: list[int] = []
    baseline_loc: list[int] = []
    baseline_methods: list[int] = []

    for pred in preds:
        pred_texts = (pred.get("preds") or {}).get("full") or []
        for pred_text in pred_texts:
            if isinstance(pred_text, str):
                loc.append(_lines_of_code(pred_text))
                methods.append(_count_methods(pred_text))

        task = tasks_by_id.get(pred.get("id")) or {}
        baseline = (task.get("preds_context") or {}).get("last") or ""
        if isinstance(baseline, list):
            baseline = baseline[0] if baseline else ""
        if isinstance(baseline, str):
            baseline_loc.append(_lines_of_code(baseline))
            baseline_methods.append(_count_methods(baseline))

    report: dict[str, Any] = {}
    if loc:
        report["av_pred_full_loc"] = sum(loc) / len(loc)
        report["av_pred_full_num_methods"] = sum(methods) / len(methods)
    if baseline_loc:
        report["av_baseline_loc"] = sum(baseline_loc) / len(baseline_loc)
        report["av_baseline_num_methods"] = sum(baseline_methods) / len(baseline_methods)
    return report


def _summarize_official_reports(predictions_path: Path, detailed_report: dict) -> dict:
    predictions = _read_jsonl(predictions_path)
    summary: dict[str, Any] = {
        "repo": "all",
        "total_predictions": len(predictions),
    }
    total_metrics: dict[str, list[Any]] = {}
    for case_report in detailed_report.values():
        for key, value in case_report.items():
            total_metrics.setdefault(key, []).append(value)

    for metric, values in total_metrics.items():
        clean_values = [value for value in values if value != -1]
        if not clean_values:
            summary[metric] = -1
        else:
            summary[metric] = sum(clean_values) / len(clean_values)
    return summary


def _build_report_map(predictions_path: Path, logs_dir: Path, model_name: str) -> dict:
    predictions = _read_jsonl(predictions_path)
    report = {
        "no_generation": [],
        "generated": [],
        "with_logs": [],
        "install_fail": [],
        "reset_failed": [],
        "test_errored": [],
        "test_timeout": [],
        "mutation_timeout": [],
    }
    for pred in predictions:
        case_id = pred.get("id")
        if not case_id:
            continue
        preds = pred.get("preds") or {}
        if not preds.get("full"):
            report["no_generation"].append(case_id)
            continue
        report["generated"].append(case_id)
        log_path = logs_dir / f"{case_id}.{model_name}.full.eval.log"
        if not log_path.exists():
            continue
        report["with_logs"].append(case_id)
        content = log_path.read_text(encoding="utf-8", errors="replace")
        if "Reset Failed" in content:
            report["reset_failed"].append(case_id)
        if "Tests Errored" in content:
            report["test_errored"].append(case_id)
        if "Test script run timed out" in content or "Tests Timed Out" in content:
            report["test_timeout"].append(case_id)
        if "MutationTimeout" in content:
            report["mutation_timeout"].append(case_id)
    return report


def _write_fallback_report_outputs(
    official_repo_dir: Path,
    predictions_path: Path,
    tasks_path: Path,
    logs_dir: Path,
    reports_dir: Path,
    model_name: str,
) -> bool:
    """Generate official report artifacts without the Windows-broken report CLI."""
    sys.path.insert(0, str(official_repo_dir))
    try:
        from swebench_docker.swebench_utils import get_eval_reports_for_logs

        tasks_by_id = {task["id"]: task for task in _read_jsonl(tasks_path)}
        log_paths = sorted(logs_dir.glob(f"*{model_name}*.log"))
        if not log_paths:
            return False

        normalized_logs = [_official_path_arg(path) for path in log_paths]
        raw_report = get_eval_reports_for_logs(
            normalized_logs,
            tasks_by_id,
            verbose=False,
            raw_only=True,
        )
        detailed_report = get_eval_reports_for_logs(
            normalized_logs,
            tasks_by_id,
            verbose=False,
            raw_only=False,
        )
        summary = _summarize_official_reports(predictions_path, detailed_report)
        summary.update(_prediction_lexical_report(predictions_path, tasks_by_id))
        report = _build_report_map(predictions_path, logs_dir, model_name)
    finally:
        try:
            sys.path.remove(str(official_repo_dir))
        except ValueError:
            pass

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{model_name}_full.json").write_text(
        json.dumps(raw_report, indent=4),
        encoding="utf-8",
    )
    (reports_dir / f"{model_name}_summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )
    (reports_dir / f"{model_name}_report.json").write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )
    return True


def run_official_bridge(
    benchmark: str,
    cases: Iterable[BenchmarkCase],
    output_dir: Path,
    model_name: str,
    official_repo_dir: Path,
    *,
    namespace: str = "kdjain",
    timeout: int = 900,
    num_processes: int = 1,
    skip_mutation: bool = False,
    skip_existing: bool = True,
    max_cases: Optional[int] = None,
    generate: Callable[[BenchmarkCase], str],
    validate_prediction: Optional[Callable[[str], str]] = validate_official_prediction,
    reuse_predictions: bool = False,
) -> OfficialBridgeResult:
    """Run the official TestGenEval bridge.

    Args:
        benchmark: Benchmark variant. Must be "testgeneval_lite" or "testgeneval".
        cases: Iterable of benchmark cases to process.
        output_dir: Directory where predictions, logs, and reports will be written.
        model_name: Model identifier to embed in predictions.
        official_repo_dir: Path to the cloned official TestGenEval repository.
        namespace: Docker namespace for evaluation.
        timeout: Timeout per evaluation instance in seconds.
        num_processes: Number of parallel evaluation processes.
        skip_mutation: Whether to skip mutation testing during evaluation.
        skip_existing: Whether to skip already-evaluated instances.
        max_cases: Optional cap on the number of cases to process.
        generate: Callable that receives a ``BenchmarkCase`` and returns the
            generated test code as a ``str``.
        validate_prediction: Optional generated-test validator. Defaults to
            rejecting empty, non-Python, no-test, and dummy-only predictions.
        reuse_predictions: Skip generation and score existing predictions/tasks
            files in the requested output directory.

    Returns:
        ``OfficialBridgeResult`` with paths, counts, commands, return codes,
        captured (truncated) logs, and any errors.
    """
    if benchmark not in DATASET_IDS:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Choose from: {', '.join(DATASET_IDS)}")
    if "/" in model_name or "\\" in model_name:
        raise ValueError(
            "model_name must not contain path separators because the official "
            "TestGenEval report script uses it in output filenames"
        )

    official_repo_dir = official_repo_dir.resolve()
    output_dir = output_dir.resolve()
    _validate_official_repo(official_repo_dir)

    bench_out = output_dir / benchmark
    bench_out.mkdir(parents=True, exist_ok=True)

    predictions_path = bench_out / "predictions.jsonl"
    tasks_path = bench_out / "official_tasks.jsonl"
    manifest_path = bench_out / "generation_manifest.json"
    official_logs_dir = bench_out / "official_logs"
    official_reports_dir = bench_out / "official_reports"
    official_logs_dir.mkdir(parents=True, exist_ok=True)
    official_reports_dir.mkdir(parents=True, exist_ok=True)
    eval_logs_dir = official_logs_dir
    if _needs_docker_safe_path(official_logs_dir):
        eval_logs_dir = _staging_logs_dir_for(official_logs_dir, benchmark)
        eval_logs_dir.mkdir(parents=True, exist_ok=True)

    result = OfficialBridgeResult(
        predictions_path=predictions_path,
        tasks_path=tasks_path,
        manifest_path=manifest_path,
        official_logs_dir=official_logs_dir,
        official_reports_dir=official_reports_dir,
        staging_logs_dir=eval_logs_dir if eval_logs_dir != official_logs_dir else None,
    )

    if reuse_predictions:
        missing = [str(path) for path in (predictions_path, tasks_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot reuse official TestGenEval predictions; missing files: "
                + ", ".join(missing)
            )
        result.counts = _existing_generation_counts(predictions_path, manifest_path)
        written = result.counts["written"]
        log.info("Reusing %d existing predictions from %s", written, predictions_path)
    else:
        # Generate predictions
        written, failed, manifest = _write_predictions_jsonl(
            predictions_path,
            tasks_path,
            cases,
            model_name,
            generate,
            max_cases,
            validate_prediction,
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result.counts = {
            "written": written,
            "failed": failed,
            "total_attempted": written + failed,
        }
        log.info(
            "Wrote %d predictions (%d failed) to %s",
            written,
            failed,
            predictions_path,
        )

    if result.staging_logs_dir:
        result.counts["staging_logs_used"] = True

    # Run official evaluation
    eval_rc = report_rc = 0
    if written:
        eval_cmd: List[str] = [
            sys.executable,
            "run_evaluation.py",
            "--predictions_path",
            _official_path_arg(predictions_path),
            "--swe_bench_tasks",
            _official_path_arg(tasks_path),
            "--namespace",
            namespace,
            "--timeout",
            str(timeout),
            "--num_processes",
            str(num_processes),
            "--log_dir",
            _official_dir_arg(eval_logs_dir),
        ]
        if skip_mutation:
            eval_cmd.append("--skip_mutation")
        if skip_existing:
            eval_cmd.append("--skip_existing")

        eval_rc = _run_subprocess(eval_cmd, official_repo_dir, result, "evaluation")

        # Run official report generation. The official script writes
        # <model>_summary.json, <model>_report.json, and <model>_full.json
        # into --output_dir.
        report_cmd: List[str] = [
            sys.executable,
            "generate_report.py",
            "--predictions_path",
            _official_path_arg(predictions_path),
            "--swe_bench_tasks",
            _official_path_arg(tasks_path),
            "--log_dir",
            _official_dir_arg(eval_logs_dir),
            "--output_dir",
            _official_dir_arg(official_reports_dir),
        ]

        error_start = len(result.errors)
        report_rc = _run_subprocess(report_cmd, official_repo_dir, result, "report")

        if report_rc != 0:
            try:
                fallback_ok = _write_fallback_report_outputs(
                    official_repo_dir,
                    predictions_path,
                    tasks_path,
                    eval_logs_dir,
                    official_reports_dir,
                    model_name,
                )
            except Exception as exc:  # noqa: BLE001 - fallback should report, not crash
                fallback_ok = False
                result.errors.append(f"fallback report generation failed: {exc}")
            if fallback_ok:
                report_rc = 0
                del result.errors[error_start:]
                result.counts["report_fallback_used"] = True

        result.summary_copied, result.report_copied = _copy_report_outputs(
            official_reports_dir,
            bench_out,
            model_name,
        )
        if result.staging_logs_dir:
            _copy_tree_contents(eval_logs_dir, official_logs_dir)
    else:
        result.errors.append("No predictions were written; official evaluation was skipped.")

    if eval_rc != 0:
        result.errors.append("Evaluation subprocess failed; report may be incomplete.")
    if report_rc != 0:
        result.errors.append("Report subprocess failed.")

    return result


def run_official_bridge_windowed(
    benchmark: str,
    cases: Iterable[BenchmarkCase],
    output_dir: Path,
    model_name: str,
    official_repo_dir: Path,
    *,
    namespace: str = "kdjain",
    timeout: int = 900,
    window_images: int = 1,
    num_processes_per_image: int = 2,
    delete_images_after: bool = False,
    skip_mutation: bool = False,
    skip_existing: bool = True,
    max_cases: Optional[int] = None,
    generate: Callable[[BenchmarkCase], str],
    validate_prediction: Optional[Callable[[str], str]] = validate_official_prediction,
    reuse_predictions: bool = False,
) -> OfficialBridgeResult:
    """Run official TestGenEval evaluation in Docker-image windows.

    The function writes the same full predictions/tasks artifacts as
    ``run_official_bridge``. It then groups generated predictions by the Docker
    image required by each task, evaluates one window of images at a time, and
    finally runs the official report over the aggregate logs.
    """
    if benchmark not in DATASET_IDS:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Choose from: {', '.join(DATASET_IDS)}")
    if "/" in model_name or "\\" in model_name:
        raise ValueError(
            "model_name must not contain path separators because the official "
            "TestGenEval report script uses it in output filenames"
        )

    window_images = max(1, int(window_images))
    num_processes_per_image = max(1, int(num_processes_per_image))

    official_repo_dir = official_repo_dir.resolve()
    output_dir = output_dir.resolve()
    _validate_official_repo(official_repo_dir)

    bench_out = output_dir / benchmark
    bench_out.mkdir(parents=True, exist_ok=True)

    predictions_path = bench_out / "predictions.jsonl"
    tasks_path = bench_out / "official_tasks.jsonl"
    manifest_path = bench_out / "generation_manifest.json"
    official_logs_dir = bench_out / "official_logs"
    official_reports_dir = bench_out / "official_reports"
    windows_dir = bench_out / "windows"
    official_logs_dir.mkdir(parents=True, exist_ok=True)
    official_reports_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)
    eval_logs_dir = official_logs_dir
    if _needs_docker_safe_path(official_logs_dir):
        eval_logs_dir = _staging_logs_dir_for(official_logs_dir, benchmark)
        eval_logs_dir.mkdir(parents=True, exist_ok=True)

    result = OfficialBridgeResult(
        predictions_path=predictions_path,
        tasks_path=tasks_path,
        manifest_path=manifest_path,
        official_logs_dir=official_logs_dir,
        official_reports_dir=official_reports_dir,
        staging_logs_dir=eval_logs_dir if eval_logs_dir != official_logs_dir else None,
    )

    if reuse_predictions:
        missing = [str(path) for path in (predictions_path, tasks_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot reuse official TestGenEval predictions; missing files: "
                + ", ".join(missing)
            )
        result.counts = _existing_generation_counts(predictions_path, manifest_path)
        written = result.counts["written"]
        log.info("Reusing %d existing predictions from %s", written, predictions_path)
    else:
        written, failed, manifest = _write_predictions_jsonl(
            predictions_path,
            tasks_path,
            cases,
            model_name,
            generate,
            max_cases,
            validate_prediction,
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result.counts = {
            "written": written,
            "failed": failed,
            "total_attempted": written + failed,
        }
        log.info(
            "Wrote %d predictions (%d failed) to %s",
            written,
            failed,
            predictions_path,
        )

    if result.staging_logs_dir:
        result.counts["staging_logs_used"] = True

    if not written:
        result.errors.append("No predictions were written; official evaluation was skipped.")
        return result

    pairs = _prediction_task_pairs(predictions_path, tasks_path)
    image_groups = _group_pairs_by_image(pairs, namespace)
    image_items = list(image_groups.items())
    result.counts.update(
        {
            "image_count": len(image_items),
            "window_images": window_images,
            "num_processes_per_image": num_processes_per_image,
            "delete_images_after": delete_images_after,
            "window_count": 0,
            "windows": [],
            "images": [image for image, _pairs in image_items],
        }
    )

    eval_failures = 0
    deleted_images: list[str] = []
    for start in range(0, len(image_items), window_images):
        window_number = len(result.counts["windows"]) + 1
        selected = image_items[start : start + window_images]
        window_pairs = [pair for _image, group_pairs in selected for pair in group_pairs]
        window_prediction_records = [prediction for prediction, _task in window_pairs]
        window_task_records = [task for _prediction, task in window_pairs]
        window_dir = windows_dir / f"window_{window_number:03d}"
        window_predictions_path = window_dir / "predictions.jsonl"
        window_tasks_path = window_dir / "official_tasks.jsonl"
        _write_jsonl(window_predictions_path, window_prediction_records)
        _write_jsonl(window_tasks_path, window_task_records)

        images = [image for image, _group_pairs in selected]
        num_processes = num_processes_per_image * max(1, len(images))
        result.counts["windows"].append(
            {
                "index": window_number,
                "images": images,
                "prediction_count": len(window_prediction_records),
                "predictions_path": str(window_predictions_path),
                "tasks_path": str(window_tasks_path),
                "num_processes": num_processes,
            }
        )

        eval_cmd: List[str] = [
            sys.executable,
            "run_evaluation.py",
            "--predictions_path",
            _official_path_arg(window_predictions_path),
            "--swe_bench_tasks",
            _official_path_arg(window_tasks_path),
            "--namespace",
            namespace,
            "--timeout",
            str(timeout),
            "--num_processes",
            str(num_processes),
            "--log_dir",
            _official_dir_arg(eval_logs_dir),
        ]
        if skip_mutation:
            eval_cmd.append("--skip_mutation")
        if skip_existing:
            eval_cmd.append("--skip_existing")

        eval_rc = _run_subprocess(
            eval_cmd,
            official_repo_dir,
            result,
            f"evaluation window {window_number}",
        )
        if eval_rc != 0:
            eval_failures += 1

        if delete_images_after:
            for image in images:
                delete_rc = _run_subprocess(
                    ["docker", "image", "rm", image],
                    official_repo_dir,
                    result,
                    f"delete image {image}",
                )
                if delete_rc == 0:
                    deleted_images.append(image)

    result.counts["window_count"] = len(result.counts["windows"])
    if deleted_images:
        result.counts["deleted_images"] = deleted_images

    report_cmd: List[str] = [
        sys.executable,
        "generate_report.py",
        "--predictions_path",
        _official_path_arg(predictions_path),
        "--swe_bench_tasks",
        _official_path_arg(tasks_path),
        "--log_dir",
        _official_dir_arg(eval_logs_dir),
        "--output_dir",
        _official_dir_arg(official_reports_dir),
    ]

    error_start = len(result.errors)
    report_rc = _run_subprocess(report_cmd, official_repo_dir, result, "report")

    if report_rc != 0:
        try:
            fallback_ok = _write_fallback_report_outputs(
                official_repo_dir,
                predictions_path,
                tasks_path,
                eval_logs_dir,
                official_reports_dir,
                model_name,
            )
        except Exception as exc:  # noqa: BLE001 - fallback should report, not crash
            fallback_ok = False
            result.errors.append(f"fallback report generation failed: {exc}")
        if fallback_ok:
            report_rc = 0
            del result.errors[error_start:]
            result.counts["report_fallback_used"] = True

    result.summary_copied, result.report_copied = _copy_report_outputs(
        official_reports_dir,
        bench_out,
        model_name,
    )
    if result.staging_logs_dir:
        _copy_tree_contents(eval_logs_dir, official_logs_dir)

    if eval_failures:
        result.errors.append("One or more evaluation windows failed; report may be incomplete.")
    if report_rc != 0:
        result.errors.append("Report subprocess failed.")

    return result
