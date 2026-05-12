"""Official TestGenEval bridge.

Generates official-compatible prediction JSONL and runs the official
TestGenEval evaluation/report scripts as subprocesses.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

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


@dataclass
class OfficialBridgeResult:
    """Result of running the official TestGenEval bridge."""

    predictions_path: Path
    tasks_path: Path
    manifest_path: Path
    official_logs_dir: Path
    official_reports_dir: Path
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


def _write_predictions_jsonl(
    predictions_path: Path,
    tasks_path: Path,
    cases: Iterable[BenchmarkCase],
    model_name: str,
    generate: Callable[[BenchmarkCase], str],
    max_cases: Optional[int] = None,
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
                if not isinstance(test_code, str):
                    raise TypeError(f"Generator returned {type(test_code)}, expected str")
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

    result = OfficialBridgeResult(
        predictions_path=predictions_path,
        tasks_path=tasks_path,
        manifest_path=manifest_path,
        official_logs_dir=official_logs_dir,
        official_reports_dir=official_reports_dir,
    )

    # Generate predictions
    written, failed, manifest = _write_predictions_jsonl(
        predictions_path, tasks_path, cases, model_name, generate, max_cases
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
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
            _official_dir_arg(official_logs_dir),
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
            _official_dir_arg(official_logs_dir),
            "--output_dir",
            _official_dir_arg(official_reports_dir),
        ]

        report_rc = _run_subprocess(report_cmd, official_repo_dir, result, "report")

        result.summary_copied, result.report_copied = _copy_report_outputs(
            official_reports_dir,
            bench_out,
            model_name,
        )
    else:
        result.errors.append("No predictions were written; official evaluation was skipped.")

    if eval_rc != 0:
        result.errors.append("Evaluation subprocess failed; report may be incomplete.")
    if report_rc != 0:
        result.errors.append("Report subprocess failed.")

    return result
