"""Run an optional Pynguin baseline on Python benchmark cases.

This script is intentionally separate from the GDR pipeline. It generates tests
with Pynguin, then evaluates those tests with the same sandbox/result models so
the summaries can be compared with LLM pipeline runs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SandboxConfig  # noqa: E402
from src.evaluation.benchmarks import get_dataset  # noqa: E402
from src.evaluation.models import EvalMetrics, EvalResult  # noqa: E402
from src.evaluation.quality import (  # noqa: E402
    compute_coverage_metrics,
    compute_mutation_summary,
    compute_oracle_metrics,
    compute_relevance_metrics,
    generate_python_mutants,
)
from src.verification.sandbox import SandboxExecutor  # noqa: E402


def _pynguin_command() -> list[str]:
    exe = shutil.which("pynguin")
    if exe:
        return [exe]
    return [sys.executable, "-m", "pynguin"]


def _generated_tests(output_dir: Path) -> str:
    files = sorted(output_dir.rglob("test*.py"))
    if not files:
        files = sorted(output_dir.rglob("*.py"))
    return "\n\n".join(path.read_text(encoding="utf-8", errors="replace") for path in files)


def _timeout_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination for Pynguin and any surviving worker children."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001 - fallback for platforms without process groups
        proc.kill()


def _run_pynguin_command(
    cmd: list[str],
    env: dict[str, str],
    timeout_seconds: int,
) -> tuple[int | None, str, str, bool]:
    """Run Pynguin with a hard timeout that kills the whole process tree."""
    popen_kwargs: dict[str, Any] = {
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return proc.returncode, stdout or "", stderr or "", False
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout = _timeout_text(exc.stdout)
            stderr = _timeout_text(exc.stderr)
        return None, stdout or _timeout_text(exc.stdout), stderr or _timeout_text(exc.stderr), True


def _build_quality_payload(
    quality: str,
    case,
    test_code: str,
    sandbox_result,
    mutation_max_mutants: int = 25,
    sandbox_config: SandboxConfig | None = None,
) -> dict[str, Any]:
    """Build quality metrics dict matching BenchmarkRunner EvalResult fields."""
    if quality == "off":
        return {}

    source_code = case.code
    metadata = case.metadata if hasattr(case, "metadata") else {}

    coverage_data = None
    line_coverage = None
    branch_coverage = None
    if sandbox_result is not None:
        coverage_data = getattr(sandbox_result, "coverage_data", None)
        line_coverage = getattr(sandbox_result, "coverage", None)
        branch_coverage = getattr(sandbox_result, "branch_coverage", None)

    coverage_metrics = compute_coverage_metrics(
        coverage_data=coverage_data,
        source_code=source_code,
        metadata=metadata,
        line_coverage=line_coverage,
        source_file_path=metadata.get("target_file") or metadata.get("code_file"),
    )
    if branch_coverage is not None:
        coverage_metrics.setdefault("branch_coverage", branch_coverage)
        coverage_metrics.setdefault("target_branch_coverage", branch_coverage)

    oracle_metrics = compute_oracle_metrics(test_code)

    passed = sandbox_result.success if sandbox_result else False
    relevance_metrics = compute_relevance_metrics(
        test_code=test_code,
        source_code=source_code,
        metadata=metadata,
        passed=passed,
        coverage_data=coverage_data,
        source_file_path=metadata.get("target_file") or metadata.get("code_file"),
    )

    payload = {
        "coverage_metrics": coverage_metrics,
        "oracle_metrics": oracle_metrics,
        "relevance_metrics": relevance_metrics,
    }

    if quality == "full":
        mutation_metrics = {}
        case_language = getattr(case, "language", "python") or "python"
        if case_language.lower() == "python":
            mutants = generate_python_mutants(source_code, limit=mutation_max_mutants)
            if passed and test_code and mutants:
                mutation_metrics = _run_mutation_metrics(mutants, test_code, sandbox_config)
            else:
                mutation_metrics = compute_mutation_summary(mutants)
        payload["mutation_metrics"] = mutation_metrics

    return payload


def _run_mutation_metrics(
    mutants: list[dict[str, Any]],
    test_code: str,
    sandbox_config: SandboxConfig | None = None,
) -> dict[str, Any]:
    """Run generated tests against each mutant; True = killed."""
    if sandbox_config is None:
        sandbox_config = SandboxConfig.from_env()
    outcomes = []
    errors = []
    executor = SandboxExecutor(sandbox_config)
    for mutant in mutants:
        source = mutant.get("source_code")
        if not isinstance(source, str):
            errors.append(f"{mutant.get('id', '<unknown>')}: missing source")
            continue
        try:
            result = executor.execute(source_code=source, test_code=test_code)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{mutant.get('id', '<unknown>')}: {exc}")
            continue
        outcomes.append(not result.success)
    summary = compute_mutation_summary(mutants, outcomes)
    if errors:
        summary["errors"] = errors
    return summary


def _run_case(
    case,
    sandbox_config: SandboxConfig,
    seconds: int,
    quality: str = "off",
    mutation_max_mutants: int = 25,
) -> EvalResult:
    start = time.time()
    if case.language != "python":
        return EvalResult(case_id=case.id, error="Pynguin baseline supports Python only")

    with tempfile.TemporaryDirectory(prefix="pynguin-baseline-") as tmp:
        project_dir = Path(tmp) / "project"
        output_dir = Path(tmp) / "generated"
        project_dir.mkdir()
        output_dir.mkdir()
        (project_dir / "source_module.py").write_text(case.code, encoding="utf-8")

        cmd = [
            *_pynguin_command(),
            "--project-path",
            str(project_dir),
            "--output-path",
            str(output_dir),
            "--module-name",
            "source_module",
            "--maximum-search-time",
            str(seconds),
        ]
        env = dict(os.environ)
        env.setdefault("PYNGUIN_DANGER_AWARE", "YES")
        timeout_seconds = seconds + 30
        returncode, stdout, stderr, timed_out = _run_pynguin_command(
            cmd,
            env,
            timeout_seconds,
        )

        test_code = _generated_tests(output_dir)
        if not test_code.strip():
            if timed_out:
                error = (
                    f"Pynguin timed out after {timeout_seconds}s "
                    f"(maximum-search-time={seconds}s)"
                )
                detail = (stderr.strip() or stdout.strip())[:300]
                if detail:
                    error = f"{error}: {detail}"
            else:
                error = stderr.strip() or stdout.strip() or "no tests generated"
            return EvalResult(
                case_id=case.id,
                error=error[:500],
                elapsed_seconds=round(time.time() - start, 2),
                pipeline_state={
                    "baseline": "pynguin",
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "timeout_seconds": timeout_seconds if timed_out else None,
                },
            )

        sandbox = SandboxExecutor(sandbox_config)
        result = sandbox.execute(source_code=case.code, test_code=test_code)

        eval_kwargs = dict(
            case_id=case.id,
            passed=result.success,
            elapsed_seconds=round(time.time() - start, 2),
            tests_run=result.tests_run,
            tests_passed=result.tests_passed,
            coverage=result.coverage,
        )

        if quality != "off" and test_code.strip():
            quality_payload = _build_quality_payload(
                quality=quality,
                case=case,
                test_code=test_code,
                sandbox_result=result,
                mutation_max_mutants=mutation_max_mutants,
                sandbox_config=sandbox_config,
            )
            for key in (
                "coverage_metrics",
                "oracle_metrics",
                "relevance_metrics",
                "mutation_metrics",
            ):
                if key in quality_payload:
                    eval_kwargs[key] = quality_payload[key]

        return EvalResult(
            **eval_kwargs,
            pipeline_state={
                "baseline": "pynguin",
                "returncode": returncode,
                "timed_out": timed_out,
                "timeout_seconds": timeout_seconds if timed_out else None,
            },
            error=None if result.success else result.stderr[:500],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="testgeneval_lite")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--data-dir", default="data/benchmarks")
    parser.add_argument("--output-dir", default="eval_results_pynguin")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument(
        "--quality",
        choices=["off", "fast", "full"],
        default="off",
        help="Quality metrics mode: off (no extra metrics), fast (coverage+oracle+relevance), full (fast + mutation testing)",
    )
    parser.add_argument(
        "--mutation-max-mutants",
        type=int,
        default=25,
        help="Upper bound on mutants generated in full quality mode (default: 25)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-case JSON results and continue with missing cases",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Run only cases whose existing result is failed or missing",
    )
    args = parser.parse_args()

    dataset = get_dataset(args.benchmark, data_dir=Path(args.data_dir))
    cases = [case for case in dataset.load() if case.language == "python"][: args.max_cases]
    result_dir = Path(args.output_dir) / args.benchmark
    result_dir.mkdir(parents=True, exist_ok=True)
    sandbox_config = SandboxConfig.from_env()

    results = []
    for case in cases:
        result_path = result_dir / f"{case.id}.json"
        if result_path.is_file() and (args.skip_existing or args.failed_only):
            existing = EvalResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            if args.skip_existing or (args.failed_only and existing.passed):
                results.append(existing)
                print(f"[skip] {case.id}")
                continue

        print(f"[run] {case.id}", flush=True)
        result = _run_case(
            case,
            sandbox_config,
            args.seconds,
            quality=args.quality,
            mutation_max_mutants=args.mutation_max_mutants,
        )
        results.append(result)
        result_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )

    metrics = EvalMetrics.from_results(results, dataset_name=f"pynguin/{args.benchmark}")
    (result_dir / "summary.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    (result_dir / "summary.md").write_text(metrics.to_markdown(), encoding="utf-8")
    print(metrics.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
