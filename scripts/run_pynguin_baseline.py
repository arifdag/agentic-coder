"""Run an optional Pynguin baseline on Python benchmark cases.

This script is intentionally separate from the GDR pipeline. It generates tests
with Pynguin, then evaluates those tests with the same sandbox/result models so
the summaries can be compared with LLM pipeline runs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SandboxConfig  # noqa: E402
from src.evaluation.benchmarks import get_dataset  # noqa: E402
from src.evaluation.models import EvalMetrics, EvalResult  # noqa: E402
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


def _run_case(case, sandbox_config: SandboxConfig, seconds: int) -> EvalResult:
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
        completed = subprocess.run(
            cmd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=seconds + 30,
        )
        test_code = _generated_tests(output_dir)
        if not test_code.strip():
            error = completed.stderr.strip() or completed.stdout.strip() or "no tests generated"
            return EvalResult(
                case_id=case.id,
                error=error[:500],
                elapsed_seconds=round(time.time() - start, 2),
            )

        sandbox = SandboxExecutor(sandbox_config)
        result = sandbox.execute(source_code=case.code, test_code=test_code)
        return EvalResult(
            case_id=case.id,
            passed=result.success,
            elapsed_seconds=round(time.time() - start, 2),
            tests_run=result.tests_run,
            tests_passed=result.tests_passed,
            coverage=result.coverage,
            pipeline_state={"baseline": "pynguin", "returncode": completed.returncode},
            error=None if result.success else result.stderr[:500],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="testgeneval_lite")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--data-dir", default="data/benchmarks")
    parser.add_argument("--output-dir", default="eval_results_pynguin")
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()

    dataset = get_dataset(args.benchmark, data_dir=Path(args.data_dir))
    cases = [case for case in dataset.load() if case.language == "python"][: args.max_cases]
    result_dir = Path(args.output_dir) / args.benchmark
    result_dir.mkdir(parents=True, exist_ok=True)
    sandbox_config = SandboxConfig.from_env()

    results: list[EvalResult] = []
    for case in cases:
        result = _run_case(case, sandbox_config, args.seconds)
        results.append(result)
        (result_dir / f"{case.id}.json").write_text(
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
