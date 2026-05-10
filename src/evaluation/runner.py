"""BenchmarkRunner: feeds datasets through the GDR pipeline and collects results."""

import logging
import time
from pathlib import Path
from typing import Any, List, Optional

from .models import BenchmarkCase, EvalMetrics, EvalResult
from .quality import (
    compute_coverage_metrics,
    compute_gate_quality_metrics,
    compute_mutation_summary,
    compute_oracle_metrics,
    compute_relevance_metrics,
    generate_python_mutants,
    infer_primary_target,
)

log = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run the GDR pipeline against every case in a benchmark dataset."""

    def __init__(self, config, dataset, results_dir: str = "eval_results"):
        self.config = config
        self.dataset = dataset
        self.results_dir = Path(results_dir) / dataset.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._results: List[EvalResult] = []

    def run(self, max_cases: Optional[int] = None) -> List[EvalResult]:
        from ..graph.pipeline import run_pipeline

        cases = self.dataset.load()
        if max_cases is not None:
            cases = cases[:max_cases]

        log.info("Running %d cases from %s", len(cases), self.dataset.name)

        for i, case in enumerate(cases):
            log.info("[%d/%d] %s", i + 1, len(cases), case.id)
            result = self._run_one(case, run_pipeline)
            self._results.append(result)

            result_path = self.results_dir / f"{case.id}.json"
            result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        return self._results

    def _run_one(self, case: BenchmarkCase, run_pipeline_fn) -> EvalResult:
        start = time.time()
        try:
            state = run_pipeline_fn(
                code=case.code,
                user_request=case.user_request or "Generate comprehensive unit tests",
                file_path=None,
                max_retries=self.config.pipeline.max_retries,
                config=self.config,
                target_function=infer_primary_target(case.code, case.metadata),
            )
            elapsed = time.time() - start

            report = state.get("verification_report") or {}
            gates = report.get("gates", [])
            coverage_val = report.get("coverage")

            # Prefer the *actual* pytest counts the sandbox parsed; fall
            # back to the LLM's declared test_functions only when the
            # sandbox didn't run (e.g. pre-sandbox failure). This makes
            # the per-case `tests_passed` field a real signal usable for
            # paper-grade metrics like "mean per-case test pass rate"
            # instead of an all-or-nothing flag that collapses an
            # 82-percent-passing case to "0 passed".
            sb_run = state.get("sandbox_tests_run")
            sb_pass = state.get("sandbox_tests_passed")
            declared = len(state.get("test_functions") or [])
            tests_run = sb_run if sb_run is not None else declared
            if sb_pass is not None:
                tests_passed = sb_pass
            else:
                tests_passed = declared if state.get("status") == "success" else 0

            quality_payload = self._build_quality_payload(case, state, gates, coverage_val)

            return EvalResult(
                case_id=case.id,
                passed=state.get("status") == "success",
                elapsed_seconds=round(elapsed, 2),
                tests_run=tests_run,
                tests_passed=tests_passed,
                coverage=coverage_val,
                iterations=state.get("retry_count", 0) + 1,
                gate_results=gates,
                pipeline_state={
                    "status": state.get("status"),
                    "task_type": state.get("task_type"),
                    "language": state.get("language"),
                    "error_type": state.get("error_type"),
                    "error_message": state.get("error_message"),
                },
                **quality_payload,
            )
        except Exception as exc:
            elapsed = time.time() - start
            log.error("Case %s failed: %s", case.id, exc)
            return EvalResult(
                case_id=case.id,
                passed=False,
                elapsed_seconds=round(elapsed, 2),
                error=str(exc),
            )

    def _build_quality_payload(
        self,
        case: BenchmarkCase,
        state: dict[str, Any],
        gates: list[dict],
        coverage_val: Optional[float],
    ) -> dict[str, Any]:
        """Build non-blocking quality metrics for a completed pipeline state."""
        mode = getattr(getattr(self.config, "evaluation", None), "quality_mode", "fast")
        if mode == "off":
            return {}

        test_code = state.get("generated_tests") or ""
        passed = state.get("status") == "success"
        coverage_metrics = compute_coverage_metrics(
            state.get("sandbox_coverage_data"),
            case.code,
            case.metadata,
            line_coverage=coverage_val,
        )
        if state.get("sandbox_branch_coverage") is not None:
            coverage_metrics.setdefault("branch_coverage", state.get("sandbox_branch_coverage"))
            coverage_metrics.setdefault(
                "target_branch_coverage", state.get("sandbox_branch_coverage")
            )

        gate_metrics = compute_gate_quality_metrics(gates, case.metadata)
        sandbox_passed = next(
            (bool(g.get("passed")) for g in gates if g.get("gate_name") == "sandbox"),
            passed,
        )
        mutation_metrics: dict[str, Any] = {}
        if mode == "full" and case.language.lower() == "python":
            limit = getattr(self.config.evaluation, "mutation_max_mutants", 25)
            mutants = generate_python_mutants(case.code, limit=limit)
            if passed and test_code and mutants:
                mutation_metrics = self._run_mutation_metrics(mutants, test_code)
            else:
                mutation_metrics = compute_mutation_summary(mutants)

        reliability_metrics: dict[str, Any] = {}
        repeats = getattr(self.config.evaluation, "reliability_repeats", 0)
        if mode == "full" and repeats > 0 and case.language.lower() == "python" and test_code:
            reliability_metrics = self._run_reliability_metrics(case.code, test_code, repeats)

        bug_metrics: dict[str, Any] = {}
        if case.metadata.get("buggy_code") or case.metadata.get("fixed_code"):
            bug_metrics = {"eligible": True, "bug_detected": None}

        return {
            "quality": {
                "mode": mode,
                "provider_error": False,
            },
            "coverage_metrics": coverage_metrics,
            "mutation_metrics": mutation_metrics,
            "reliability_metrics": reliability_metrics,
            "oracle_metrics": compute_oracle_metrics(test_code),
            "relevance_metrics": compute_relevance_metrics(
                test_code,
                case.code,
                case.metadata,
                passed=sandbox_passed,
                coverage_data=state.get("sandbox_coverage_data"),
            ),
            "safety_metrics": gate_metrics["safety_metrics"],
            "dependency_metrics": gate_metrics["dependency_metrics"],
            "bug_metrics": bug_metrics,
        }

    def _run_mutation_metrics(
        self,
        mutants: list[dict[str, Any]],
        test_code: str,
    ) -> dict[str, Any]:
        """Run generated tests against mutants; True outcome means killed."""
        from ..verification.sandbox import SandboxExecutor

        outcomes: list[bool] = []
        errors: list[str] = []
        executor = SandboxExecutor(self.config.sandbox)
        for mutant in mutants:
            source = mutant.get("source_code")
            if not isinstance(source, str):
                errors.append(f"{mutant.get('id', '<unknown>')}: missing source")
                continue
            try:
                result = executor.execute(source_code=source, test_code=test_code)
            except Exception as exc:  # noqa: BLE001 - non-blocking metric collection
                errors.append(f"{mutant.get('id', '<unknown>')}: {exc}")
                continue
            outcomes.append(not result.success)

        summary = compute_mutation_summary(mutants, outcomes)
        if errors:
            summary["errors"] = errors
        return summary

    def _run_reliability_metrics(
        self,
        source_code: str,
        test_code: str,
        repeats: int,
    ) -> dict[str, Any]:
        """Repeat a passing suite to estimate flakiness; does not affect case-pass."""
        from ..verification.sandbox import SandboxExecutor

        executor = SandboxExecutor(self.config.sandbox)
        outcomes: list[bool] = []
        errors: list[str] = []
        for i in range(max(0, repeats)):
            try:
                outcomes.append(
                    executor.execute(source_code=source_code, test_code=test_code).success
                )
            except Exception as exc:  # noqa: BLE001 - non-blocking metric collection
                outcomes.append(False)
                errors.append(f"repeat {i + 1}: {exc}")

        reliable = bool(outcomes) and all(outcomes)
        metrics: dict[str, Any] = {
            "repeats": len(outcomes),
            "passed_repeats": sum(1 for item in outcomes if item),
            "reliable": reliable,
            "flaky": bool(outcomes) and not reliable,
        }
        if errors:
            metrics["errors"] = errors
        return metrics

    def summarize(self) -> EvalMetrics:
        return EvalMetrics.from_results(self._results, dataset_name=self.dataset.name)

    def save_summary(self) -> Path:
        metrics = self.summarize()
        summary_path = self.results_dir / "summary.json"
        summary_path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")

        md_path = self.results_dir / "summary.md"
        md_path.write_text(metrics.to_markdown(), encoding="utf-8")

        log.info("Saved summary to %s", summary_path)
        return summary_path
