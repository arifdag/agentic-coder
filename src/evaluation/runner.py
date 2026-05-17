"""BenchmarkRunner: feeds datasets through the GDR pipeline and collects results."""

import logging
import time
from pathlib import Path
from typing import Any, List, Optional

from .models import BenchmarkCase, EvalMetrics, EvalResult, _number
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


def _extract_numeric_coverage(value: Any, key_hint: str = "") -> Optional[float]:
    key_lower = key_hint.lower()
    if "coverage" in key_lower or "cov" in key_lower or "percent" in key_lower:
        num = _number(value)
        if num is not None:
            return num
    if isinstance(value, dict):
        for key, child in value.items():
            found = _extract_numeric_coverage(child, str(key))
            if found is not None:
                return found
        for child in value.values():
            found = _extract_numeric_coverage(child)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for child in value:
            found = _extract_numeric_coverage(child, key_hint)
            if found is not None:
                return found
    return _number(value)


def _extract_baseline_coverage(metadata: dict[str, Any]) -> Optional[float]:
    """Conservatively extract a numeric baseline coverage value from metadata."""
    raw = metadata.get("baseline_covs")
    if raw is None:
        return None
    return _extract_numeric_coverage(raw, "baseline_covs")


class BenchmarkRunner:
    """Run the GDR pipeline against every case in a benchmark dataset."""

    def __init__(self, config, dataset, results_dir: str = "eval_results"):
        self.config = config
        self.dataset = dataset
        self.results_dir = Path(results_dir) / dataset.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._results: List[EvalResult] = []

    def run(
        self,
        max_cases: Optional[int] = None,
        *,
        skip_existing: bool = False,
        failed_only: bool = False,
    ) -> List[EvalResult]:
        from ..graph.pipeline import run_pipeline

        cases = self.dataset.load()
        if max_cases is not None:
            cases = cases[:max_cases]

        log.info("Running %d cases from %s", len(cases), self.dataset.name)

        for i, case in enumerate(cases):
            result_path = self.results_dir / f"{case.id}.json"
            existing = self._load_existing_result(result_path)
            if existing is not None and (skip_existing or (failed_only and existing.passed)):
                log.info("[%d/%d] %s already has a result; reusing it", i + 1, len(cases), case.id)
                self._results.append(existing)
                continue

            log.info("[%d/%d] %s", i + 1, len(cases), case.id)
            result = self._run_one(case, run_pipeline)
            self._results.append(result)

            result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        return self._results

    def _load_existing_result(self, path: Path) -> EvalResult | None:
        if not path.is_file():
            return None
        try:
            return EvalResult.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - bad cache should not block reruns
            log.warning("Ignoring unreadable existing result %s: %s", path, exc)
            return None

    def _benchmark_oracle_pass(
        self,
        case: BenchmarkCase,
        gates: list[dict],
        pipeline_passed: bool,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Apply benchmark-specific pass semantics when raw pipeline pass is misleading."""
        if getattr(self.dataset, "name", None) != "dep_hallucination":
            return pipeline_passed, None

        dep_gate = next((g for g in gates if g.get("gate_name") == "dependency"), None)
        dependency_passed = None if dep_gate is None else bool(dep_gate.get("passed"))
        phantom_packages = case.metadata.get("phantom_packages") or []
        valid_packages = case.metadata.get("valid_packages") or []
        expected_phantom = bool(phantom_packages)
        expected_clean = bool(valid_packages) and not expected_phantom
        if expected_phantom:
            passed = dependency_passed is False
            reason = "phantom_detected" if passed else "phantom_missed"
        elif expected_clean:
            passed = dependency_passed is True
            reason = "clean_accepted" if passed else "clean_rejected"
        else:
            passed = pipeline_passed
            reason = "no_dependency_oracle"

        return passed, {
            "benchmark": "dep_hallucination",
            "reason": reason,
            "pipeline_passed": pipeline_passed,
            "dependency_passed": dependency_passed,
            "expected_phantom": expected_phantom,
            "expected_clean": expected_clean,
        }

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
                repo_metadata=case.metadata,
            )
            elapsed = time.time() - start

            report = state.get("verification_report") or {}
            gates = report.get("gates", [])
            coverage_val = report.get("coverage")
            pipeline_passed = state.get("status") == "success"
            benchmark_passed, benchmark_oracle = self._benchmark_oracle_pass(
                case,
                gates,
                pipeline_passed,
            )

            sb_run = state.get("sandbox_tests_run")
            sb_pass = state.get("sandbox_tests_passed")
            declared = len(state.get("test_functions") or [])
            tests_run = sb_run if sb_run is not None else declared
            if sb_pass is not None:
                tests_passed = sb_pass
            else:
                tests_passed = declared if state.get("status") == "success" else 0

            quality_payload = self._build_quality_payload(case, state, gates, coverage_val)

            execution_context = state.get("execution_context") or "single-file"
            repo_meta = case.metadata
            eligible = bool(
                repo_meta.get("execution_context") == "repo"
                or repo_meta.get("project_root")
                or repo_meta.get("repo")
            )
            infrastructure_pass = state.get("infrastructure_pass")
            repo_setup_pass = state.get("repo_setup_pass")
            execution_metrics = {
                "execution_context": execution_context,
                "eligible": eligible,
                "infrastructure_pass": infrastructure_pass if eligible else None,
                "repo_setup_pass": repo_setup_pass if eligible else None,
                "infrastructure_failure_reason": (
                    state.get("error_message") if infrastructure_pass is False else None
                ),
            }

            return EvalResult(
                case_id=case.id,
                passed=benchmark_passed,
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
                    "pipeline_passed": pipeline_passed,
                    "benchmark_oracle": benchmark_oracle,
                },
                execution_metrics=execution_metrics,
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
        source_file_path = case.metadata.get("target_file") or case.metadata.get("code_file")
        coverage_metrics = compute_coverage_metrics(
            state.get("sandbox_coverage_data"),
            case.code,
            case.metadata,
            line_coverage=coverage_val,
            source_file_path=source_file_path,
        )
        if state.get("sandbox_branch_coverage") is not None:
            coverage_metrics.setdefault("branch_coverage", state.get("sandbox_branch_coverage"))
            coverage_metrics.setdefault(
                "target_branch_coverage", state.get("sandbox_branch_coverage")
            )

        baseline_cov = _extract_baseline_coverage(case.metadata)
        if baseline_cov is not None and coverage_metrics.get("line_coverage") is not None:
            gain = coverage_metrics["line_coverage"] - baseline_cov
            coverage_metrics["coverage_gain"] = round(gain, 2)

        gate_metrics = compute_gate_quality_metrics(gates, case.metadata)
        sandbox_passed = next(
            (bool(g.get("passed")) for g in gates if g.get("gate_name") == "sandbox"),
            passed,
        )
        relevance_metrics = compute_relevance_metrics(
            test_code,
            case.code,
            case.metadata,
            passed=sandbox_passed,
            coverage_data=state.get("sandbox_coverage_data"),
            source_file_path=source_file_path,
        )
        mutation_metrics: dict[str, Any] = {}
        if mode == "full" and case.language.lower() == "python":
            if not passed:
                mutation_metrics = {"skipped": True, "skip_reason": "pipeline_failed"}
            elif not sandbox_passed:
                mutation_metrics = {"skipped": True, "skip_reason": "sandbox_failed"}
            elif not relevance_metrics.get("relevance_pass", False):
                mutation_metrics = {"skipped": True, "skip_reason": "relevance_failed"}
            elif not test_code:
                mutation_metrics = {"skipped": True, "skip_reason": "no_tests"}
            else:
                limit = getattr(self.config.evaluation, "mutation_max_mutants", 25)
                mutants = generate_python_mutants(case.code, limit=limit)
                mutation_metrics = self._run_mutation_metrics(mutants, test_code)

        reliability_metrics: dict[str, Any] = {}
        repeats = getattr(self.config.evaluation, "reliability_repeats", 0)
        if mode == "full" and repeats > 0 and case.language.lower() == "python" and test_code:
            reliability_metrics = self._run_reliability_metrics(case.code, test_code, repeats)

        bug_metrics: dict[str, Any] = {}
        buggy_code = case.metadata.get("buggy_code")
        if buggy_code:
            bug_metrics = self._run_bug_detection(case, test_code, buggy_code, sandbox_passed)
        elif case.metadata.get("fixed_code"):
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
            "relevance_metrics": relevance_metrics,
            "safety_metrics": gate_metrics["safety_metrics"],
            "dependency_metrics": gate_metrics["dependency_metrics"],
            "bug_metrics": bug_metrics,
        }

    def _run_bug_detection(
        self,
        case: BenchmarkCase,
        test_code: str,
        buggy_code: str,
        fixed_passed: bool,
    ) -> dict[str, Any]:
        """Run generated tests against buggy code. Non-blocking; does not change case-pass."""
        from ..verification.sandbox import SandboxExecutor

        metrics: dict[str, Any] = {
            "eligible": True,
            "fixed_passed": fixed_passed,
            "buggy_failed": None,
            "bug_detected": None,
            "errors": [],
        }
        if not fixed_passed or not test_code:
            return metrics

        executor = SandboxExecutor(self.config.sandbox)
        try:
            result = executor.execute(source_code=buggy_code, test_code=test_code)
        except Exception as exc:  # noqa: BLE001 - non-blocking metric collection
            metrics["errors"].append(str(exc))
            return metrics

        buggy_failed = not result.success
        metrics["buggy_failed"] = buggy_failed
        metrics["bug_detected"] = buggy_failed
        return metrics

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
