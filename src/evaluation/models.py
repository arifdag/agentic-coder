"""Data models for Phase 6 evaluation and ablation workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    """Single benchmark input case."""

    id: str
    code: str
    language: str
    expected_tests: Optional[List[str]] = None
    expected_cwe: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user_request: Optional[str] = None


class EvalResult(BaseModel):
    """Result of running the pipeline on a single BenchmarkCase."""

    case_id: str
    passed: bool = False
    elapsed_seconds: float = 0.0
    tests_run: int = 0
    tests_passed: int = 0
    coverage: Optional[float] = None
    iterations: int = 1
    gate_results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    pipeline_state: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)
    coverage_metrics: Dict[str, Any] = Field(default_factory=dict)
    mutation_metrics: Dict[str, Any] = Field(default_factory=dict)
    reliability_metrics: Dict[str, Any] = Field(default_factory=dict)
    oracle_metrics: Dict[str, Any] = Field(default_factory=dict)
    relevance_metrics: Dict[str, Any] = Field(default_factory=dict)
    safety_metrics: Dict[str, Any] = Field(default_factory=dict)
    dependency_metrics: Dict[str, Any] = Field(default_factory=dict)
    bug_metrics: Dict[str, Any] = Field(default_factory=dict)


_RATE_LIMIT_HINTS = (
    "429",
    "rate_limit",
    "rate limit",
    "tokens per day",
    "tpd",
    "per minute",
)


def _is_provider_error(result: EvalResult) -> bool:
    """Return True when a case failed because an upstream provider was unavailable."""
    quality_flag = result.quality.get("provider_error")
    if quality_flag is True:
        return True
    msg = (result.error or result.pipeline_state.get("error_message") or "").lower()
    return bool(msg) and any(hint in msg for hint in _RATE_LIMIT_HINTS)


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


class EvalMetrics(BaseModel):
    """Aggregated metrics for a benchmark run."""

    dataset_name: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    pass_rate: float = 0.0
    # Aggregate test-level metrics. ``test_pass_rate`` = pooled
    # tests_passed / tests_run across all cases. This is the metric to
    # report for multi-file projects where the all-or-nothing case-level
    # ``pass_rate`` collapses 80%-passing runs to "0% passed".
    total_tests_run: int = 0
    total_tests_passed: int = 0
    test_pass_rate: float = 0.0
    avg_coverage: Optional[float] = None
    avg_iterations: float = 0.0
    avg_time: float = 0.0
    gate_pass_rates: Dict[str, float] = Field(default_factory=dict)
    # Research quality aggregate fields.
    clean_total: int = 0
    provider_error_count: int = 0
    clean_pass_rate: Optional[float] = None
    avg_line_coverage: Optional[float] = None
    avg_branch_coverage: Optional[float] = None
    avg_target_line_coverage: Optional[float] = None
    avg_target_branch_coverage: Optional[float] = None
    mutation_score: Optional[float] = None
    mutation_coverage: Optional[float] = None
    bug_detection_rate: Optional[float] = None
    relevance_pass_rate: Optional[float] = None
    gaming_rate: Optional[float] = None
    sast_catch_rate: Optional[float] = None
    vulnerability_escape_rate: Optional[float] = None
    dependency_phantom_detection_rate: Optional[float] = None
    dependency_clean_acceptance_rate: Optional[float] = None
    reliability_rate: Optional[float] = None
    flakiness_rate: Optional[float] = None
    assertion_presence_rate: Optional[float] = None
    avg_assertions_per_test: Optional[float] = None
    avg_coverage_gain: Optional[float] = None

    @classmethod
    def from_results(cls, results: List[EvalResult], dataset_name: str = "") -> "EvalMetrics":
        total = len(results)
        if total == 0:
            return cls(dataset_name=dataset_name)

        passed = sum(1 for r in results if r.passed)
        errored = sum(1 for r in results if r.error)
        failed = total - passed

        total_tests_run = sum(r.tests_run for r in results)
        total_tests_passed = sum(r.tests_passed for r in results)
        test_pass_rate = total_tests_passed / total_tests_run if total_tests_run else 0.0

        coverages = [r.coverage for r in results if r.coverage is not None]
        avg_cov = sum(coverages) / len(coverages) if coverages else None

        avg_iter = sum(r.iterations for r in results) / total
        avg_time = sum(r.elapsed_seconds for r in results) / total

        gate_counts: Dict[str, List[bool]] = {}
        for r in results:
            for g in r.gate_results:
                name = g.get("gate_name", "unknown")
                gate_counts.setdefault(name, []).append(g.get("passed", False))
        gate_rates = {name: sum(vals) / len(vals) for name, vals in gate_counts.items()}

        provider_error_count = sum(1 for r in results if _is_provider_error(r))
        clean_results = [r for r in results if not _is_provider_error(r)]
        clean_total = len(clean_results)
        clean_pass_rate = _rate(sum(1 for r in clean_results if r.passed), clean_total)

        def coverage_value(result: EvalResult, key: str) -> Optional[float]:
            val = _number(result.coverage_metrics.get(key))
            if val is not None:
                return val
            if key == "line_coverage":
                return _number(result.coverage)
            return None

        line_coverages = [
            v for r in results if (v := coverage_value(r, "line_coverage")) is not None
        ]
        branch_coverages = [
            v for r in results if (v := coverage_value(r, "branch_coverage")) is not None
        ]
        target_line_coverages = [
            v for r in results if (v := coverage_value(r, "target_line_coverage")) is not None
        ]
        target_branch_coverages = [
            v for r in results if (v := coverage_value(r, "target_branch_coverage")) is not None
        ]

        mutants_killed = mutants_survived = mutants_uncovered = 0
        explicit_mut_scores: List[float] = []
        explicit_mut_covs: List[float] = []
        for r in results:
            mm = r.mutation_metrics
            killed = int(_number(mm.get("mutants_killed") or mm.get("killed")) or 0)
            survived = int(_number(mm.get("mutants_survived") or mm.get("survived")) or 0)
            uncovered = int(_number(mm.get("mutants_uncovered") or mm.get("uncovered")) or 0)
            mutants_killed += killed
            mutants_survived += survived
            mutants_uncovered += uncovered
            score = _number(mm.get("mutation_score"))
            if score is not None:
                explicit_mut_scores.append(score)
            cov = _number(mm.get("mutation_coverage"))
            if cov is not None:
                explicit_mut_covs.append(cov)

        mutation_den = mutants_killed + mutants_survived
        mutation_score = (
            mutants_killed / mutation_den if mutation_den else _mean(explicit_mut_scores)
        )
        mutation_cov_den = mutants_killed + mutants_survived + mutants_uncovered
        mutation_coverage = (
            (mutants_killed + mutants_survived) / mutation_cov_den
            if mutation_cov_den
            else _mean(explicit_mut_covs)
        )

        bug_eligible = [
            r for r in results if r.bug_metrics.get("eligible") or "bug_detected" in r.bug_metrics
        ]
        bug_detection_rate = _rate(
            sum(1 for r in bug_eligible if bool(r.bug_metrics.get("bug_detected"))),
            len(bug_eligible),
        )

        relevance_eligible = [
            r
            for r in results
            if r.relevance_metrics.get("eligible", True)
            and ("relevance_pass" in r.relevance_metrics or "gaming_flag" in r.relevance_metrics)
        ]
        relevance_pass_rate = _rate(
            sum(1 for r in relevance_eligible if bool(r.relevance_metrics.get("relevance_pass"))),
            len(relevance_eligible),
        )
        gaming_rate = _rate(
            sum(1 for r in relevance_eligible if bool(r.relevance_metrics.get("gaming_flag"))),
            len(relevance_eligible),
        )

        safety_eligible = [r for r in results if bool(r.safety_metrics.get("expected_vulnerable"))]
        sast_catch_rate = _rate(
            sum(1 for r in safety_eligible if bool(r.safety_metrics.get("vulnerability_detected"))),
            len(safety_eligible),
        )
        vulnerability_escape_rate = 1.0 - sast_catch_rate if sast_catch_rate is not None else None

        phantom_eligible = [
            r for r in results if bool(r.dependency_metrics.get("expected_phantom"))
        ]
        dependency_phantom_detection_rate = _rate(
            sum(1 for r in phantom_eligible if bool(r.dependency_metrics.get("phantom_detected"))),
            len(phantom_eligible),
        )
        clean_dep_eligible = [
            r for r in results if bool(r.dependency_metrics.get("expected_clean"))
        ]
        dependency_clean_acceptance_rate = _rate(
            sum(1 for r in clean_dep_eligible if bool(r.dependency_metrics.get("clean_accepted"))),
            len(clean_dep_eligible),
        )

        reliability_eligible = [
            r
            for r in results
            if "reliable" in r.reliability_metrics or "flaky" in r.reliability_metrics
        ]
        reliability_rate = _rate(
            sum(1 for r in reliability_eligible if bool(r.reliability_metrics.get("reliable"))),
            len(reliability_eligible),
        )
        flakiness_rate = _rate(
            sum(1 for r in reliability_eligible if bool(r.reliability_metrics.get("flaky"))),
            len(reliability_eligible),
        )

        oracle_eligible = [
            r
            for r in results
            if "assertion_count" in r.oracle_metrics or "has_assertions" in r.oracle_metrics
        ]
        assertion_presence_rate = _rate(
            sum(1 for r in oracle_eligible if bool(r.oracle_metrics.get("has_assertions"))),
            len(oracle_eligible),
        )
        total_assertions = sum(
            int(_number(r.oracle_metrics.get("assertion_count")) or 0) for r in oracle_eligible
        )
        total_oracle_tests = sum(
            int(_number(r.oracle_metrics.get("test_count")) or 0) for r in oracle_eligible
        )
        avg_assertions_per_test = _rate(total_assertions, total_oracle_tests)

        coverage_gains = [
            v
            for r in results
            if (v := _number(r.coverage_metrics.get("coverage_gain"))) is not None
        ]
        avg_coverage_gain = _mean(coverage_gains)

        return cls(
            dataset_name=dataset_name,
            total=total,
            passed=passed,
            failed=failed,
            errored=errored,
            pass_rate=passed / total,
            total_tests_run=total_tests_run,
            total_tests_passed=total_tests_passed,
            test_pass_rate=test_pass_rate,
            avg_coverage=avg_cov,
            avg_iterations=avg_iter,
            avg_time=avg_time,
            gate_pass_rates=gate_rates,
            clean_total=clean_total,
            provider_error_count=provider_error_count,
            clean_pass_rate=clean_pass_rate,
            avg_line_coverage=_mean(line_coverages),
            avg_branch_coverage=_mean(branch_coverages),
            avg_target_line_coverage=_mean(target_line_coverages),
            avg_target_branch_coverage=_mean(target_branch_coverages),
            mutation_score=mutation_score,
            mutation_coverage=mutation_coverage,
            bug_detection_rate=bug_detection_rate,
            relevance_pass_rate=relevance_pass_rate,
            gaming_rate=gaming_rate,
            sast_catch_rate=sast_catch_rate,
            vulnerability_escape_rate=vulnerability_escape_rate,
            dependency_phantom_detection_rate=dependency_phantom_detection_rate,
            dependency_clean_acceptance_rate=dependency_clean_acceptance_rate,
            reliability_rate=reliability_rate,
            flakiness_rate=flakiness_rate,
            assertion_presence_rate=assertion_presence_rate,
            avg_assertions_per_test=avg_assertions_per_test,
            avg_coverage_gain=avg_coverage_gain,
        )

    def to_markdown(self) -> str:
        lines = [
            f"## {self.dataset_name or 'Evaluation'} Results\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total cases | {self.total} |",
            f"| Passed | {self.passed} |",
            f"| Failed | {self.failed} |",
            f"| Errored | {self.errored} |",
            f"| Pass rate (case-level) | {self.pass_rate:.1%} |",
        ]
        if self.total_tests_run:
            lines.append(
                f"| Tests passed | {self.total_tests_passed}/{self.total_tests_run} "
                f"({self.test_pass_rate:.1%}) |"
            )
        if self.avg_coverage is not None:
            lines.append(f"| Avg coverage | {self.avg_coverage:.1f}% |")
        if self.provider_error_count:
            lines.append(f"| Provider errors | {self.provider_error_count} |")
        if self.clean_pass_rate is not None and self.clean_total != self.total:
            lines.append(
                f"| Clean pass rate | {self.clean_pass_rate:.1%} ({self.clean_total} cases) |"
            )
        if self.avg_branch_coverage is not None:
            lines.append(f"| Avg branch coverage | {self.avg_branch_coverage:.1f}% |")
        if self.avg_target_line_coverage is not None:
            lines.append(f"| Avg target line coverage | {self.avg_target_line_coverage:.1f}% |")
        if self.avg_target_branch_coverage is not None:
            lines.append(f"| Avg target branch coverage | {self.avg_target_branch_coverage:.1f}% |")
        if self.mutation_score is not None:
            lines.append(f"| Mutation score | {self.mutation_score:.1%} |")
        if self.mutation_coverage is not None:
            lines.append(f"| Mutation coverage | {self.mutation_coverage:.1%} |")
        if self.bug_detection_rate is not None:
            lines.append(f"| Bug detection rate | {self.bug_detection_rate:.1%} |")
        if self.relevance_pass_rate is not None:
            lines.append(f"| Relevance pass rate | {self.relevance_pass_rate:.1%} |")
        if self.gaming_rate is not None:
            lines.append(f"| Gaming rate | {self.gaming_rate:.1%} |")
        if self.reliability_rate is not None:
            lines.append(f"| Reliability rate | {self.reliability_rate:.1%} |")
        if self.assertion_presence_rate is not None:
            lines.append(f"| Assertion presence | {self.assertion_presence_rate:.1%} |")
        if self.avg_assertions_per_test is not None:
            lines.append(f"| Avg assertions/test | {self.avg_assertions_per_test:.2f} |")
        if self.avg_coverage_gain is not None:
            lines.append(f"| Avg coverage gain | {self.avg_coverage_gain:+.2f} |")
        lines.append(f"| Avg iterations | {self.avg_iterations:.2f} |")
        lines.append(f"| Avg time (s) | {self.avg_time:.2f} |")

        if self.gate_pass_rates:
            lines.append("")
            lines.append("### Per-gate pass rates\n")
            lines.append("| Gate | Pass rate |")
            lines.append("|------|-----------|")
            for name, rate in sorted(self.gate_pass_rates.items()):
                lines.append(f"| {name} | {rate:.1%} |")

        return "\n".join(lines)


class AblationConfig(BaseModel):
    """Describes one ablation variant."""

    name: str
    sast_enabled: bool = True
    dependency_enabled: bool = True
    judge_enabled: bool = True
    retry_budget: int = 3
    description: str = ""


@runtime_checkable
class BenchmarkDataset(Protocol):
    """Protocol that all benchmark loaders must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def language(self) -> Optional[str]: ...

    def load(self) -> List[BenchmarkCase]: ...
