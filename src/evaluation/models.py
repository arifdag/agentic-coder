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


class EvalMetrics(BaseModel):
    """Aggregated metrics for a benchmark run."""

    dataset_name: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    pass_rate: float = 0.0
    avg_coverage: Optional[float] = None
    avg_iterations: float = 0.0
    avg_time: float = 0.0
    gate_pass_rates: Dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_results(cls, results: List[EvalResult], dataset_name: str = "") -> "EvalMetrics":
        total = len(results)
        if total == 0:
            return cls(dataset_name=dataset_name)

        passed = sum(1 for r in results if r.passed)
        errored = sum(1 for r in results if r.error)
        failed = total - passed

        coverages = [r.coverage for r in results if r.coverage is not None]
        avg_cov = sum(coverages) / len(coverages) if coverages else None

        avg_iter = sum(r.iterations for r in results) / total
        avg_time = sum(r.elapsed_seconds for r in results) / total

        gate_counts: Dict[str, List[bool]] = {}
        for r in results:
            for g in r.gate_results:
                name = g.get("gate_name", "unknown")
                gate_counts.setdefault(name, []).append(g.get("passed", False))
        gate_rates = {
            name: sum(vals) / len(vals) for name, vals in gate_counts.items()
        }

        return cls(
            dataset_name=dataset_name,
            total=total,
            passed=passed,
            failed=failed,
            errored=errored,
            pass_rate=passed / total,
            avg_coverage=avg_cov,
            avg_iterations=avg_iter,
            avg_time=avg_time,
            gate_pass_rates=gate_rates,
        )

    def to_markdown(self) -> str:
        lines = [
            f"## {self.dataset_name or 'Evaluation'} Results\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total cases | {self.total} |",
            f"| Passed | {self.passed} |",
            f"| Failed | {self.failed} |",
            f"| Errored | {self.errored} |",
            f"| Pass rate | {self.pass_rate:.1%} |",
        ]
        if self.avg_coverage is not None:
            lines.append(f"| Avg coverage | {self.avg_coverage:.1f}% |")
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
