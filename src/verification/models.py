"""Shared models for the verification layer."""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity level for findings."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class JudgeVerdict(str, Enum):
    """Verdict from the LLM judge."""
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    UNCERTAIN = "uncertain"


class Finding(BaseModel):
    """A single issue found by a verification gate."""

    severity: Severity = Field(description="Severity level")
    code: Optional[str] = Field(default=None, description="CWE-ID or error code")
    message: str = Field(description="Human-readable description")
    line: Optional[int] = Field(default=None, description="Line number in the artifact")
    file: Optional[str] = Field(default=None, description="File where the issue was found")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix")
    judge_verdict: Optional[JudgeVerdict] = Field(
        default=None, description="LLM judge classification"
    )


class GateResult(BaseModel):
    """Result from a single verification gate."""

    gate_name: str = Field(description="Gate identifier: sast, dependency, sandbox")
    passed: bool = Field(description="Whether this gate passed")
    findings: List[Finding] = Field(default_factory=list, description="Issues found")
    details: Optional[str] = Field(default=None, description="Additional details or raw output")

    @property
    def error_findings(self) -> List[Finding]:
        """Return only error-severity findings not judged as false positives."""
        return [
            f for f in self.findings
            if f.severity == Severity.ERROR
            and f.judge_verdict != JudgeVerdict.FALSE_POSITIVE
        ]


class VerificationReport(BaseModel):
    """Unified report aggregating all gate results."""

    gates: List[GateResult] = Field(default_factory=list, description="Results per gate")
    overall_passed: bool = Field(default=False, description="Whether all gates passed")
    coverage: Optional[float] = Field(default=None, description="Code coverage percentage")
    coverage_gaps: Optional[str] = Field(
        default=None, description="Uncovered lines for re-prompting"
    )
    summary: str = Field(default="", description="Human-readable summary")

    @classmethod
    def from_gates(
        cls,
        gates: List[GateResult],
        coverage: Optional[float] = None,
        coverage_gaps: Optional[str] = None,
    ) -> "VerificationReport":
        """Build a report from a list of gate results."""
        overall = all(g.passed for g in gates)

        parts = []
        for g in gates:
            status = "PASS" if g.passed else "FAIL"
            n_findings = len(g.findings)
            parts.append(f"[{g.gate_name}] {status} ({n_findings} finding(s))")
        summary = "; ".join(parts)

        if coverage is not None:
            summary += f" | coverage: {coverage:.0f}%"

        return cls(
            gates=gates,
            overall_passed=overall,
            coverage=coverage,
            coverage_gaps=coverage_gaps,
            summary=summary,
        )

    def format_for_repair(self) -> str:
        """Format the report as structured diagnostics for a repair prompt."""
        sections: List[str] = []

        for gate in self.gates:
            if gate.passed:
                sections.append(f"[GATE: {gate.gate_name}] PASS")
                continue

            sections.append(f"[GATE: {gate.gate_name}] FAIL")
            for f in gate.error_findings:
                line_info = f" at line {f.line}" if f.line else ""
                code_info = f" ({f.code})" if f.code else ""
                sections.append(f"  - {f.severity.value.upper()}{code_info}{line_info}: {f.message}")
                if f.suggestion:
                    sections.append(f"    Suggestion: {f.suggestion}")

        if self.coverage_gaps:
            sections.append(f"\n[COVERAGE GAPS]\n  Lines not covered: {self.coverage_gaps}")

        return "\n".join(sections)
