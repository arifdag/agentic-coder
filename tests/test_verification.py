"""Tests for the Phase 2 verification layer."""

import pytest

from src.verification.models import (
    Finding, GateResult, VerificationReport,
    Severity, JudgeVerdict,
)
from src.verification.dependency import DependencyValidator, extract_imports
from src.verification.sast import SastAnalyzer


class TestModels:
    """Tests for verification models."""

    def test_finding_creation(self):
        f = Finding(
            severity=Severity.ERROR,
            code="CWE-78",
            message="OS command injection",
            line=15,
        )
        assert f.severity == Severity.ERROR
        assert f.code == "CWE-78"

    def test_gate_result_error_findings(self):
        findings = [
            Finding(severity=Severity.ERROR, message="real issue"),
            Finding(
                severity=Severity.ERROR, message="false alarm",
                judge_verdict=JudgeVerdict.FALSE_POSITIVE,
            ),
            Finding(severity=Severity.WARNING, message="minor thing"),
        ]
        gate = GateResult(gate_name="sast", passed=False, findings=findings)

        errors = gate.error_findings
        assert len(errors) == 1
        assert errors[0].message == "real issue"

    def test_verification_report_from_gates(self):
        g1 = GateResult(gate_name="sast", passed=True, findings=[])
        g2 = GateResult(gate_name="dependency", passed=True, findings=[])
        g3 = GateResult(gate_name="sandbox", passed=True, findings=[])

        report = VerificationReport.from_gates([g1, g2, g3], coverage=85.0)

        assert report.overall_passed is True
        assert report.coverage == 85.0
        assert "sast" in report.summary
        assert "dependency" in report.summary

    def test_verification_report_fails_if_any_gate_fails(self):
        g1 = GateResult(gate_name="sast", passed=True, findings=[])
        g2 = GateResult(
            gate_name="dependency", passed=False,
            findings=[Finding(severity=Severity.ERROR, message="phantom package")],
        )
        report = VerificationReport.from_gates([g1, g2])
        assert report.overall_passed is False

    def test_format_for_repair(self):
        g1 = GateResult(
            gate_name="sast", passed=False,
            findings=[
                Finding(
                    severity=Severity.ERROR,
                    code="CWE-78",
                    message="Command injection",
                    line=10,
                ),
            ],
        )
        report = VerificationReport.from_gates(
            [g1], coverage_gaps="12, 15-18"
        )
        text = report.format_for_repair()

        assert "[GATE: sast] FAIL" in text
        assert "CWE-78" in text
        assert "line 10" in text
        assert "12, 15-18" in text


class TestDependencyValidator:
    """Tests for the dependency validation gate."""

    def test_extract_imports_basic(self):
        code = """
import os
import json
from pathlib import Path
import numpy as np
from source_module import some_func
"""
        imports = extract_imports(code)
        assert "os" in imports
        assert "json" in imports
        assert "pathlib" in imports
        assert "numpy" in imports
        assert "source_module" in imports

    def test_extract_imports_syntax_error(self):
        code = "this is not python code {{{"
        imports = extract_imports(code)
        assert len(imports) == 0

    def test_stdlib_filtered_out(self):
        validator = DependencyValidator()
        code = """
import os
import sys
import json
import pytest
from source_module import func
"""
        result = validator.validate(code)
        assert result.passed is True
        assert result.gate_name == "dependency"

    def test_phantom_package_detected(self):
        validator = DependencyValidator(pypi_timeout=5)
        code = """
import this_package_definitely_does_not_exist_xyzzy_12345
"""
        result = validator.validate(code)
        assert result.passed is False
        assert any("PHANTOM-PKG" == f.code for f in result.findings)


class TestSastAnalyzer:
    """Tests for the SAST gate."""

    def test_analyze_safe_code(self):
        analyzer = SastAnalyzer(timeout=30)
        code = """
def add(a, b):
    return a + b

def test_add():
    assert add(1, 2) == 3
"""
        result = analyzer.analyze(code)
        assert result.gate_name == "sast"
        real_errors = [
            f for f in result.findings
            if f.severity == Severity.ERROR and "not installed" not in f.message
        ]
        assert len(real_errors) == 0

    def test_analyze_returns_gate_result(self):
        analyzer = SastAnalyzer(timeout=10)
        result = analyzer.analyze("x = 1")
        assert isinstance(result, GateResult)
        assert result.gate_name == "sast"
