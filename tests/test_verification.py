"""Tests for the Phase 2 verification layer."""

from src.verification.dependency import DependencyValidator, extract_imports
from src.verification.models import Finding, GateResult, JudgeVerdict, Severity, VerificationReport
from src.verification.relevance import RelevanceValidator
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
                severity=Severity.ERROR,
                message="false alarm",
                judge_verdict=JudgeVerdict.FALSE_POSITIVE,
            ),
            Finding(severity=Severity.WARNING, message="minor thing"),
        ]
        gate = GateResult(gate_name="sast", passed=False, findings=findings)

        errors = gate.error_findings
        assert len(errors) == 1
        assert errors[0].message == "real issue"

    def test_blocking_findings_include_warnings(self):
        """Regression: WARNING-level SAST hits (e.g. SQL injection in Bandit MEDIUM)
        must be treated as blocking so the repair loop acts on them."""
        findings = [
            Finding(severity=Severity.ERROR, code="CWE-78", message="shell=True"),
            Finding(severity=Severity.WARNING, code="CWE-89", message="SQL injection vector"),
            Finding(
                severity=Severity.WARNING,
                code="CWE-327",
                message="Weak hash MD5",
                judge_verdict=JudgeVerdict.FALSE_POSITIVE,
            ),
            Finding(severity=Severity.INFO, code="CWE-703", message="Use of assert"),
        ]
        gate = GateResult(gate_name="sast", passed=False, findings=findings)

        blocking = gate.blocking_findings
        codes = {f.code for f in blocking}
        assert "CWE-78" in codes
        assert "CWE-89" in codes, "WARNING-severity SQL injection must be blocking"
        assert "CWE-327" not in codes, "FP-judged findings must be excluded"
        assert "CWE-703" not in codes, "INFO severity must not block"

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
            gate_name="dependency",
            passed=False,
            findings=[Finding(severity=Severity.ERROR, message="phantom package")],
        )
        report = VerificationReport.from_gates([g1, g2])
        assert report.overall_passed is False

    def test_format_for_repair(self):
        g1 = GateResult(
            gate_name="sast",
            passed=False,
            findings=[
                Finding(
                    severity=Severity.ERROR,
                    code="CWE-78",
                    message="Command injection",
                    line=10,
                ),
            ],
        )
        report = VerificationReport.from_gates([g1], coverage_gaps="12, 15-18")
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

    def test_phantom_package_in_source_is_caught_when_combined(self):
        """Regression: the dep_hallucination benchmark places the
        hallucinated import in the *source* code, not the generated
        test. The pipeline therefore has to feed the validator the
        concatenation of source + test. When that happens, the phantom
        must still be detected."""
        validator = DependencyValidator(pypi_timeout=5)
        source_code = (
            "import superturboparser_xyzzy_not_a_real_package\n"
            "def parse(x):\n"
            "    return superturboparser_xyzzy_not_a_real_package.parse(x)\n"
        )
        test_code = (
            "from source_module import parse\n"
            "def test_parse():\n"
            "    assert parse('x') is not None\n"
        )
        combined = source_code + "\n\n" + test_code
        result = validator.validate(combined)
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
            f
            for f in result.findings
            if f.severity == Severity.ERROR and "not installed" not in f.message
        ]
        assert len(real_errors) == 0

    def test_analyze_returns_gate_result(self):
        analyzer = SastAnalyzer(timeout=10)
        result = analyzer.analyze("x = 1")
        assert isinstance(result, GateResult)
        assert result.gate_name == "sast"

    def test_sql_injection_blocks_gate(self):
        """Regression: a known SQL-injection pattern must block the SAST gate
        even when SAST maps it to WARNING severity."""
        analyzer = SastAnalyzer(timeout=30)
        code = (
            "import sqlite3\n"
            "def get_user(conn, user_id):\n"
            '    q = "SELECT * FROM users WHERE id = \'" + user_id + "\'"\n'
            "    return conn.execute(q).fetchall()\n"
        )
        result = analyzer.analyze(code)
        real_findings = [f for f in result.findings if "not installed" not in f.message]
        if not real_findings:
            import pytest

            pytest.skip("neither semgrep nor bandit detected anything on this host")
        assert result.passed is False, (
            "SAST gate should fail when SQL-injection pattern is detected "
            "regardless of whether the tool reports it as ERROR or WARNING"
        )

    def test_cwe_703_assert_noise_is_ignored(self):
        """Regression: Bandit B101 / CWE-703 (use of assert) is test-file noise
        and must never, on its own, block the gate."""
        analyzer = SastAnalyzer(timeout=30)
        code = (
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "def test_add():\n"
            "    assert add(2, 2) == 4\n"
            "    assert add(0, 0) == 0\n"
        )
        result = analyzer.analyze(code)
        blocking = [f for f in result.findings if f.severity in (Severity.ERROR, Severity.WARNING)]
        assert all(
            (f.code or "").upper() not in ("CWE-703", "B101") for f in result.findings
        ), "CWE-703/B101 should be filtered out"
        assert (
            result.passed is True or len(blocking) == 0
        ), "assert-only test code should not fail the SAST gate"


class TestSandboxFixImports:
    """Regression tests for ``SandboxExecutor._fix_imports``.

    Historical bug: the regex-based rewriter only matched single-line
    imports, so a multi-line parenthesized import like
    ``from foo import (\\n    bar,\\n    baz,\\n)`` would get line 1
    replaced with ``from source_module import *`` and leave lines 2-N
    dangling as orphaned indented code -- leading to an ``IndentationError``
    on the very first pytest collection. The fixer is now AST-driven so it
    replaces each import statement as a single logical unit.
    """

    def _fix(self, code: str) -> str:
        from src.verification.sandbox import SandboxExecutor

        return SandboxExecutor()._fix_imports(code)

    def test_multiline_paren_import_from_unknown_module_is_rewritten_cleanly(self):
        # This is the exact shape of the real bug: a multi-line parenthesized
        # import from an UNKNOWN module. The old regex rewriter would only
        # replace line 1 and leave lines 2..N as orphan indented code.
        code = (
            "from fuzzywuzzy import (\n"
            "    ratio,\n"
            "    partial_ratio,\n"
            "    token_sort_ratio,\n"
            ")\n"
            "\n"
            "def test_ratio():\n"
            "    assert ratio('a', 'a') == 100\n"
        )
        fixed = self._fix(code)
        # Must be parseable
        import ast

        ast.parse(fixed)
        assert "from source_module import *" in fixed
        # No dangling indented orphan lines left behind
        assert "    partial_ratio," not in fixed
        assert "    ratio," not in fixed
        # Original unknown module reference is gone
        assert "fuzzywuzzy" not in fixed

    def test_multiline_paren_import_from_unknown_module_redirected_to_source(self):
        code = (
            "from fuzzywuzzy import (\n"
            "    ratio,\n"
            "    partial_ratio,\n"
            ")\n"
            "\n"
            "def test_ratio():\n"
            "    assert ratio('a', 'a') == 100\n"
        )
        fixed = self._fix(code)
        import ast

        ast.parse(fixed)
        assert "from source_module import *" in fixed
        # The unknown module itself should be gone
        assert "fuzzywuzzy" not in fixed

    def test_known_third_party_multiline_import_is_preserved(self):
        code = (
            "from typing import (\n"
            "    List,\n"
            "    Dict,\n"
            ")\n"
            "\n"
            "def test_x():\n"
            "    x: List[int] = [1]\n"
            "    assert x == [1]\n"
        )
        fixed = self._fix(code)
        import ast

        ast.parse(fixed)
        assert "from typing import" in fixed
        assert "List" in fixed and "Dict" in fixed

    def test_unparseable_input_falls_back_gracefully(self):
        # If the LLM emits garbage, the fixer must not itself crash.
        code = "def broken(:\n    pass\n"
        fixed = self._fix(code)
        assert "from source_module import *" in fixed


class TestRelevanceValidator:
    """Tests for the test-relevance validator (anti-gaming gate).

    These regression tests are derived from the ULT ablation finding:
    at k=5 with no other gates, ~33% of "passing" cases had zero tests
    that even referenced the target function. The validator's job is to
    catch that pattern.
    """

    def test_relevant_test_with_import_passes(self):
        v = RelevanceValidator()
        test = (
            "from source_module import DetPiece\n"
            "\n"
            "def test_detpiece_pawn():\n"
            "    assert DetPiece('P1') == ('Pawn', True)\n"
        )
        result = v.validate(test, target_function="DetPiece")
        assert result.passed
        assert result.gate_name == "relevance"

    def test_calculator_gaming_case_is_caught(self):
        """The exact pattern observed in eval_results: LLM ignores the
        DetPiece target and writes Calculator tests + Calculator code."""
        v = RelevanceValidator()
        test = (
            "class Calculator:\n"
            "    def add(self, a, b): return a + b\n"
            "\n"
            "class TestCalculatorAdd:\n"
            "    def test_add_positive_numbers(self):\n"
            "        c = Calculator()\n"
            "        assert c.add(1, 2) == 3\n"
        )
        result = v.validate(test, target_function="DetPiece")
        assert not result.passed
        codes = {f.code for f in result.findings}
        assert "tests_unrelated_to_source" in codes

    def test_target_redefined_locally_is_blocked(self):
        v = RelevanceValidator()
        test = (
            "def DetPiece(x):\n"
            "    return ('Pawn', True)\n"
            "\n"
            "def test_detpiece():\n"
            "    assert DetPiece('P1') == ('Pawn', True)\n"
        )
        result = v.validate(test, target_function="DetPiece")
        assert not result.passed
        codes = {f.code for f in result.findings}
        assert "target_shadowed_in_tests" in codes

    def test_test_name_reference_alone_is_not_enough(self):
        v = RelevanceValidator()
        test = "def test_detpiece_returns_tuple():\n" "    value = 1\n" "    assert value == 1\n"
        result = v.validate(test, target_function="DetPiece")
        assert not result.passed
        assert any(f.code == "tests_unrelated_to_source" for f in result.findings)

    def test_camelcase_split_keyword_without_target_call_is_not_enough(self):
        v = RelevanceValidator()
        test = "def test_det_piece_logic():\n" "    value = 1\n" "    assert value == 1\n"
        result = v.validate(test, target_function="DetPiece")
        assert not result.passed

    def test_calls_target_signal_passes(self):
        v = RelevanceValidator()
        test = (
            "from source_module import add\n"
            "\n"
            "def test_basic_math():\n"
            "    assert add(1, 2) == 3\n"
        )
        result = v.validate(test, target_function="add")
        assert result.passed

    def test_module_target_call_signal_passes(self):
        v = RelevanceValidator()
        test = (
            "import source_module as sm\n"
            "\n"
            "def test_basic_math():\n"
            "    assert sm.add(1, 2) == 3\n"
        )
        result = v.validate(test, target_function="add")
        assert result.passed

    def test_dynamic_relevance_fails_when_target_not_executed(self):
        v = RelevanceValidator()
        source = "def add(a, b):\n    return a + b\n\ndef helper():\n    return 1\n"
        test = (
            "from source_module import add\n"
            "\n"
            "def test_basic_math():\n"
            "    assert add is not None\n"
        )
        coverage = {
            "files": {
                "source_module.py": {
                    "executed_lines": [4, 5],
                    "missing_lines": [1, 2],
                }
            }
        }
        result = v.validate_dynamic(test, source, coverage, target_function="add")
        assert not result.passed
        assert any(f.code == "target_not_executed" for f in result.findings)

    def test_dynamic_relevance_ignores_import_only_definition_line(self):
        v = RelevanceValidator()
        source = "def add(a, b):\n    return a + b\n"
        test = (
            "from source_module import add\n"
            "\n"
            "def test_basic_math():\n"
            "    assert add is not None\n"
        )
        coverage = {"files": {"source_module.py": {"executed_lines": [1]}}}
        result = v.validate_dynamic(test, source, coverage, target_function="add")
        assert not result.passed
        assert any(f.code == "target_not_executed" for f in result.findings)

    def test_dynamic_relevance_passes_when_target_executes(self):
        v = RelevanceValidator()
        source = "def add(a, b):\n    return a + b\n"
        test = (
            "from source_module import add\n"
            "\n"
            "def test_basic_math():\n"
            "    assert add(1, 2) == 3\n"
        )
        coverage = {"files": {"source_module.py": {"executed_lines": [1, 2]}}}
        result = v.validate_dynamic(test, source, coverage, target_function="add")
        assert result.passed

    def test_snake_case_target_keywords_match(self):
        v = RelevanceValidator()
        test = (
            "from source_module import compute_comment_stats\n"
            "\n"
            "def test_compute_comment_stats_basic():\n"
            "    assert compute_comment_stats('') == 0\n"
        )
        result = v.validate(test, target_function="compute_comment_stats")
        assert result.passed

    def test_empty_test_code_fails(self):
        v = RelevanceValidator()
        result = v.validate("", target_function="foo")
        assert not result.passed
        assert any(f.code == "empty_tests" for f in result.findings)

    def test_syntax_error_in_tests_fails_with_clear_finding(self):
        v = RelevanceValidator()
        result = v.validate("def broken(:\n    pass", target_function="foo")
        assert not result.passed
        assert any(f.code == "test_syntax_error" for f in result.findings)

    def test_auto_detect_target_from_source_code(self):
        v = RelevanceValidator()
        source = "def my_special_func(x):\n    return x + 1\n"
        test = (
            "from source_module import my_special_func\n"
            "def test_my_special_func():\n"
            "    assert my_special_func(1) == 2\n"
        )
        result = v.validate(test, source_code=source)
        assert result.passed

    def test_no_target_just_imports_source_passes(self):
        v = RelevanceValidator()
        test = (
            "import source_module\n"
            "def test_anything():\n"
            "    assert source_module is not None\n"
        )
        result = v.validate(test)
        assert result.passed
