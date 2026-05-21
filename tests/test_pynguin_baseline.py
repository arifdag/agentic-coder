"""Tests for the optional Pynguin baseline runner."""

from __future__ import annotations

from types import SimpleNamespace

import scripts.run_pynguin_baseline as pynguin_baseline
from src.config import SandboxConfig
from src.evaluation.models import BenchmarkCase


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        id="case-1",
        code="def add(a, b):\n    return a + b\n",
        language="python",
        metadata={"target": "add"},
    )


def test_pynguin_timeout_without_generated_tests_returns_eval_error(monkeypatch):
    def fake_run_pynguin(cmd, env, timeout_seconds):
        return None, "partial stdout", "partial stderr", True

    monkeypatch.setattr(pynguin_baseline, "_run_pynguin_command", fake_run_pynguin)

    result = pynguin_baseline._run_case(_case(), SandboxConfig.from_env(), seconds=1)

    assert result.passed is False
    assert "Pynguin timed out after 31s" in result.error
    assert result.pipeline_state["baseline"] == "pynguin"
    assert result.pipeline_state["timed_out"] is True
    assert result.pipeline_state["returncode"] is None


def test_pynguin_timeout_with_generated_tests_evaluates_partial_output(monkeypatch):
    def fake_run_pynguin(cmd, env, timeout_seconds):
        return None, "", "", True

    class FakeSandboxExecutor:
        def __init__(self, config):
            self.config = config

        def execute(self, source_code: str, test_code: str):
            assert "assert add(1, 2) == 3" in test_code
            return SimpleNamespace(
                success=True,
                tests_run=1,
                tests_passed=1,
                coverage=100.0,
                stderr="",
            )

    monkeypatch.setattr(pynguin_baseline, "_run_pynguin_command", fake_run_pynguin)
    monkeypatch.setattr(
        pynguin_baseline,
        "_generated_tests",
        lambda output_dir: "from source_module import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
    )
    monkeypatch.setattr(pynguin_baseline, "SandboxExecutor", FakeSandboxExecutor)

    result = pynguin_baseline._run_case(_case(), SandboxConfig.from_env(), seconds=1)

    assert result.passed is True
    assert result.tests_run == 1
    assert result.coverage == 100.0
    assert result.pipeline_state["timed_out"] is True
    assert result.pipeline_state["timeout_seconds"] == 31


def test_pynguin_fast_quality_metrics_are_attached(monkeypatch):
    def fake_run_pynguin(cmd, env, timeout_seconds):
        return 0, "", "", False

    class FakeSandboxExecutor:
        def __init__(self, config):
            self.config = config

        def execute(self, source_code: str, test_code: str):
            assert "assert add(1, 2) == 3" in test_code
            return SimpleNamespace(
                success=True,
                tests_run=1,
                tests_passed=1,
                coverage=100.0,
                branch_coverage=100.0,
                coverage_data={
                    "files": {
                        "source_module.py": {
                            "executed_lines": [1, 2],
                            "executed_branches": [],
                            "missing_branches": [],
                        }
                    },
                    "totals": {
                        "percent_covered": 100.0,
                        "percent_covered_branches": 100.0,
                    },
                },
                stderr="",
            )

    monkeypatch.setattr(pynguin_baseline, "_run_pynguin_command", fake_run_pynguin)
    monkeypatch.setattr(
        pynguin_baseline,
        "_generated_tests",
        lambda output_dir: "from source_module import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
    )
    monkeypatch.setattr(pynguin_baseline, "SandboxExecutor", FakeSandboxExecutor)

    result = pynguin_baseline._run_case(
        _case(),
        SandboxConfig.from_env(),
        seconds=1,
        quality="fast",
    )

    assert result.passed is True
    assert result.coverage_metrics["line_coverage"] == 100.0
    assert result.coverage_metrics["target_line_coverage"] == 100.0
    assert result.coverage_metrics["target_branch_coverage"] == 100.0
    assert result.oracle_metrics["has_assertions"] is True
    assert result.relevance_metrics["relevance_pass"] is True
    assert result.relevance_metrics["direct_target_relevance"] is True


def test_pynguin_full_quality_runs_bounded_mutation_metrics(monkeypatch):
    class FakeSandboxExecutor:
        def __init__(self, config):
            self.config = config
            self.calls = 0

        def execute(self, source_code: str, test_code: str):
            self.calls += 1
            return SimpleNamespace(success=self.calls == 2)

    sandbox_result = SimpleNamespace(
        success=True,
        coverage=100.0,
        branch_coverage=None,
        coverage_data={
            "files": {
                "source_module.py": {
                    "executed_lines": [1, 2],
                    "executed_branches": [],
                    "missing_branches": [],
                }
            },
            "totals": {"percent_covered": 100.0},
        },
    )
    mutants = [
        {"id": "mutant-1", "source_code": "def add(a, b):\n    return a - b\n"},
        {"id": "mutant-2", "source_code": "def add(a, b):\n    return a + b\n"},
    ]

    monkeypatch.setattr(
        pynguin_baseline,
        "generate_python_mutants",
        lambda source_code, limit: mutants[:limit],
    )
    monkeypatch.setattr(pynguin_baseline, "SandboxExecutor", FakeSandboxExecutor)

    payload = pynguin_baseline._build_quality_payload(
        quality="full",
        case=_case(),
        test_code="from source_module import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n",
        sandbox_result=sandbox_result,
        mutation_max_mutants=2,
        sandbox_config=SandboxConfig.from_env(),
    )

    assert payload["mutation_metrics"]["mutants_total"] == 2
    assert payload["mutation_metrics"]["mutants_killed"] == 1
    assert payload["mutation_metrics"]["mutation_score"] == 0.5
