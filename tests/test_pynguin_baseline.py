"""Tests for the optional Pynguin baseline runner."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import scripts.run_pynguin_baseline as pynguin_baseline
from src.config import SandboxConfig
from src.evaluation.models import BenchmarkCase


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        id="case-1",
        code="def add(a, b):\n    return a + b\n",
        language="python",
    )


def test_pynguin_timeout_without_generated_tests_returns_eval_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args") or "pynguin",
            timeout=kwargs["timeout"],
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(pynguin_baseline.subprocess, "run", fake_run)

    result = pynguin_baseline._run_case(_case(), SandboxConfig.from_env(), seconds=1)

    assert result.passed is False
    assert "Pynguin timed out after 31s" in result.error
    assert result.pipeline_state["baseline"] == "pynguin"
    assert result.pipeline_state["timed_out"] is True
    assert result.pipeline_state["returncode"] is None


def test_pynguin_timeout_with_generated_tests_evaluates_partial_output(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pynguin", timeout=kwargs["timeout"])

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

    monkeypatch.setattr(pynguin_baseline.subprocess, "run", fake_run)
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
