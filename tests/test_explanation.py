"""Tests for the Code Explanation Agent (Phase 4)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.agents.explanation import (
    CodeExplanation,
    WalkthroughBlock,
    ComplexityAnalysis,
    ExplanationAgent,
    ExplanationRepairContext,
)
from src.agents.router import RouterAgent, TaskType
from src.verification.complexity import ComplexityValidator, _rank
from src.verification.explanation_judge import ExplanationJudge
from src.config import ExplanationConfig


# ── CodeExplanation model tests ───────────────────────────────────────


class TestCodeExplanation:
    def test_creation_and_serialization(self):
        expl = CodeExplanation(
            summary="Sorts a list using bubble sort.",
            walkthrough=[
                WalkthroughBlock(block="lines 1-3", explanation="Function definition"),
                WalkthroughBlock(block="lines 4-8", explanation="Nested loop comparison"),
            ],
            complexity=ComplexityAnalysis(time="O(n^2)", space="O(1)", assumptions="n = length of input list"),
            dependencies=[],
            potential_issues=["Inefficient for large inputs"],
        )
        data = expl.model_dump()
        assert data["summary"] == "Sorts a list using bubble sort."
        assert len(data["walkthrough"]) == 2
        assert data["complexity"]["time"] == "O(n^2)"
        assert data["potential_issues"] == ["Inefficient for large inputs"]

    def test_json_roundtrip(self):
        expl = CodeExplanation(
            summary="Test function.",
            walkthrough=[],
            complexity=ComplexityAnalysis(time="O(1)", space="O(1)", assumptions="constant"),
            dependencies=["numpy"],
            potential_issues=[],
        )
        json_str = json.dumps(expl.model_dump())
        restored = CodeExplanation(**json.loads(json_str))
        assert restored.summary == expl.summary
        assert restored.dependencies == ["numpy"]

    def test_to_markdown_contains_sections(self):
        expl = CodeExplanation(
            summary="Sums numbers.",
            walkthrough=[WalkthroughBlock(block="lines 1-2", explanation="loop")],
            complexity=ComplexityAnalysis(time="O(n)", space="O(1)", assumptions="n items"),
            dependencies=["requests"],
            potential_issues=["No error handling"],
        )
        md = expl.to_markdown()
        assert "## Summary" in md
        assert "## Walkthrough" in md
        assert "## Complexity" in md
        assert "O(n)" in md
        assert "requests" in md
        assert "No error handling" in md


# ── ExplanationAgent tests ────────────────────────────────────────────


class TestExplanationAgent:
    def _mock_llm(self, response_text: str) -> MagicMock:
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        llm.invoke.return_value = mock_response
        return llm

    def test_generate_parses_valid_json(self):
        valid_json = json.dumps({
            "summary": "Adds two numbers.",
            "walkthrough": [{"block": "lines 1-2", "explanation": "add function"}],
            "complexity": {"time": "O(1)", "space": "O(1)", "assumptions": "constant"},
            "dependencies": [],
            "potential_issues": [],
        })
        llm = self._mock_llm(valid_json)
        agent = ExplanationAgent(llm)
        result = agent.generate("def add(a,b): return a+b")
        assert result.summary == "Adds two numbers."
        assert len(result.walkthrough) == 1

    def test_generate_fallback_on_invalid_json(self):
        llm = self._mock_llm("This is not valid JSON at all")
        agent = ExplanationAgent(llm)
        result = agent.generate("x = 1")
        assert "Failed to parse" in result.potential_issues[0]

    def test_repair_sends_feedback(self):
        valid_json = json.dumps({
            "summary": "Fixed explanation.",
            "walkthrough": [],
            "complexity": {"time": "O(n)", "space": "O(1)", "assumptions": "n items"},
            "dependencies": [],
            "potential_issues": [],
        })
        llm = self._mock_llm(valid_json)
        agent = ExplanationAgent(llm)

        ctx = ExplanationRepairContext(
            previous_explanation='{"summary": "wrong"}',
            error_type="rubric_completeness",
            error_message="Missing walkthrough",
            judge_feedback="Walkthrough is empty",
            complexity_feedback="Claimed O(1) but has loop",
        )
        result = agent.repair(ctx)
        assert result.summary == "Fixed explanation."
        call_args = llm.invoke.call_args[0][0]
        assert any("wrong" in str(m.content) for m in call_args)


# ── ComplexityValidator tests ─────────────────────────────────────────


class TestComplexityValidator:
    def test_nested_loops_detects_quadratic(self):
        code = """
def func(items):
    for i in items:
        for j in items:
            pass
"""
        v = ComplexityValidator()
        result = v.validate(code, "O(n)", "O(1)")
        assert not result.passed
        assert any("too low" in f.message for f in result.findings)

    def test_nested_loops_passes_with_correct_claim(self):
        code = """
def func(items):
    for i in items:
        for j in items:
            pass
"""
        v = ComplexityValidator()
        result = v.validate(code, "O(n^2)", "O(1)")
        assert result.passed

    def test_sort_call_detected(self):
        code = """
def func(items):
    return sorted(items)
"""
        v = ComplexityValidator()
        result = v.validate(code, "O(n)", "O(1)")
        assert not result.passed
        assert any("sort" in f.message.lower() for f in result.findings)

    def test_no_loops_passes_o1(self):
        code = """
def func(x):
    return x + 1
"""
        v = ComplexityValidator()
        result = v.validate(code, "O(1)", "O(1)")
        assert result.passed

    def test_recursion_detected(self):
        code = """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
"""
        v = ComplexityValidator()
        result = v.validate(code, "O(1)", "O(1)")
        assert not result.passed

    def test_syntax_error_passes(self):
        v = ComplexityValidator()
        result = v.validate("this is not python!!!", "O(n)", "O(1)")
        assert result.passed

    def test_rank_function(self):
        assert _rank("O(1)") < _rank("O(n)")
        assert _rank("O(n)") < _rank("O(n^2)")
        assert _rank("O(n log n)") < _rank("O(n^2)")

    def test_collection_allocation_warns_space(self):
        code = """
def func(items):
    return [x*2 for x in items]
"""
        v = ComplexityValidator()
        result = v.validate(code, "O(n)", "O(1)")
        assert any(f.code == "space_underestimate" for f in result.findings)


# ── ExplanationJudge tests ────────────────────────────────────────────


class TestExplanationJudge:
    def _mock_llm(self, response_text: str) -> MagicMock:
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        llm.invoke.return_value = mock_response
        return llm

    def test_all_pass(self):
        response = json.dumps([
            {"criterion": "factual_consistency", "passed": True, "reasoning": "OK"},
            {"criterion": "completeness", "passed": True, "reasoning": "OK"},
            {"criterion": "complexity_plausibility", "passed": True, "reasoning": "OK"},
            {"criterion": "clarity", "passed": True, "reasoning": "OK"},
        ])
        llm = self._mock_llm(response)
        judge = ExplanationJudge(llm)
        result = judge.verify("def f(): pass", '{"summary":"ok"}')
        assert result.passed
        assert len(result.findings) == 0

    def test_partial_failure(self):
        response = json.dumps([
            {"criterion": "factual_consistency", "passed": True, "reasoning": "OK"},
            {"criterion": "completeness", "passed": False, "reasoning": "Missing func g"},
            {"criterion": "complexity_plausibility", "passed": True, "reasoning": "OK"},
            {"criterion": "clarity", "passed": True, "reasoning": "OK"},
        ])
        llm = self._mock_llm(response)
        judge = ExplanationJudge(llm)
        result = judge.verify("def f(): pass\ndef g(): pass", '{"summary":"ok"}')
        assert not result.passed
        assert len(result.findings) == 1
        assert "completeness" in result.findings[0].code

    def test_unparseable_response(self):
        llm = self._mock_llm("I cannot evaluate this")
        judge = ExplanationJudge(llm)
        result = judge.verify("x=1", '{"summary":"x"}')
        assert result.passed


# ── Router classification tests ───────────────────────────────────────


class TestRouterExplanation:
    def test_explain_keyword(self):
        router = RouterAgent()
        assert router.classify_task("Explain this code") == TaskType.EXPLANATION

    def test_complexity_keyword(self):
        router = RouterAgent()
        assert router.classify_task("What is the complexity?") == TaskType.EXPLANATION

    def test_walkthrough_keyword(self):
        router = RouterAgent()
        assert router.classify_task("Give me a walkthrough") == TaskType.EXPLANATION

    def test_unit_test_not_explanation(self):
        router = RouterAgent()
        assert router.classify_task("Generate unit tests") == TaskType.UNIT_TEST


# ── ExplanationConfig tests ───────────────────────────────────────────


class TestExplanationConfig:
    def test_defaults(self):
        cfg = ExplanationConfig()
        assert cfg.enabled is True
        assert cfg.max_retries == 2
        assert cfg.judge_enabled is True
        assert cfg.complexity_check_enabled is True

    def test_from_env(self):
        with patch.dict("os.environ", {
            "EXPLANATION_ENABLED": "false",
            "EXPLANATION_MAX_RETRIES": "4",
        }):
            cfg = ExplanationConfig.from_env()
            assert cfg.enabled is False
            assert cfg.max_retries == 4
