"""Tests for the LLM Agent Platform pipeline components."""

from types import SimpleNamespace

import pytest

from src.agents.router import Language, RouterAgent, TaskType
from src.agents.unit_test import (
    GENERATION_TEMPLATE,
    REPAIR_TEMPLATE,
    SYSTEM_PROMPT,
    RepairContext,
    UnitTestAgent,
)
from src.config import Config


class TestRouterAgent:
    """Tests for the RouterAgent."""

    def setup_method(self):
        self.router = RouterAgent()

    def test_detect_python_by_extension(self):
        result = self.router.detect_language("some code", "test.py")
        assert result == Language.PYTHON

    def test_detect_python_by_syntax(self):
        code = """
def hello():
    print("Hello, world!")

class MyClass:
    def __init__(self):
        self.value = None
"""
        result = self.router.detect_language(code)
        assert result == Language.PYTHON

    def test_detect_javascript_by_syntax(self):
        code = """
const hello = () => {
    console.log("Hello");
};

function greet(name) {
    return `Hello, ${name}`;
}
"""
        result = self.router.detect_language(code)
        assert result == Language.JAVASCRIPT

    def test_detect_typescript_by_syntax(self):
        code = """
interface User {
    name: string;
    age: number;
}

const greet = (user: User): string => {
    return `Hello, ${user.name}`;
};
"""
        result = self.router.detect_language(code)
        assert result == Language.TYPESCRIPT

    def test_classify_unit_test_task(self):
        result = self.router.classify_task("Generate unit tests for this function")
        assert result == TaskType.UNIT_TEST

    def test_classify_ui_test_task(self):
        result = self.router.classify_task("Create e2e tests with playwright")
        assert result == TaskType.UI_TEST

    def test_classify_explanation_task(self):
        result = self.router.classify_task("Explain what this code does")
        assert result == TaskType.EXPLANATION

    def test_classify_unit_test_wins_over_description_words(self):
        """Regression: benchmark descriptions often contain words like
        ``comment``, ``complexity`` or ``describe`` which the router
        previously treated as strong explanation signal and mis-routed
        away from sandbox execution. The explicit ``Generate ... unit
        tests`` prefix must take precedence."""
        result = self.router.classify_task(
            "Generate comprehensive unit tests for the function below.\n\n"
            "Problem description:\nCollect data on number of comments in "
            "the current file and describe complexity."
        )
        assert result == TaskType.UNIT_TEST

    def test_classify_pure_explanation_still_routes_to_explanation(self):
        result = self.router.classify_task(
            "Explain how does this function work and walkthrough the logic."
        )
        assert result == TaskType.EXPLANATION

    def test_classify_ui_test_wins_over_unit_test_and_explain(self):
        result = self.router.classify_task(
            "Write unit tests? Actually we want an e2e test with playwright "
            "that describes user flow."
        )
        assert result == TaskType.UI_TEST

    def test_route_python_unit_test(self):
        code = """
def add(a, b):
    return a + b
"""
        result = self.router.route(code, "Generate tests", "utils.py")
        assert result.language == Language.PYTHON
        assert result.task_type == TaskType.UNIT_TEST
        assert result.framework_hint == "pytest"
        assert result.confidence == 1.0

    def test_route_unknown_language(self):
        result = self.router.route("random text", "test this")
        assert result.language == Language.UNKNOWN
        assert result.confidence == 0.5


class TestUnitTestAgentParsing:
    """Tests for UnitTestAgent parsing methods."""

    class CapturingLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(
                content=(
                    "from django.db.models.base import Model\n\n"
                    "def test_model_importable():\n"
                    "    assert Model is not None\n"
                )
            )

    def test_extract_test_functions(self):
        code = """
def test_add_positive():
    assert add(1, 2) == 3

def test_add_negative():
    assert add(-1, -2) == -3

def helper_function():
    pass

def test_add_zero():
    assert add(0, 0) == 0
"""
        agent = UnitTestAgent.__new__(UnitTestAgent)
        result = agent._extract_test_functions(code)

        assert len(result) == 3
        assert "test_add_positive" in result
        assert "test_add_negative" in result
        assert "test_add_zero" in result
        assert "helper_function" not in result

    def test_extract_imports(self):
        code = """
import pytest
from source_module import add, subtract
import os
from pathlib import Path

def test_something():
    pass
"""
        agent = UnitTestAgent.__new__(UnitTestAgent)
        result = agent._extract_imports(code)

        assert len(result) == 4
        assert "import pytest" in result
        assert "from source_module import add, subtract" in result

    def test_extract_code_from_markdown(self):
        response = """
Here are the tests:

```python
import pytest

def test_example():
    assert True
```

These tests cover the basic functionality.
"""
        agent = UnitTestAgent.__new__(UnitTestAgent)
        result = agent._extract_code_from_response(response)

        assert "import pytest" in result
        assert "def test_example" in result
        assert "Here are the tests" not in result

    def test_generation_prompt_includes_repo_import_module(self):
        llm = self.CapturingLLM()
        agent = UnitTestAgent(llm)

        agent.generate("class Model:\n    pass\n", import_module="django.db.models.base")

        prompt = llm.messages[-1].content
        assert "Repo-context import module: django.db.models.base" in prompt
        assert "from django.db.models.base import <target>" in prompt
        assert "Do NOT import from source_module in repo-context runs" in prompt

    def test_testgeneval_prompt_includes_official_context(self):
        llm = self.CapturingLLM()
        agent = UnitTestAgent(llm)

        result = agent.generate_testgeneval(
            code="class Model:\n    pass\n",
            metadata={
                "repo": "django/django",
                "version": "5.0",
                "code_file": "django/db/models/base.py",
                "test_file": "tests/model_tests/test_base.py",
                "import_module": "django.db.models.base",
            },
            user_request="Generate comprehensive pytest tests.",
        )

        prompt = llm.messages[-1].content
        assert "official TestGenEval evaluation case" in prompt
        assert "Repository: django/django" in prompt
        assert "Source file under test: django/db/models/base.py" in prompt
        assert "Existing/target test file path: tests/model_tests/test_base.py" in prompt
        assert "from django.db.models.base import <public function or class>" in prompt
        assert "Do NOT import from source_module for TestGenEval official runs" in prompt
        assert "Do NOT define test classes" in prompt
        assert "file-level test functions named test_*" in prompt
        assert "Do NOT import pytest or use pytest.raises" in prompt
        assert "official TestGenEval-compatible unit tests" in llm.messages[0].content
        assert result.test_functions == ["test_model_importable"]

    def test_testgeneval_prompt_includes_validation_feedback(self):
        llm = self.CapturingLLM()
        agent = UnitTestAgent(llm)

        agent.generate_testgeneval(
            code="class Model:\n    pass\n",
            metadata={"import_module": "django.db.models.base"},
            feedback="Generated test code is empty",
        )

        prompt = llm.messages[-1].content
        assert "Previous attempt was rejected by local validation" in prompt
        assert "Generated test code is empty" in prompt
        assert "Correct the issue in this attempt" in prompt

    def test_generation_context_preserves_file_path_module_hint(self):
        agent = UnitTestAgent.__new__(UnitTestAgent)

        context = agent._build_context_section(file_path="source_module.py")

        assert "Module to import: source_module" in context

    def test_generation_prompt_includes_exact_target_function(self):
        llm = self.CapturingLLM()
        agent = UnitTestAgent(llm)

        agent.generate("def add(a, b):\n    return a + b\n", target_function="add")

        prompt = llm.messages[-1].content
        assert "Target function/class: add" in prompt
        assert "import add directly from source_module" in prompt
        assert "broad module API smoke tests" in prompt
        assert "normal inputs, edge cases, and invalid-or-error inputs" in prompt


class TestRepairContext:
    """Tests for RepairContext model."""

    def test_repair_context_creation(self):
        context = RepairContext(
            previous_code="def test(): pass",
            error_type="syntax_error",
            error_message="Invalid syntax",
            line_number=1,
        )
        assert context.previous_code == "def test(): pass"
        assert context.error_type == "syntax_error"
        assert context.line_number == 1

    def test_repair_context_with_coverage_gaps(self):
        context = RepairContext(
            previous_code="def test(): pass",
            error_type="test_failure",
            error_message="1 test failed",
            coverage_gaps="12, 15-18, 23",
            diagnostics="[GATE: sandbox] FAIL\n  - ERROR: 1 test(s) failed",
        )
        assert context.coverage_gaps == "12, 15-18, 23"
        assert context.diagnostics is not None

    def test_repair_context_with_import_module(self):
        context = RepairContext(
            previous_code="def test(): pass",
            error_type="import_error",
            error_message="No module named source_module",
            import_module="sklearn.preprocessing._label",
        )
        assert context.import_module == "sklearn.preprocessing._label"

    def test_repair_prompt_includes_repo_import_module(self):
        llm = TestUnitTestAgentParsing.CapturingLLM()
        agent = UnitTestAgent(llm)
        context = RepairContext(
            previous_code="from source_module import LabelEncoder\n",
            error_type="import_error",
            error_message="No module named source_module",
            import_module="sklearn.preprocessing._label",
        )

        agent.repair(context)

        prompt = llm.messages[-1].content
        assert "Import from: sklearn.preprocessing._label" in prompt
        assert "Do NOT use source_module in repo-context repairs" in prompt
        assert "from sklearn.preprocessing._label import <target>" in prompt

    def test_repair_prompt_includes_target_function(self):
        llm = TestUnitTestAgentParsing.CapturingLLM()
        agent = UnitTestAgent(llm)
        context = RepairContext(
            previous_code="import source_module\n",
            error_type="tests_unrelated_to_source",
            error_message="never calls target",
            target_function="add",
        )

        agent.repair(context)

        prompt = llm.messages[-1].content
        assert "Target function/class: add" in prompt
        assert "import add directly from source_module" in prompt
        assert "Call or instantiate add inside every test assertion" in prompt


class TestIntegration:
    """Integration tests (require API key to run)."""

    @pytest.mark.skip(reason="Requires API key")
    def test_full_pipeline(self):
        from src.graph.pipeline import run_pipeline

        code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''

        result = run_pipeline(
            code=code,
            user_request="Generate unit tests",
            max_retries=1,
        )

        assert result["status"] in ["success", "failed_after_retries"]
        assert result["generated_tests"] is not None


def test_pipeline_does_not_repair_source_only_phantom_import(monkeypatch, tmp_path):
    from src.graph import pipeline as pipeline_module

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return SimpleNamespace(
                content=(
                    "from source_module import target\n\n"
                    "def test_target():\n"
                    "    assert target() == 1\n"
                )
            )

    class FakeSandboxExecutor:
        def __init__(self, config):
            self.config = config

        def execute(self, source_code, test_code):
            return SimpleNamespace(
                success=True,
                tests_run=1,
                tests_passed=1,
                tests_failed=0,
                coverage=100.0,
                branch_coverage=None,
                coverage_data={
                    "files": {
                        "source_module.py": {
                            "executed_lines": [2, 3],
                            "executed_branches": [],
                            "missing_branches": [],
                        }
                    }
                },
                coverage_gaps=None,
                stdout="",
                stderr="",
                error_type=None,
                error_message=None,
                line_number=None,
                infrastructure_pass=True,
                repo_setup_pass=None,
            )

    fake_llm = FakeLLM()
    monkeypatch.setattr(pipeline_module, "get_role_llm", lambda config, role: fake_llm)
    monkeypatch.setattr(pipeline_module, "SandboxExecutor", FakeSandboxExecutor)
    monkeypatch.setattr(
        pipeline_module.DependencyValidator,
        "_check_pypi",
        lambda self, package: package != "phantom_source_pkg",
    )

    config = Config()
    config.sast.enabled = False
    config.judge.enabled = False
    config.pipeline.audit_log_dir = str(tmp_path / "audit")

    result = pipeline_module.run_pipeline(
        code="import phantom_source_pkg\n\ndef target():\n    return 1\n",
        user_request="Generate unit tests",
        max_retries=3,
        config=config,
        target_function="target",
    )

    assert result["source_phantom_import"] is True
    assert result["source_phantom_packages"] == ["phantom_source_pkg"]
    assert result["error_type"] == "source_phantom_import"
    assert result["retry_count"] == 0
    assert fake_llm.calls == 1


class TestGenerationPromptContent:
    """Verify that SYSTEM_PROMPT and GENERATION_TEMPLATE include all critical guidance."""

    def test_system_prompt_import_and_source_module_guidance(self):
        """SYSTEM_PROMPT must guide importing from the source module."""
        text = SYSTEM_PROMPT.lower()
        assert "source_module" in text
        assert "source module" in text
        assert "import" in text

    def test_system_prompt_call_or_instantiate_target(self):
        """SYSTEM_PROMPT must require calling or instantiating the original targets."""
        text = SYSTEM_PROMPT.lower()
        assert "call" in text or "instantiate" in text

    def test_system_prompt_forbid_shadow_reimplement(self):
        """SYSTEM_PROMPT must forbid redefining, shadowing, or re-implementing source functions."""
        text = SYSTEM_PROMPT.lower()
        assert "redefine" in text or "shadow" in text or "re-implement" in text

    def test_system_prompt_meaningful_assertions(self):
        """SYSTEM_PROMPT must require meaningful assertions and pytest.raises."""
        text = SYSTEM_PROMPT.lower()
        assert "meaningful" in text
        assert "assert" in text
        assert "pytest.raises" in SYSTEM_PROMPT

    def test_system_prompt_forbid_dummy_assertions(self):
        """SYSTEM_PROMPT must explicitly forbid dummy assertions like assert True."""
        text = SYSTEM_PROMPT.lower()
        assert "assert true" in text or "assert 1 == 1" in text

    def test_system_prompt_branch_coverage(self):
        """SYSTEM_PROMPT must require executing target logic, not just importing."""
        text = SYSTEM_PROMPT.lower()
        assert "branch" in text or "execute" in text or "logic" in text

    def test_generation_template_import_and_call_targets(self):
        """GENERATION_TEMPLATE must require importing and calling real targets."""
        text = GENERATION_TEMPLATE.lower()
        assert "import" in text
        assert "call" in text or "instantiate" in text
        assert "source_module" in text
        assert "source module" in text

    def test_generation_template_forbid_dummy_assertions(self):
        """GENERATION_TEMPLATE must forbid dummy assertions."""
        text = GENERATION_TEMPLATE.lower()
        assert "assert true" in text or "assert 1 == 1" in text or "dummy" in text


class TestRepairPromptContent:
    """Verify that REPAIR_TEMPLATE includes relevance-specific repair guidance."""

    def test_repair_relevance_not_relevant(self):
        """REPAIR_TEMPLATE must include guidance for relevance / target_not_relevant."""
        text = REPAIR_TEMPLATE.lower()
        assert "target_not_relevant" in text or "relevance" in text
        assert "tests_unrelated_to_source" in text

    def test_repair_target_not_executed(self):
        """REPAIR_TEMPLATE must include guidance for target_not_executed."""
        text = REPAIR_TEMPLATE.lower()
        assert "target_not_executed" in text or "target_relevance" in text

    def test_repair_no_assertions_and_dummy(self):
        """REPAIR_TEMPLATE must cover no_assertions and dummy_assertions_only."""
        text = REPAIR_TEMPLATE.lower()
        assert "no_assertions" in text
        assert "dummy_assertions" in text or "dummy assertion" in text

    def test_repair_target_shadowed(self):
        """REPAIR_TEMPLATE must cover target_shadowed_in_tests."""
        assert "target_shadowed" in REPAIR_TEMPLATE

    def test_repair_body_execution_required(self):
        """REPAIR_TEMPLATE must state that target body execution is required, not import-only."""
        text = REPAIR_TEMPLATE.lower()
        assert "source_module" in text
        assert "body" in text or "executed" in text
        assert "not merely import" in text or "call" in text or "instantiate" in text

    def test_repair_timeout_and_no_tests_collected_guidance(self):
        text = REPAIR_TEMPLATE.lower()
        assert "no_tests_collected" in text
        assert "test_*" in text
        assert "timeout" in text
        assert "randomized" in text
        assert "large-input" in text
        assert "tiny deterministic inputs" in text
