"""Tests for the Phase 3 UI Test Agent and supporting modules."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.ui_test import (
    UITestAgent,
    UIRepairContext,
    GeneratedUITest,
    SYSTEM_PROMPT,
    GENERATION_TEMPLATE,
    REPAIR_TEMPLATE,
)
from src.agents.router import RouterAgent, TaskType
from src.config import UITestConfig


class TestGeneratedUITestModel:

    def test_basic_creation(self):
        t = GeneratedUITest(
            test_code="def test_foo(page): ...",
            test_functions=["test_foo"],
            imports=["import os"],
            target_url="http://localhost:3000",
        )
        assert t.test_code == "def test_foo(page): ..."
        assert t.target_url == "http://localhost:3000"

    def test_optional_url(self):
        t = GeneratedUITest(test_code="code")
        assert t.target_url is None


class TestUIRepairContext:

    def test_full_context(self):
        ctx = UIRepairContext(
            previous_code="old code",
            error_type="timeout_error",
            error_message="Playwright timeout",
            selector_hint="get_by_role('button') not found",
            screenshot_hint="Page shows a loading spinner",
            diagnostics="[GATE: sandbox] FAIL",
        )
        assert ctx.selector_hint is not None
        assert ctx.screenshot_hint is not None

    def test_minimal_context(self):
        ctx = UIRepairContext(
            previous_code="code",
            error_type="test_failure",
            error_message="1 test failed",
        )
        assert ctx.selector_hint is None
        assert ctx.diagnostics is None


class TestUITestAgentCodeExtraction:

    def setup_method(self):
        self.agent = UITestAgent(llm=MagicMock())

    def test_extract_from_code_block(self):
        response = '```python\nimport os\ndef test_x(page):\n    pass\n```'
        code = self.agent._extract_code_from_response(response)
        assert "def test_x(page):" in code
        assert "```" not in code

    def test_extract_plain_code(self):
        response = "import os\ndef test_nav(page):\n    page.goto('http://localhost')\n"
        code = self.agent._extract_code_from_response(response)
        assert "def test_nav(page):" in code

    def test_extract_test_functions(self):
        code = "def test_login(page): ...\ndef test_signup(page): ...\ndef helper(): ..."
        fns = self.agent._extract_test_functions(code)
        assert fns == ["test_login", "test_signup"]

    def test_extract_imports(self):
        code = "import os\nfrom playwright.sync_api import expect\ndef test_x(): ..."
        imports = self.agent._extract_imports(code)
        assert len(imports) == 2


class TestUITestAgentGenerate:

    def test_generate_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="import os\ndef test_home(page):\n    page.goto(os.environ['BASE_URL'])\n"
        )
        agent = UITestAgent(llm=mock_llm)
        result = agent.generate(
            description="Test the home page loads",
            target_url="http://localhost:3000",
        )

        assert isinstance(result, GeneratedUITest)
        assert "test_home" in result.test_functions
        assert result.target_url == "http://localhost:3000"
        mock_llm.invoke.assert_called_once()

    def test_generate_with_html_content(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="def test_form(page):\n    pass\n"
        )
        agent = UITestAgent(llm=mock_llm)
        result = agent.generate(
            description="Test form submission",
            html_content="<html><body><form></form></body></html>",
        )
        assert isinstance(result, GeneratedUITest)

        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[1].content
        assert "HTML content" in prompt_text


class TestUITestAgentRepair:

    def test_repair_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="def test_fixed(page):\n    page.goto('http://localhost')\n"
        )
        agent = UITestAgent(llm=mock_llm)
        ctx = UIRepairContext(
            previous_code="def test_broken(page): ...",
            error_type="timeout_error",
            error_message="Locator timed out",
            selector_hint="get_by_role('button', name='Submit') not found",
        )
        result = agent.repair(ctx)
        assert isinstance(result, GeneratedUITest)
        assert "test_fixed" in result.test_functions


class TestRouterUIDetection:

    def test_classify_ui_test(self):
        router = RouterAgent()
        assert router.classify_task("Generate Playwright E2E tests for the login page") == TaskType.UI_TEST
        assert router.classify_task("Write e2e test for signup flow") == TaskType.UI_TEST
        assert router.classify_task("browser test for checkout") == TaskType.UI_TEST

    def test_framework_hint_playwright(self):
        router = RouterAgent()
        from src.agents.router import Language
        assert router.get_framework_hint(Language.PYTHON, TaskType.UI_TEST) == "playwright"


class TestUITestConfig:

    def test_defaults(self):
        cfg = UITestConfig()
        assert cfg.playwright_image == "llm-agent-playwright"
        assert cfg.retry_budget == 5
        assert cfg.timeout == 120
        assert cfg.network_enabled is True
        assert cfg.memory_limit == "1g"

    @patch.dict("os.environ", {
        "UI_TEST_RETRY_BUDGET": "10",
        "PLAYWRIGHT_IMAGE": "custom-pw",
    })
    def test_from_env(self):
        cfg = UITestConfig.from_env()
        assert cfg.retry_budget == 10
        assert cfg.playwright_image == "custom-pw"
