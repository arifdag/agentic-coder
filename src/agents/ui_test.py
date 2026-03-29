"""UI Test Agent for generating Playwright E2E tests (Python / pytest-playwright)."""

import re
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


class GeneratedUITest(BaseModel):
    """Result of UI test generation."""

    test_code: str = Field(description="Generated Playwright test code")
    imports: List[str] = Field(default_factory=list, description="Required imports")
    test_functions: List[str] = Field(default_factory=list, description="Names of test functions")
    target_url: Optional[str] = Field(default=None, description="Target URL for the tests")


class UIRepairContext(BaseModel):
    """Context for repairing a failed UI test."""

    previous_code: str = Field(description="Previously generated test code")
    error_type: str = Field(description="Type of error encountered")
    error_message: str = Field(description="Error message from verification")
    line_number: Optional[int] = Field(default=None, description="Line number of error")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix approach")
    selector_hint: Optional[str] = Field(
        default=None, description="Hint about selectors that failed"
    )
    screenshot_hint: Optional[str] = Field(
        default=None, description="Description of screenshot evidence if available"
    )
    diagnostics: Optional[str] = Field(
        default=None, description="Full structured diagnostics from verification report"
    )


SYSTEM_PROMPT = """You are an expert QA engineer specialising in Playwright end-to-end tests written in Python with pytest-playwright.

Guidelines:
1. Use the pytest-playwright fixtures (`page`, `browser`, `context`).
   Every test function must accept `page` as its first argument.

2. Selector strategy (in order of preference):
   - `page.get_by_role(...)` with accessible roles
   - `page.get_by_label(...)`
   - `page.get_by_text(...)`
   - `page.get_by_test_id(...)` for data-testid attributes
   - CSS selectors only as a last resort

3. Assertions:
   - Use `expect(locator).to_be_visible()`, `to_have_text()`, `to_have_value()`, etc.
   - Never use bare `assert`; always use Playwright's `expect` API for auto-retry.

4. Navigation:
   - Read the target URL from the BASE_URL environment variable:
       `import os; BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")`
   - Navigate with `page.goto(BASE_URL)` or `page.goto(f"{BASE_URL}/path")`.

5. Best practices:
   - Add explicit waits only when the auto-wait is insufficient.
   - Keep tests independent – no shared mutable state between tests.
   - Use descriptive function names: `test_<feature>_<scenario>`.
   - Include brief docstrings explaining the user flow being tested.

6. Output format:
   - Return ONLY the Python test code.
   - Start with imports.
   - Do NOT include markdown code blocks or explanations.
   - The code must be immediately executable with `pytest --browser chromium`.
"""

GENERATION_TEMPLATE = """Generate Playwright E2E tests for the following target.

{target_section}

Test description:
{description}

Requirements:
- Test all described user flows
- Include at least 2-3 test functions covering happy-path and basic error states
- Use stable selectors (get_by_role, get_by_label, get_by_text, get_by_test_id)
- Assert visible page state after each interaction
- Read the base URL from the BASE_URL environment variable

Generate the complete test file now:"""

REPAIR_TEMPLATE = """The previously generated Playwright test code failed verification.

Previous test code:
```python
{previous_code}
```

Error encountered:
- Type: {error_type}
- Message: {error_message}
{line_info}
{suggestion_info}
{selector_hint_section}
{screenshot_hint_section}
{diagnostics_section}

Please fix the test code to resolve these issues. Return ONLY the corrected Python test code without any explanations or markdown."""


class UITestAgent:
    """Agent for generating Playwright E2E tests using pytest-playwright."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _extract_code_from_response(self, response: str) -> str:
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        lines = response.strip().split('\n')
        code_lines: List[str] = []
        in_code = False
        for line in lines:
            if line.strip().startswith(('import ', 'from ', 'def ', 'class ', '@', '#')) or in_code:
                in_code = True
                code_lines.append(line)
            elif in_code and (line.strip() == '' or line.startswith(' ') or line.startswith('\t')):
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()

        return response.strip()

    def _extract_test_functions(self, code: str) -> List[str]:
        return re.findall(r'def\s+(test_\w+)\s*\(', code)

    def _extract_imports(self, code: str) -> List[str]:
        imports: List[str] = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(stripped)
        return imports

    def _build_target_section(
        self,
        target_url: Optional[str] = None,
        html_content: Optional[str] = None,
    ) -> str:
        parts: List[str] = []
        if target_url:
            parts.append(f"Target URL: {target_url}")
        if html_content:
            preview = html_content[:3000]
            parts.append(f"HTML content of the page:\n```html\n{preview}\n```")
        if not parts:
            parts.append("Target: a web application accessible at BASE_URL (read from environment).")
        return '\n'.join(parts)

    def generate(
        self,
        description: str,
        target_url: Optional[str] = None,
        html_content: Optional[str] = None,
    ) -> GeneratedUITest:
        target_section = self._build_target_section(target_url, html_content)

        prompt = GENERATION_TEMPLATE.format(
            target_section=target_section,
            description=description,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        test_code = self._extract_code_from_response(response_text)
        test_functions = self._extract_test_functions(test_code)
        imports = self._extract_imports(test_code)

        return GeneratedUITest(
            test_code=test_code,
            imports=imports,
            test_functions=test_functions,
            target_url=target_url,
        )

    def repair(self, context: UIRepairContext) -> GeneratedUITest:
        line_info = f"- Line: {context.line_number}" if context.line_number else ""
        suggestion_info = f"- Suggested fix: {context.suggestion}" if context.suggestion else ""
        selector_hint_section = (
            f"\nSelector hint: {context.selector_hint}" if context.selector_hint else ""
        )
        screenshot_hint_section = (
            f"\nScreenshot evidence: {context.screenshot_hint}" if context.screenshot_hint else ""
        )
        diagnostics_section = (
            f"\nFull verification diagnostics:\n{context.diagnostics}"
            if context.diagnostics else ""
        )

        prompt = REPAIR_TEMPLATE.format(
            previous_code=context.previous_code,
            error_type=context.error_type,
            error_message=context.error_message,
            line_info=line_info,
            suggestion_info=suggestion_info,
            selector_hint_section=selector_hint_section,
            screenshot_hint_section=screenshot_hint_section,
            diagnostics_section=diagnostics_section,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        test_code = self._extract_code_from_response(response_text)
        test_functions = self._extract_test_functions(test_code)
        imports = self._extract_imports(test_code)

        return GeneratedUITest(
            test_code=test_code,
            imports=imports,
            test_functions=test_functions,
        )
