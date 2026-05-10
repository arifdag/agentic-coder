"""Unit Test Agent for generating pytest tests."""

import re
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class GeneratedTest(BaseModel):
    """Result of test generation."""

    test_code: str = Field(description="Generated test code")
    imports: List[str] = Field(default_factory=list, description="Required imports")
    test_functions: List[str] = Field(default_factory=list, description="Names of test functions")
    assumptions: Optional[str] = Field(
        default=None, description="Assumptions made during generation"
    )


class RepairContext(BaseModel):
    """Context for repair iteration."""

    previous_code: str = Field(description="Previously generated test code")
    error_type: str = Field(description="Type of error encountered")
    error_message: str = Field(description="Error message from verification")
    line_number: Optional[int] = Field(
        default=None, description="Line number of error if available"
    )
    suggestion: Optional[str] = Field(default=None, description="Suggested fix approach")
    coverage_gaps: Optional[str] = Field(
        default=None, description="Uncovered lines from coverage report"
    )
    diagnostics: Optional[str] = Field(
        default=None, description="Full structured diagnostics from verification report"
    )


SYSTEM_PROMPT = """You are an expert Python test engineer. Your task is to generate high-quality pytest unit tests.

Guidelines:
1. Test quality and coverage:
   - Cover normal/happy paths, edge cases, boundary conditions, and error handling.
   - Every test MUST contain meaningful assertions (e.g., assert result == expected,
     assert isinstance(...), pytest.raises) or verify side effects.
   - NEVER emit dummy assertions: assert True, assert 1 == 1, assert False == False,
     pass-only test bodies, or tests that only check importability without calling the target.
   - Tests must execute branches and logic of the code under test, not merely import it.

2. Target relevance (critical):
   - Import the real public target(s) from the module under test. In benchmark/sandbox
     runs, that module is named source_module, e.g. from source_module import target.
   - Call or instantiate the target in your test assertions to exercise its body.
   - Do NOT redefine, shadow, or re-implement functions/classes from the source module
     inside the test file. Always import and delegate to the original implementation.
   - If the prompt names a source module, import from that module - do not substitute
     a local mock implementation.

3. Follow pytest best practices:
   - Use descriptive test function names (test_<function>_<scenario>)
   - Use pytest.raises for exception testing
   - Use parametrize for multiple similar test cases when appropriate
   - Keep tests independent and deterministic

4. Code quality:
   - Include necessary imports
   - Add brief docstrings explaining what each test verifies
   - Avoid external dependencies (network, filesystem) unless testing that specifically
   - Use fixtures for common setup when needed

5. Output format:
   - Return ONLY the Python test code
   - Start with imports
   - Do not include markdown code blocks or explanations
   - The code should be immediately executable with pytest
"""

GENERATION_TEMPLATE = """Generate pytest unit tests for the following Python code:

```python
{code}
```

{context_section}

Requirements:
- Import the real public target(s) from the source module. In benchmark/sandbox runs,
  use source_module, e.g. from source_module import target. Do NOT redefine targets locally.
- Call or instantiate the target in assertions to exercise its logic and branches.
- Every test must have meaningful assertions; never assert True, assert 1 == 1, or pass-only bodies.
- Cover edge cases: empty inputs, None values, boundary values, and error paths.
- Test expected exceptions with pytest.raises where applicable.
- Use pytest.mark.parametrize for similar test cases.
- Include at least 3-5 test cases per public function/method.

Generate the complete test file now:"""

REPAIR_TEMPLATE = """The previously generated test code failed verification.

Previous test code:
```python
{previous_code}
```

Error encountered:
- Type: {error_type}
- Message: {error_message}
{line_info}
{suggestion_info}
{diagnostics_section}
{coverage_section}

Repair rules (apply all that match the diagnostics above):
- relevance / tests_unrelated_to_source / target_not_relevant: The test does not reference the target from the source module.
  Replace irrelevant tests with ones that import and call/instantiate the original target.
- target_relevance / target_not_executed: The target body was never executed under coverage.
  Tests must call or instantiate the target, not merely import it. Add assertions that
  exercise the target's branches and return values.
- target_shadowed_in_tests: A function/class in the test file shadows the source module target.
  Remove the local redefinition and import the real target from the source module instead.
- no_assertions: Tests have no assertions at all. Add meaningful assertions that check
  return values, state changes, or expected exceptions.
- dummy_assertions_only: Tests contain only vacuous assertions (assert True, assert 1 == 1, etc.).
  Replace every dummy assertion with a real assertion that verifies target behavior.
- coverage gaps: Add tests that call the target on inputs reaching the uncovered lines.

Always import the real target from the source module; in benchmark/sandbox runs, use source_module.
Never re-implement it locally.
Return ONLY the corrected Python test code without any explanations or markdown."""


class UnitTestAgent:
    """Agent for generating Python unit tests using pytest."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the unit test agent.

        Args:
            llm: Language model to use for generation
        """
        self.llm = llm

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned Python code
        """
        code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        lines = response.strip().split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith(("import ", "from ", "def ", "class ", "@", "#")) or in_code:
                in_code = True
                code_lines.append(line)
            elif in_code and (line.strip() == "" or line.startswith(" ") or line.startswith("\t")):
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        return response.strip()

    def _extract_test_functions(self, code: str) -> List[str]:
        """Extract test function names from code.

        Args:
            code: Test code

        Returns:
            List of test function names
        """
        pattern = r"def\s+(test_\w+)\s*\("
        return re.findall(pattern, code)

    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code.

        Args:
            code: Test code

        Returns:
            List of import lines
        """
        imports = []
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
        return imports

    def _build_context_section(self, file_path: Optional[str] = None) -> str:
        """Build context section for the prompt.

        Args:
            file_path: Optional file path for context

        Returns:
            Context section string
        """
        sections = []

        if file_path:
            module_name = file_path.replace(".py", "").replace("/", ".").replace("\\", ".")
            if module_name.startswith("."):
                module_name = module_name[1:]
            sections.append(f"Module to import: {module_name}")

        return "\n".join(sections) if sections else ""

    def generate(
        self,
        code: str,
        file_path: Optional[str] = None,
    ) -> GeneratedTest:
        """Generate unit tests for the given code.

        Args:
            code: Source code to generate tests for
            file_path: Optional file path for import context

        Returns:
            Generated test result
        """
        context_section = self._build_context_section(file_path)

        prompt = GENERATION_TEMPLATE.format(
            code=code,
            context_section=context_section,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        test_code = self._extract_code_from_response(response_text)
        test_functions = self._extract_test_functions(test_code)
        imports = self._extract_imports(test_code)

        return GeneratedTest(
            test_code=test_code,
            imports=imports,
            test_functions=test_functions,
        )

    def repair(
        self,
        context: RepairContext,
    ) -> GeneratedTest:
        """Repair failed test code based on error feedback.

        Args:
            context: Repair context with error information

        Returns:
            Repaired test result
        """
        line_info = ""
        if context.line_number:
            line_info = f"- Line: {context.line_number}"

        suggestion_info = ""
        if context.suggestion:
            suggestion_info = f"- Suggested fix: {context.suggestion}"

        diagnostics_section = ""
        if context.diagnostics:
            diagnostics_section = f"\nFull verification diagnostics:\n{context.diagnostics}"

        coverage_section = ""
        if context.coverage_gaps:
            coverage_section = (
                f"\nCoverage gaps (lines not covered): {context.coverage_gaps}\n"
                f"Please add tests targeting these uncovered lines."
            )

        prompt = REPAIR_TEMPLATE.format(
            previous_code=context.previous_code,
            error_type=context.error_type,
            error_message=context.error_message,
            line_info=line_info,
            suggestion_info=suggestion_info,
            diagnostics_section=diagnostics_section,
            coverage_section=coverage_section,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        test_code = self._extract_code_from_response(response_text)
        test_functions = self._extract_test_functions(test_code)
        imports = self._extract_imports(test_code)

        return GeneratedTest(
            test_code=test_code,
            imports=imports,
            test_functions=test_functions,
        )
