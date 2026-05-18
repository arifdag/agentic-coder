"""Unit Test Agent for generating pytest tests."""

import re
from typing import Any, List, Mapping, Optional

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
    import_module: Optional[str] = Field(
        default=None, description="Real repo module path to import from when available"
    )
    target_function: Optional[str] = Field(
        default=None,
        description="Specific function/class to target in tests when known",
    )
    repair_mode: Optional[str] = Field(
        default=None,
        description="Specific repair branch selected from verification failures",
    )
    diagnostic_note: Optional[str] = Field(
        default=None,
        description="Compact actionable note for the selected repair branch",
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
   - Import the real public target(s) from the module under test.
   - In single-file sandbox runs, use source_module as the import name,
     e.g. from source_module import target.
   - In repo-context runs, use the real module/package path named in the prompt
     or request, e.g. from mypackage.mymodule import target. Do NOT use source_module
     when the prompt specifies a real import path.
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
- Import the real public target(s) from the source module:
  - In single-file sandbox/benchmark runs, use source_module as the import name,
    e.g. from source_module import target.
  - In repo-context runs where the prompt specifies a real module/package path,
    use that exact path, e.g. from mypackage.mymodule import target.
    Do NOT use source_module in repo-context; import the real module path instead.
- If a target function/class is named in the context below, import that exact
  target and make every test call or instantiate it. Do not test broad module
  public API behavior instead of the named target.
- Do NOT redefine targets locally; always import and delegate to the original.
- Call or instantiate the target in assertions to exercise its logic and branches.
- Every test must have meaningful assertions; never assert True, assert 1 == 1, or pass-only bodies.
- Include at least one normal case, one edge case, and one invalid/error case
  when the target behavior makes that possible.
- Cover edge cases: empty inputs, None values, boundary values, and error paths.
- Test expected exceptions with pytest.raises where applicable.
- Use pytest.mark.parametrize for similar test cases.
- Include at least 3-5 test cases per public function/method.

Generate the complete test file now:"""

TESTGENEVAL_SYSTEM_PROMPT = """You are an expert Python test engineer. Generate official TestGenEval-compatible unit tests.
Return only Python code. Do not use pytest-only APIs unless the prompt explicitly allows them."""

TESTGENEVAL_GENERATION_TEMPLATE = """Generate a Python unit test file for an official TestGenEval evaluation case.

Repository context:
- Repository: {repo}
- Version: {version}
- Source file under test: {code_file}
- Existing/target test file path: {test_file}
- Inferred import module: {import_module}
- Benchmark request: {user_request}
{feedback_section}

Source code under test:
```python
{code}
```

Critical output requirements:
- Return NON-EMPTY Python test code only. Do not return markdown, prose, or an empty response.
- Define file-level test functions named test_*.
- Do NOT define test classes, including Test* classes. The official TestGenEval
  postprocessor extracts file-level functions most reliably, and class-based
  output can break Django postprocessing.
- Do NOT import pytest or use pytest.raises. Some official TestGenEval
  containers, including Django, run with unittest and do not install pytest.
- For exception checks, use plain try/except/else with assert statements.
- Import the real target from the repository module.
{import_guidance}
- If the exact symbol name is unclear, inspect the source code and import the public functions/classes it defines.
- Do NOT import from source_module for TestGenEval official runs.
- Do NOT redefine, shadow, copy, stub, monkeypatch away, or reimplement production targets in the test file.
- Every test must call or instantiate the real target and include meaningful assertions or pytest.raises checks.
- Never use dummy assertions such as assert True, assert 1 == 1, or pass-only test bodies.
- Prefer a compact set of 3-8 focused tests over a huge broad test file.
- Avoid network, sleeps, wall-clock timing, randomness without fixed seeds, or external services.
- Keep the test file focused and deterministic.

Return the complete Python test file now, with imports at the top and no surrounding explanation:"""

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
{import_context_section}
{target_function_section}
{repair_mode_section}

Repair rules (apply all that match the diagnostics above):
- relevance / tests_unrelated_to_source / target_not_relevant: The test does not
  reference the target from the source module. Replace irrelevant tests with ones
  that import and call/instantiate the original target.
- target_relevance / target_not_executed: The target body was never executed under
  coverage. Tests must call or instantiate the target, not merely import it.
  Add assertions that exercise the target branches and return values.
- target_shadowed_in_tests: A function/class in the test file shadows the source
  module target. Remove the local redefinition and import the real target instead.
- no_assertions: Tests have no assertions at all. Add meaningful assertions that
  check return values, state changes, or expected exceptions.
- dummy_assertions_only: Tests contain only vacuous assertions (assert True,
  assert 1 == 1, etc.). Replace every dummy assertion with a real assertion that
  verifies target behavior.
- generic_public_api_gaming: Tests only inspect module shape or public API
  availability. Delete broad module API smoke tests and assert directly on
  calls to the requested target.
- PHANTOM-PKG / dependency failure: An import refers to a package not found on PyPI.
  Do NOT invent packages or add fictitious dependencies. Keep imports to the real
  target module and fix only the test code import paths and assertions. If a
  legitimate dependency is missing, note it but do not mock or stub it away.
- infrastructure / import error / ModuleNotFoundError: The test cannot find the
  target module. Check the import path: in single-file sandbox mode use
  source_module; in repo-context use the real module/package path from the prompt.
  Fix only the import path and test code, not the production source.
- no_tests_collected: pytest collected no tests. Ensure the file defines
  at least one top-level test_* function. Discard the previous test-file
  structure if necessary and rebuild a minimal target-focused pytest file.
  Do not hide tests inside classes, conditionals, nested functions, or helpers.
  Also check for syntax errors or import errors that prevent collection.
- timeout / docker_timeout: The test execution timed out. Reduce the number
  of test cases, remove randomized/stress/large-input tests, use tiny deterministic inputs,
  avoid infinite or unbounded loops, and keep only targeted tests that call the requested target.
- low target coverage: Some target lines were covered but coverage is low.
  Add more targeted tests for uncovered branches and error-handling paths.
- assertion_error / test_failure: A test assertion failed or raised an error.
  Do not return an empty file. Preserve passing target-focused tests, remove
  only invalid assumptions, and fix expected values to match current behavior.
- coverage gaps: Add tests that call the target on inputs reaching the uncovered lines.

Target relevance and coverage rules (critical):
- Every test MUST import the real target and call/instantiate it in an assertion.
- In single-file sandbox mode: import from source_module.
- In repo-context mode: import from the real module path specified in the prompt.
- Do NOT write tests that only check importability; they must exercise target logic.
- Do NOT replace meaningful assertions with dummy assertions during repair.
- Never re-implement the source module target locally in the test file.
- NEVER return blank output, prose, markdown-only text, helper-only code, or a
  file without pytest-discoverable test_* functions.

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

    def _build_context_section(
        self,
        file_path: Optional[str] = None,
        import_module: Optional[str] = None,
        target_function: Optional[str] = None,
    ) -> str:
        """Build context section for the prompt.

        Args:
            file_path: Optional file path for context
            import_module: Exact real module path for repo-context execution
            target_function: Specific function/class to focus tests on

        Returns:
            Context section string
        """
        sections = []

        if import_module:
            sections.append(
                "Repo-context import module: "
                f"{import_module}\n"
                f"Use this exact module path in imports, e.g. "
                f"from {import_module} import <target>. "
                "Do NOT import from source_module in repo-context runs."
            )

        if target_function:
            import_path = import_module if import_module else "source_module"
            sections.append(
                f"Target function/class: {target_function}\n"
                f"You MUST import {target_function} directly from {import_path} and "
                "call or instantiate it inside every test.\n"
                "Do NOT write broad module API smoke tests; focus every test on "
                f"{target_function}.\n"
                "Cover normal inputs, edge cases, and invalid-or-error inputs "
                "where possible."
            )

        if file_path:
            module_name = file_path.replace(".py", "").replace("/", ".").replace("\\", ".")
            if module_name.startswith("."):
                module_name = module_name[1:]
            sections.append(f"Module to import: {module_name}")

        return "\n".join(sections) if sections else ""

    def _metadata_value(self, metadata: Mapping[str, Any], key: str, default: str) -> str:
        value = metadata.get(key)
        if value is None or value == "":
            return default
        return str(value)

    def generate(
        self,
        code: str,
        file_path: Optional[str] = None,
        import_module: Optional[str] = None,
        target_function: Optional[str] = None,
    ) -> GeneratedTest:
        """Generate unit tests for the given code.

        Args:
            code: Source code to generate tests for
            file_path: Optional file path for import context
            import_module: Exact real module path for repo-context execution
            target_function: Specific function/class to focus tests on

        Returns:
            Generated test result
        """
        context_section = self._build_context_section(file_path, import_module, target_function)

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

    def generate_testgeneval(
        self,
        code: str,
        metadata: Mapping[str, Any],
        user_request: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> GeneratedTest:
        """Generate a TestGenEval official-compatible pytest prediction."""
        import_module = self._metadata_value(metadata, "import_module", "unknown")
        code_file = self._metadata_value(metadata, "code_file", "unknown")
        if import_module != "unknown":
            import_guidance = f"- Prefer:\n  from {import_module} import <public function or class>"
        else:
            import_guidance = (
                f"- Infer the importable module from source file {code_file}; "
                "never use source_module."
            )
        feedback_section = ""
        if feedback:
            feedback_section = (
                "\nPrevious attempt was rejected by local validation:\n"
                f"- {feedback}\n"
                "Correct the issue in this attempt."
            )
        prompt = TESTGENEVAL_GENERATION_TEMPLATE.format(
            repo=self._metadata_value(metadata, "repo", "unknown"),
            version=self._metadata_value(metadata, "version", "unknown"),
            code_file=code_file,
            test_file=self._metadata_value(metadata, "test_file", "unknown"),
            import_module=import_module,
            import_guidance=import_guidance,
            user_request=user_request or "Generate pytest unit tests for the source code.",
            feedback_section=feedback_section,
            code=code,
        )

        messages = [
            SystemMessage(content=TESTGENEVAL_SYSTEM_PROMPT),
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

        import_context_section = ""
        if context.import_module:
            import_context_section = (
                "\nRepo-context import module:\n"
                f"- Import from: {context.import_module}\n"
                "- Do NOT use source_module in repo-context repairs.\n"
                f"- Example form: from {context.import_module} import <target>"
            )

        target_function_section = ""
        if context.target_function:
            import_path = context.import_module if context.import_module else "source_module"
            target_function_section = (
                f"\nTarget function/class: {context.target_function}\n"
                f"- You MUST import {context.target_function} directly from {import_path}.\n"
                f"- Call or instantiate {context.target_function} inside every test assertion.\n"
                "- Do NOT write broad module API smoke tests; focus every test on "
                f"{context.target_function}.\n"
                "- Cover normal inputs, edge cases, and invalid-or-error input cases "
                "where possible."
            )

        repair_mode_section = ""
        if context.repair_mode or context.diagnostic_note:
            parts = ["\nSelected repair mode:"]
            if context.repair_mode:
                parts.append(f"- Mode: {context.repair_mode}")
            if context.diagnostic_note:
                parts.append(f"- Action: {context.diagnostic_note}")
            repair_mode_section = "\n".join(parts)

        prompt = REPAIR_TEMPLATE.format(
            previous_code=context.previous_code,
            error_type=context.error_type,
            error_message=context.error_message,
            line_info=line_info,
            suggestion_info=suggestion_info,
            diagnostics_section=diagnostics_section,
            coverage_section=coverage_section,
            import_context_section=import_context_section,
            target_function_section=target_function_section,
            repair_mode_section=repair_mode_section,
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
