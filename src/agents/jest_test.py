"""Jest Test Agent for generating JavaScript/TypeScript unit tests."""

import re
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .unit_test import GeneratedTest, RepairContext

SYSTEM_PROMPT = """You are an expert JavaScript/TypeScript test engineer. Your task is to generate high-quality Jest unit tests.

Guidelines:
1. Generate comprehensive test cases covering:
   - Normal/happy path scenarios
   - Edge cases and boundary conditions
   - Error handling and exceptions
   - Input validation

2. Follow Jest best practices:
   - Use describe() blocks to group related tests
   - Use it() or test() with descriptive names
   - Use expect() with appropriate matchers (toBe, toEqual, toThrow, etc.)
   - Use beforeEach/afterEach for setup/teardown
   - Use jest.fn() for mocking when needed

3. Code quality:
   - Include necessary require() or import statements
   - Import the source module as: const mod = require('./source_module');
   - Ground expectations in the provided source. Do NOT call methods,
     constructors, properties, or exported names that are not implemented.
   - Use toThrow only when the source clearly throws for that input; do not
     assume stricter validation than the source implements.
   - Keep tests independent and deterministic
   - Avoid external dependencies (network, filesystem) unless testing that specifically

4. Output format:
   - Return ONLY the JavaScript test code
   - Start with require/import statements
   - Do not include markdown code blocks or explanations
   - The code should be immediately executable with Jest
"""

GENERATION_TEMPLATE = """Generate Jest unit tests for the following JavaScript/TypeScript code:

```javascript
{code}
```

{context_section}

Requirements:
- If a target export is named in the context below, import and test only that
  export. Do not generate broad module-shape tests or tests for every exported
  constant in a large module.
- Test all exported functions/methods only when no target export is named.
- Only test APIs that are actually exported or reachable from the provided source.
- Do not invent methods, properties, validation branches, or error behavior that
  does not appear in the source.
- Include at least 3-5 focused test cases for the named target or per function
- Cover edge cases: empty inputs, null/undefined values, boundary values
- Test expected exceptions only where the source clearly throws.
- Use describe() blocks to group tests by function

Generate the complete test file now:"""

REPAIR_TEMPLATE = """The previously generated test code failed verification.

Previous test code:
```javascript
{previous_code}
```

Error encountered:
- Type: {error_type}
- Message: {error_message}
{line_info}
{suggestion_info}
{diagnostics_section}
{target_section}

Repair rules:
- For assertion/test failures, preserve passing source-grounded tests and remove
  only invalid expectations.
- For TypeError/ReferenceError/import failures, remove calls to nonexistent APIs
  and import only exports that the source actually provides.
- For SyntaxError or unterminated string/template failures, rebuild the file as
  a small syntactically valid Jest suite; avoid multiline test names and huge
  generated suites.
- For toThrow failures, keep exception assertions only when the source clearly
  throws for that input; otherwise assert the implemented behavior.
- Do not replace failing tests with smoke tests that only check the module shape.

Please fix the test code to resolve these issues. Return ONLY the corrected JavaScript test code without any explanations or markdown."""


class JestTestAgent:
    """Agent for generating JavaScript/TypeScript unit tests using Jest."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _extract_code_from_response(self, response: str) -> str:
        """Extract JavaScript code from LLM response."""
        code_block_pattern = r"```(?:javascript|typescript|js|ts)?\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        lines = response.strip().split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith(
                    (
                        "const ",
                        "let ",
                        "var ",
                        "import ",
                        "require",
                        "describe(",
                        "it(",
                        "test(",
                        "function ",
                        "module.",
                        "//",
                        "/*",
                    )
                )
                or in_code
            ):
                in_code = True
                code_lines.append(line)
            elif in_code and (stripped == "" or line.startswith(" ") or line.startswith("\t")):
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        return response.strip()

    def _extract_test_functions(self, code: str) -> List[str]:
        """Extract test function names from Jest test code."""
        names = []
        for pattern in [
            r"""(?:it|test)\s*\(\s*['"](.+?)['"]""",
            r"""(?:it|test)\s*\(\s*`(.+?)`""",
        ]:
            names.extend(re.findall(pattern, code))
        return names

    def _extract_imports(self, code: str) -> List[str]:
        """Extract import/require statements from code."""
        imports = []
        for line in code.split("\n"):
            stripped = line.strip()
            if (
                stripped.startswith("import ")
                or stripped.startswith("const ")
                and "require(" in stripped
                or stripped.startswith("let ")
                and "require(" in stripped
                or stripped.startswith("var ")
                and "require(" in stripped
            ):
                imports.append(stripped)
        return imports

    def generate(
        self,
        code: str,
        file_path: Optional[str] = None,
        target_function: Optional[str] = None,
    ) -> GeneratedTest:
        context_section = ""
        if file_path:
            context_section = f"Source file: {file_path}"
        if target_function:
            target_context = (
                f"Target export: {target_function}\n"
                f"Import it with: const {{ {target_function} }} = require('./source_module');\n"
                f"Focus every test on {target_function}. Do not test every export in the module."
            )
            context_section = (
                f"{context_section}\n{target_context}" if context_section else target_context
            )

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

    def repair(self, context: RepairContext) -> GeneratedTest:
        line_info = ""
        if context.line_number:
            line_info = f"- Line: {context.line_number}"

        suggestion_info = ""
        if context.suggestion:
            suggestion_info = f"- Suggested fix: {context.suggestion}"

        diagnostics_section = ""
        if context.diagnostics:
            diagnostics_section = f"\nFull verification diagnostics:\n{context.diagnostics}"

        target_section = ""
        if context.target_function:
            target_section = (
                "\nTarget export:\n"
                f"- Import only the target export with: "
                f"const {{ {context.target_function} }} = require('./source_module');\n"
                f"- Focus every test on {context.target_function}; do not test every export."
            )

        prompt = REPAIR_TEMPLATE.format(
            previous_code=context.previous_code,
            error_type=context.error_type,
            error_message=context.error_message,
            line_info=line_info,
            suggestion_info=suggestion_info,
            diagnostics_section=diagnostics_section,
            target_section=target_section,
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
