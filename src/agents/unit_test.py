"""Unit Test Agent for generating pytest tests."""

import re
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


class GeneratedTest(BaseModel):
    """Result of test generation."""
    
    test_code: str = Field(description="Generated test code")
    imports: List[str] = Field(default_factory=list, description="Required imports")
    test_functions: List[str] = Field(default_factory=list, description="Names of test functions")
    assumptions: Optional[str] = Field(default=None, description="Assumptions made during generation")


class RepairContext(BaseModel):
    """Context for repair iteration."""
    
    previous_code: str = Field(description="Previously generated test code")
    error_type: str = Field(description="Type of error encountered")
    error_message: str = Field(description="Error message from verification")
    line_number: Optional[int] = Field(default=None, description="Line number of error if available")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix approach")
    coverage_gaps: Optional[str] = Field(default=None, description="Uncovered lines from coverage report")
    diagnostics: Optional[str] = Field(default=None, description="Full structured diagnostics from verification report")


SYSTEM_PROMPT = """You are an expert Python test engineer. Your task is to generate high-quality pytest unit tests.

Guidelines:
1. Generate comprehensive test cases covering:
   - Normal/happy path scenarios
   - Edge cases and boundary conditions
   - Error handling and exceptions
   - Input validation

2. Follow pytest best practices:
   - Use descriptive test function names (test_<function>_<scenario>)
   - Use pytest.raises for exception testing
   - Use parametrize for multiple similar test cases when appropriate
   - Keep tests independent and deterministic

3. Code quality:
   - Include necessary imports
   - Add brief docstrings explaining what each test verifies
   - Avoid external dependencies (network, filesystem) unless testing that specifically
   - Use fixtures for common setup when needed

4. Output format:
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
- Test all public functions/methods
- Include at least 3-5 test cases per function
- Cover edge cases: empty inputs, None values, boundary values
- Test expected exceptions where applicable
- Use pytest.mark.parametrize for similar test cases

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

Please fix the test code to resolve these issues. Return ONLY the corrected Python test code without any explanations or markdown."""


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
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        lines = response.strip().split('\n')
        code_lines = []
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
        """Extract test function names from code.
        
        Args:
            code: Test code
            
        Returns:
            List of test function names
        """
        pattern = r'def\s+(test_\w+)\s*\('
        return re.findall(pattern, code)
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code.
        
        Args:
            code: Test code
            
        Returns:
            List of import lines
        """
        imports = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
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
            module_name = file_path.replace('.py', '').replace('/', '.').replace('\\', '.')
            if module_name.startswith('.'):
                module_name = module_name[1:]
            sections.append(f"Module to import: {module_name}")
        
        return '\n'.join(sections) if sections else ""
    
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
        response_text = response.content if hasattr(response, 'content') else str(response)
        
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
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        test_code = self._extract_code_from_response(response_text)
        test_functions = self._extract_test_functions(test_code)
        imports = self._extract_imports(test_code)
        
        return GeneratedTest(
            test_code=test_code,
            imports=imports,
            test_functions=test_functions,
        )
