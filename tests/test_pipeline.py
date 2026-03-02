"""Tests for the LLM Agent Platform pipeline components."""

import pytest
from pathlib import Path

from src.agents.router import RouterAgent, Language, TaskType
from src.agents.unit_test import UnitTestAgent, RepairContext


class TestRouterAgent:
    """Tests for the RouterAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = RouterAgent()
    
    def test_detect_python_by_extension(self):
        """Test Python detection via file extension."""
        result = self.router.detect_language("some code", "test.py")
        assert result == Language.PYTHON
    
    def test_detect_python_by_syntax(self):
        """Test Python detection via syntax patterns."""
        code = '''
def hello():
    print("Hello, world!")

class MyClass:
    def __init__(self):
        self.value = None
'''
        result = self.router.detect_language(code)
        assert result == Language.PYTHON
    
    def test_detect_javascript_by_syntax(self):
        """Test JavaScript detection via syntax patterns."""
        code = '''
const hello = () => {
    console.log("Hello");
};

function greet(name) {
    return `Hello, ${name}`;
}
'''
        result = self.router.detect_language(code)
        assert result == Language.JAVASCRIPT
    
    def test_detect_typescript_by_syntax(self):
        """Test TypeScript detection via type annotations."""
        code = '''
interface User {
    name: string;
    age: number;
}

const greet = (user: User): string => {
    return `Hello, ${user.name}`;
};
'''
        result = self.router.detect_language(code)
        assert result == Language.TYPESCRIPT
    
    def test_classify_unit_test_task(self):
        """Test unit test task classification."""
        result = self.router.classify_task("Generate unit tests for this function")
        assert result == TaskType.UNIT_TEST
    
    def test_classify_ui_test_task(self):
        """Test UI test task classification."""
        result = self.router.classify_task("Create e2e tests with playwright")
        assert result == TaskType.UI_TEST
    
    def test_classify_explanation_task(self):
        """Test explanation task classification."""
        result = self.router.classify_task("Explain what this code does")
        assert result == TaskType.EXPLANATION
    
    def test_route_python_unit_test(self):
        """Test full routing for Python unit tests."""
        code = '''
def add(a, b):
    return a + b
'''
        result = self.router.route(code, "Generate tests", "utils.py")
        
        assert result.language == Language.PYTHON
        assert result.task_type == TaskType.UNIT_TEST
        assert result.framework_hint == "pytest"
        assert result.confidence == 1.0
    
    def test_route_unknown_language(self):
        """Test routing with unknown language."""
        result = self.router.route("random text", "test this")
        
        assert result.language == Language.UNKNOWN
        assert result.confidence == 0.5


class TestUnitTestAgentParsing:
    """Tests for UnitTestAgent parsing methods."""
    
    def test_extract_test_functions(self):
        """Test extraction of test function names."""
        code = '''
def test_add_positive():
    assert add(1, 2) == 3

def test_add_negative():
    assert add(-1, -2) == -3

def helper_function():
    pass

def test_add_zero():
    assert add(0, 0) == 0
'''
        agent = UnitTestAgent.__new__(UnitTestAgent)
        result = agent._extract_test_functions(code)
        
        assert len(result) == 3
        assert "test_add_positive" in result
        assert "test_add_negative" in result
        assert "test_add_zero" in result
        assert "helper_function" not in result
    
    def test_extract_imports(self):
        """Test extraction of import statements."""
        code = '''
import pytest
from source_module import add, subtract
import os
from pathlib import Path

def test_something():
    pass
'''
        agent = UnitTestAgent.__new__(UnitTestAgent)
        result = agent._extract_imports(code)
        
        assert len(result) == 4
        assert "import pytest" in result
        assert "from source_module import add, subtract" in result
    
    def test_extract_code_from_markdown(self):
        """Test extraction of code from markdown blocks."""
        response = '''
Here are the tests:

```python
import pytest

def test_example():
    assert True
```

These tests cover the basic functionality.
'''
        agent = UnitTestAgent.__new__(UnitTestAgent)
        result = agent._extract_code_from_response(response)
        
        assert "import pytest" in result
        assert "def test_example" in result
        assert "Here are the tests" not in result


class TestRepairContext:
    """Tests for RepairContext model."""
    
    def test_repair_context_creation(self):
        """Test RepairContext model creation."""
        context = RepairContext(
            previous_code="def test(): pass",
            error_type="syntax_error",
            error_message="Invalid syntax",
            line_number=1,
        )
        
        assert context.previous_code == "def test(): pass"
        assert context.error_type == "syntax_error"
        assert context.line_number == 1


class TestIntegration:
    """Integration tests (require API key to run)."""
    
    @pytest.mark.skip(reason="Requires API key")
    def test_full_pipeline(self):
        """Test the full pipeline end-to-end."""
        from src.graph.pipeline import run_pipeline
        from src.config import Config
        
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
