"""
Tests for State Schemas

Validates that all Pydantic schemas work correctly with proper data
and reject invalid data appropriately.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from src.core.schemas import (
    UserRequest,
    CodeDraft,
    CaseResult,
    ExecutionSummary,
    ErrorLog,
    GraphState,
    SupportedLanguage,
    create_initial_state,
)


class TestUserRequest:
    """Tests for UserRequest schema."""
    
    def test_valid_request(self):
        """Test creating a valid user request."""
        request = UserRequest(
            request_id=str(uuid4()),
            problem_description="Write a function to calculate factorial",
            language=SupportedLanguage.PYTHON,
        )
        
        assert request.problem_description == "Write a function to calculate factorial"
        assert request.language == SupportedLanguage.PYTHON
        assert request.constraints is None
        assert isinstance(request.timestamp, datetime)
    
    def test_request_with_constraints(self):
        """Test request with optional fields."""
        request = UserRequest(
            request_id="test-123",
            problem_description="Sort an array",
            language=SupportedLanguage.JAVA,
            constraints=["O(n log n) complexity", "In-place sorting"],
            context="int[] arr = {3, 1, 4, 1, 5};",
        )
        
        assert len(request.constraints) == 2
        assert request.context is not None
    
    def test_invalid_language(self):
        """Test that invalid language raises error."""
        with pytest.raises(ValueError):
            UserRequest(
                request_id="test",
                problem_description="Test",
                language="ruby",  # Not supported
            )
    
    def test_missing_required_fields(self):
        """Test that missing required fields raises error."""
        with pytest.raises(ValueError):
            UserRequest(language=SupportedLanguage.PYTHON)


class TestCodeDraft:
    """Tests for CodeDraft schema."""
    
    def test_valid_draft(self):
        """Test creating a valid code draft."""
        draft = CodeDraft(
            draft_id=str(uuid4()),
            request_id="req-123",
            code="def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
            language=SupportedLanguage.PYTHON,
        )
        
        assert draft.version == 1
        assert "factorial" in draft.code
        assert draft.explanation is None
    
    def test_draft_with_metadata(self):
        """Test draft with all optional fields."""
        draft = CodeDraft(
            draft_id="draft-1",
            request_id="req-1",
            code="print('hello')",
            language=SupportedLanguage.PYTHON,
            version=2,
            explanation="Simple hello world program",
            dependencies=["numpy"],
            entry_point="main",
        )
        
        assert draft.version == 2
        assert "numpy" in draft.dependencies


class TestExecutionSummarySchema:
    """Tests for CaseResult and ExecutionSummary schemas."""
    
    def test_single_test_result(self):
        """Test creating a single test result."""
        result = CaseResult(
            test_name="test_factorial_5",
            passed=True,
            input_data="5",
            expected_output="120",
            actual_output="120",
            execution_time_ms=1.5,
        )
        
        assert result.passed is True
        assert result.error_message is None
    
    def test_failed_test_result(self):
        """Test a failed test result."""
        result = CaseResult(
            test_name="test_factorial_negative",
            passed=False,
            input_data="-1",
            expected_output="Error",
            actual_output="RecursionError",
            error_message="Maximum recursion depth exceeded",
        )
        
        assert result.passed is False
        assert result.error_message is not None
    
    def test_aggregated_results(self):
        """Test aggregated test results."""
        results = ExecutionSummary(
            draft_id="draft-1",
            total_tests=3,
            passed_tests=2,
            failed_tests=1,
            results=[
                CaseResult(test_name="test_1", passed=True),
                CaseResult(test_name="test_2", passed=True),
                CaseResult(test_name="test_3", passed=False),
            ],
            total_execution_time_ms=15.0,
        )
        
        assert results.success_rate == pytest.approx(66.67, rel=0.1)
        assert results.all_passed is False
    
    def test_all_passed(self):
        """Test when all tests pass."""
        results = ExecutionSummary(
            draft_id="draft-1",
            total_tests=2,
            passed_tests=2,
            failed_tests=0,
        )
        
        assert results.all_passed is True
        assert results.success_rate == 100.0


class TestErrorLog:
    """Tests for ErrorLog schema."""
    
    def test_valid_error_log(self):
        """Test creating a valid error log."""
        error = ErrorLog(
            error_id=str(uuid4()),
            draft_id="draft-1",
            error_type="TypeError",
            message="unsupported operand type(s) for +: 'int' and 'str'",
            language=SupportedLanguage.PYTHON,
        )
        
        assert error.resolved is False
        assert error.line_number is None
    
    def test_error_with_traceback(self):
        """Test error with full traceback."""
        traceback = '''Traceback (most recent call last):
  File "main.py", line 5, in <module>
    result = 1 + "hello"
TypeError: unsupported operand type(s) for +: 'int' and 'str'
'''
        error = ErrorLog(
            error_id="err-1",
            draft_id="draft-1",
            error_type="TypeError",
            message="unsupported operand type(s)",
            language=SupportedLanguage.PYTHON,
            traceback=traceback,
            line_number=5,
        )
        
        assert error.line_number == 5
        assert "main.py" in error.traceback
    
    def test_to_embedding_text(self):
        """Test embedding text generation."""
        error = ErrorLog(
            error_id="err-1",
            draft_id="draft-1",
            error_type="KeyError",
            message="'name'",
            language=SupportedLanguage.PYTHON,
        )
        
        text = error.to_embedding_text()
        assert "Language: python" in text
        assert "Error Type: KeyError" in text
        assert "Message: 'name'" in text


class TestGraphState:
    """Tests for GraphState TypedDict."""
    
    def test_create_initial_state(self):
        """Test creating initial graph state."""
        request = UserRequest(
            request_id="req-1",
            problem_description="Test problem",
            language=SupportedLanguage.PYTHON,
        )
        
        state = create_initial_state(request, max_retries=5)
        
        assert state["current_request"] == request
        assert state["code_drafts"] == []
        assert state["retry_count"] == 0
        assert state["max_retries"] == 5
        assert state["current_stage"] == "initialized"
    
    def test_state_is_mutable(self):
        """Test that state can be updated."""
        request = UserRequest(
            request_id="req-1",
            problem_description="Test",
            language=SupportedLanguage.PYTHON,
        )
        
        state = create_initial_state(request)
        state["current_stage"] = "generating"
        state["retry_count"] = 1
        
        assert state["current_stage"] == "generating"
        assert state["retry_count"] == 1
