"""
State Schemas for LangGraph Agent System

Defines all data structures for inter-node communication in the LangGraph workflow.
These schemas represent the state that flows between nodes in the agent graph.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph.message import add_messages


class SupportedLanguage(str, Enum):
    """Supported programming languages for code generation and execution."""
    PYTHON = "python"
    JAVA = "java"
    CSHARP = "csharp"


class UserRequest(BaseModel):
    """
    Represents an incoming user request for code generation or problem solving.
    
    This is the entry point state for the agent workflow.
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    problem_description: str = Field(..., description="Natural language description of the problem")
    language: SupportedLanguage = Field(
        default=SupportedLanguage.PYTHON,
        description="Target programming language"
    )
    constraints: Optional[list[str]] = Field(
        default=None,
        description="Optional constraints or requirements for the solution"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context or existing code snippets"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Request creation timestamp"
    )

    model_config = ConfigDict(use_enum_values=True)


class CodeDraft(BaseModel):
    """
    Represents generated code output from the coding agent.
    
    Contains the code itself along with metadata about the generation.
    """
    draft_id: str = Field(..., description="Unique identifier for this code draft")
    request_id: str = Field(..., description="Reference to the original request")
    code: str = Field(..., description="The generated source code")
    language: SupportedLanguage = Field(..., description="Programming language of the code")
    version: int = Field(default=1, description="Draft version number for iterations")
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation of the code logic and approach"
    )
    dependencies: Optional[list[str]] = Field(
        default=None,
        description="Required libraries or packages"
    )
    entry_point: Optional[str] = Field(
        default=None,
        description="Main function or entry point for execution"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Draft creation timestamp"
    )

    model_config = ConfigDict(use_enum_values=True)


class CaseResult(BaseModel):
    """
    Represents the result of a single test case execution.
    """
    test_name: str = Field(..., description="Name or identifier of the test")
    passed: bool = Field(..., description="Whether the test passed")
    input_data: Optional[str] = Field(default=None, description="Test input data")
    expected_output: Optional[str] = Field(default=None, description="Expected output")
    actual_output: Optional[str] = Field(default=None, description="Actual output from execution")
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Test execution time in milliseconds"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if test failed"
    )


class ExecutionSummary(BaseModel):
    """
    Aggregates all test results for a code draft execution.
    
    Used to evaluate the correctness of generated code.
    """
    draft_id: str = Field(..., description="Reference to the tested code draft")
    total_tests: int = Field(..., description="Total number of tests run")
    passed_tests: int = Field(..., description="Number of tests that passed")
    failed_tests: int = Field(..., description="Number of tests that failed")
    results: list[CaseResult] = Field(
        default_factory=list,
        description="Individual test results"
    )
    total_execution_time_ms: float = Field(
        default=0.0,
        description="Total execution time for all tests"
    )
    executed_at: datetime = Field(
        default_factory=datetime.now,
        description="Test execution timestamp"
    )

    @property
    def success_rate(self) -> float:
        """Calculate the test success rate as a percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.passed_tests == self.total_tests


class ErrorLog(BaseModel):
    """
    Represents an error that occurred during code execution or compilation.
    
    Used for error analysis and retrieval of similar error solutions.
    """
    error_id: str = Field(..., description="Unique identifier for this error")
    draft_id: str = Field(..., description="Reference to the code draft that caused the error")
    error_type: str = Field(..., description="Type/category of the error (e.g., SyntaxError, RuntimeError)")
    message: str = Field(..., description="Error message")
    traceback: Optional[str] = Field(
        default=None,
        description="Full stack trace if available"
    )
    line_number: Optional[int] = Field(
        default=None,
        description="Line number where error occurred"
    )
    language: SupportedLanguage = Field(..., description="Programming language")
    occurred_at: datetime = Field(
        default_factory=datetime.now,
        description="Error occurrence timestamp"
    )
    resolved: bool = Field(
        default=False,
        description="Whether this error has been resolved"
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Description of how the error was resolved"
    )

    model_config = ConfigDict(use_enum_values=True)

    def to_embedding_text(self) -> str:
        """
        Generate text representation for vector embedding.
        Used for similarity search in error solution retrieval.
        """
        parts = [
            f"Language: {self.language}",
            f"Error Type: {self.error_type}",
            f"Message: {self.message}",
        ]
        if self.traceback:
            parts.append(f"Traceback: {self.traceback}")
        return "\n".join(parts)


class GraphState(TypedDict, total=False):
    """
    Unified state for LangGraph node communication.
    
    This TypedDict defines the complete state that flows through the agent graph.
    Each node can read from and write to specific parts of this state.
    
    Uses TypedDict for LangGraph compatibility with total=False to allow
    partial state updates.
    """
    # Message history for conversational agents
    messages: Annotated[list, add_messages]
    
    # Current request being processed
    current_request: Optional[UserRequest]
    
    # Generated code drafts (supports multiple iterations)
    code_drafts: list[CodeDraft]
    
    # Test results for each draft
    test_results: list[ExecutionSummary]
    
    # Error logs encountered during execution
    error_logs: list[ErrorLog]
    
    # Retrieved similar error solutions (Few-Shot Examples)
    similar_solutions: list[dict[str, Any]]
    
    # Current workflow stage
    current_stage: str
    
    # Number of retry attempts
    retry_count: int
    
    # Maximum allowed retries
    max_retries: int
    
    # Final output/response
    final_response: Optional[str]


def create_initial_state(request: UserRequest, max_retries: int = 3) -> GraphState:
    """
    Factory function to create an initial GraphState for a new request.
    
    Args:
        request: The incoming user request
        max_retries: Maximum number of code generation retries
        
    Returns:
        Initialized GraphState ready for the workflow
    """
    return GraphState(
        messages=[],
        current_request=request,
        code_drafts=[],
        test_results=[],
        error_logs=[],
        similar_solutions=[],
        current_stage="initialized",
        retry_count=0,
        max_retries=max_retries,
        final_response=None,
    )
