"""
Tests for Sandbox Executor

Tests the Docker sandbox execution functionality.
Note: These tests require Docker to be installed and running.
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from src.sandbox.executor import (
    SandboxExecutor,
    ExecutionResult,
    ExecutionStatus,
)
from src.core.schemas import SupportedLanguage


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""
    
    def test_successful_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            execution_id="exec-1",
            language=SupportedLanguage.PYTHON,
            status=ExecutionStatus.SUCCESS,
            stdout="Hello, World!\n",
            execution_time_ms=50.0,
        )
        
        assert result.success is True
        assert result.stdout == "Hello, World!\n"
    
    def test_failed_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            execution_id="exec-2",
            language=SupportedLanguage.PYTHON,
            status=ExecutionStatus.RUNTIME_ERROR,
            error_type="ZeroDivisionError",
            error_message="division by zero",
        )
        
        assert result.success is False
        assert result.error_type == "ZeroDivisionError"
    
    def test_timeout_result(self):
        """Test timeout execution result."""
        result = ExecutionResult(
            execution_id="exec-3",
            language=SupportedLanguage.JAVA,
            status=ExecutionStatus.TIMEOUT,
            error_message="Execution exceeded 30 seconds",
        )
        
        assert result.success is False
        assert result.status == ExecutionStatus.TIMEOUT


class TestSandboxExecutor:
    """Tests for SandboxExecutor class."""
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        executor = SandboxExecutor(default_timeout=60)
        
        assert executor.default_timeout == 60
        assert executor.docker_compose_path is not None
    
    def test_image_mapping(self):
        """Test language to image mapping."""
        executor = SandboxExecutor()
        
        assert SupportedLanguage.PYTHON in executor.IMAGES
        assert SupportedLanguage.JAVA in executor.IMAGES
        assert SupportedLanguage.CSHARP in executor.IMAGES
    
    def test_extension_mapping(self):
        """Test language to file extension mapping."""
        executor = SandboxExecutor()
        
        assert executor.EXTENSIONS[SupportedLanguage.PYTHON] == ".py"
        assert executor.EXTENSIONS[SupportedLanguage.JAVA] == ".java"
        assert executor.EXTENSIONS[SupportedLanguage.CSHARP] == ".cs"
    
    @patch('subprocess.run')
    def test_execute_python_success(self, mock_run):
        """Test successful Python execution (mocked)."""
        mock_run.return_value = MagicMock(
            stdout=json.dumps({
                "success": True,
                "stdout": "42\n",
                "stderr": "",
                "execution_time_ms": 25.0,
            }),
            stderr="",
            returncode=0,
        )
        
        executor = SandboxExecutor()
        result = executor.execute("print(42)", SupportedLanguage.PYTHON)
        
        assert result.success is True
        assert result.stdout == "42\n"
    
    @patch('subprocess.run')
    def test_execute_python_error(self, mock_run):
        """Test Python execution with error (mocked)."""
        mock_run.return_value = MagicMock(
            stdout=json.dumps({
                "success": False,
                "stdout": "",
                "stderr": "",
                "error_type": "NameError",
                "error_message": "name 'undefined' is not defined",
                "execution_time_ms": 10.0,
            }),
            stderr="",
            returncode=0,
        )
        
        executor = SandboxExecutor()
        result = executor.execute("print(undefined)", SupportedLanguage.PYTHON)
        
        assert result.success is False
        assert result.error_type == "NameError"
    
    @patch('subprocess.run')
    def test_check_docker_available_true(self, mock_run):
        """Test Docker availability check when available."""
        mock_run.return_value = MagicMock(returncode=0)
        
        executor = SandboxExecutor()
        assert executor.check_docker_available() is True
    
    @patch('subprocess.run')
    def test_check_docker_available_false(self, mock_run):
        """Test Docker availability check when not available."""
        mock_run.side_effect = FileNotFoundError()
        
        executor = SandboxExecutor()
        assert executor.check_docker_available() is False


class TestIntegration:
    """Integration tests for sandbox (require Docker)."""
    
    @pytest.mark.skipif(
        not SandboxExecutor().check_docker_available(),
        reason="Docker not available"
    )
    def test_real_python_execution(self):
        """Test real Python execution in Docker."""
        executor = SandboxExecutor()
        
        # First ensure image is built
        # executor.build_images()
        
        result = executor.execute(
            "print('Integration test success')",
            SupportedLanguage.PYTHON,
            timeout=30,
        )
        
        # This will only pass if Docker is set up correctly
        # Check actual success status now that we know Docker is working
        assert isinstance(result, ExecutionResult)
        assert result.success is True, f"Execution failed with error: {result.error_message}\nStderr: {result.stderr}"
        assert result.language == SupportedLanguage.PYTHON
        assert "success" in result.stdout
