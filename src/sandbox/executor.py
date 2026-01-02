"""
Sandbox Executor for Secure Code Execution

Provides a Python API to execute untrusted code in isolated Docker containers.
Supports Python, Java, and C# with timeout enforcement and output capture.
"""

import json
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..core.schemas import SupportedLanguage


class ExecutionStatus(str, Enum):
    """Status of code execution."""
    SUCCESS = "success"
    COMPILATION_ERROR = "compilation_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    SYSTEM_ERROR = "system_error"


@dataclass
class ExecutionResult:
    """
    Result of code execution in sandbox.
    
    Attributes:
        execution_id: Unique identifier for this execution
        language: Programming language used
        status: Execution status
        stdout: Standard output from execution
        stderr: Standard error from execution
        execution_time_ms: Time taken for execution in milliseconds
        error_type: Type of error if any
        error_message: Error message if any
        traceback: Full traceback if available
    """
    execution_id: str
    language: SupportedLanguage
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS


class SandboxExecutor:
    """
    Executes code in isolated Docker sandbox containers.
    
    Provides secure, isolated code execution with:
    - Resource limits (CPU, memory)
    - Network isolation
    - Timeout enforcement
    - Output capture
    
    Example:
        executor = SandboxExecutor()
        result = executor.execute("print('Hello, World!')", SupportedLanguage.PYTHON)
        print(result.stdout)  # "Hello, World!\n"
    """
    
    # Docker image names for each language
    IMAGES = {
        SupportedLanguage.PYTHON: "sayzek-sandbox-python:latest",
        SupportedLanguage.JAVA: "sayzek-sandbox-java:latest",
        SupportedLanguage.CSHARP: "sayzek-sandbox-csharp:latest",
    }
    
    # File extensions for each language
    EXTENSIONS = {
        SupportedLanguage.PYTHON: ".py",
        SupportedLanguage.JAVA: ".java",
        SupportedLanguage.CSHARP: ".cs",
    }
    
    # Default file names
    DEFAULT_FILENAMES = {
        SupportedLanguage.PYTHON: "code.py",
        SupportedLanguage.JAVA: "Main.java",
        SupportedLanguage.CSHARP: "Program.cs",
    }
    
    def __init__(
        self,
        docker_compose_path: Optional[Path] = None,
        default_timeout: int = 30,
    ):
        """
        Initialize the sandbox executor.
        
        Args:
            docker_compose_path: Path to docker-compose.yml (optional)
            default_timeout: Default execution timeout in seconds
        """
        self.docker_compose_path = docker_compose_path or Path(__file__).parent.parent.parent / "docker" / "docker-compose.yml"
        self.default_timeout = default_timeout
    
    def execute(
        self,
        code: str,
        language: SupportedLanguage,
        filename: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute code in the appropriate sandbox container.
        
        Args:
            code: Source code to execute
            language: Programming language
            filename: Optional custom filename for the code
            timeout: Execution timeout in seconds (default: 30)
            
        Returns:
            ExecutionResult with output and status
        """
        execution_id = str(uuid.uuid4())
        timeout = timeout or self.default_timeout
        
        # Determine filename
        if filename is None:
            filename = self.DEFAULT_FILENAMES[language]
        
        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write code to file
            code_file = temp_path / filename
            code_file.write_text(code, encoding='utf-8')
            
            # Execute in container
            try:
                result = self._run_in_container(
                    language=language,
                    code_dir=temp_path,
                    timeout=timeout,
                    execution_id=execution_id,
                )
                return result
                
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    execution_id=execution_id,
                    language=language,
                    status=ExecutionStatus.TIMEOUT,
                    error_type="TimeoutError",
                    error_message=f"Execution exceeded {timeout} seconds",
                )
            except Exception as e:
                return ExecutionResult(
                    execution_id=execution_id,
                    language=language,
                    status=ExecutionStatus.SYSTEM_ERROR,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
    
    def _run_in_container(
        self,
        language: SupportedLanguage,
        code_dir: Path,
        timeout: int,
        execution_id: str,
    ) -> ExecutionResult:
        """
        Run code in Docker container.
        
        Args:
            language: Programming language
            code_dir: Directory containing code files
            timeout: Execution timeout in seconds
            execution_id: Unique execution ID
            
        Returns:
            ExecutionResult from container execution
        """
        image = self.IMAGES[language]
        
        # Docker run command with security restrictions
        cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            "--network", "none",  # No network access
            "--memory", "256m",  # Memory limit
            "--cpus", "0.5",  # CPU limit
            "--read-only",  # Read-only filesystem
            "--tmpfs", "/tmp:size=64m",  # Temporary filesystem
            "--tmpfs", "/code/output:size=64m,mode=1777",  # Output directory needs to be writable
            "-v", f"{code_dir}:/code/input:ro",  # Mount code read-only
            image,
        ]
        
        start_time = time.time()
        
        # Run container
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Parse result from stdout (JSON format)
        try:
            result_data = json.loads(process.stdout)
            
            if result_data.get("success", False):
                status = ExecutionStatus.SUCCESS
            elif result_data.get("error_type") == "CompilationError":
                status = ExecutionStatus.COMPILATION_ERROR
            else:
                status = ExecutionStatus.RUNTIME_ERROR
            
            return ExecutionResult(
                execution_id=execution_id,
                language=language,
                status=status,
                stdout=result_data.get("stdout", ""),
                stderr=result_data.get("stderr", ""),
                execution_time_ms=result_data.get("execution_time_ms", execution_time_ms),
                error_type=result_data.get("error_type"),
                error_message=result_data.get("error_message"),
                traceback=result_data.get("traceback"),
            )
            
        except json.JSONDecodeError:
            # If output is not valid JSON, treat as system error
            return ExecutionResult(
                execution_id=execution_id,
                language=language,
                status=ExecutionStatus.SYSTEM_ERROR,
                stdout=process.stdout,
                stderr=process.stderr,
                execution_time_ms=execution_time_ms,
                error_type="OutputParseError",
                error_message="Failed to parse container output",
            )
    
    def build_images(self) -> bool:
        """
        Build all sandbox Docker images.
        
        Returns:
            True if all images built successfully
        """
        try:
            subprocess.run(
                ["docker-compose", "-f", str(self.docker_compose_path), "build"],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def check_docker_available(self) -> bool:
        """
        Check if Docker is available and running.
        
        Returns:
            True if Docker is available
        """
        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
