"""Docker sandbox for secure test execution."""

import os
import re
import tempfile
import shutil
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field

from ..config import SandboxConfig


class ExecutionResult(BaseModel):
    """Result from sandbox execution."""
    
    success: bool = Field(description="Whether all tests passed")
    exit_code: int = Field(description="Container exit code")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    error_type: Optional[str] = Field(default=None, description="Type of error if failed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    line_number: Optional[int] = Field(default=None, description="Line number of error")
    tests_run: int = Field(default=0, description="Number of tests executed")
    tests_passed: int = Field(default=0, description="Number of tests passed")
    tests_failed: int = Field(default=0, description="Number of tests failed")
    coverage: Optional[float] = Field(default=None, description="Code coverage percentage")
    coverage_gaps: Optional[str] = Field(default=None, description="Uncovered lines from term-missing")


class SandboxExecutor:
    """Execute tests in an isolated Docker container."""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """Initialize the sandbox executor.
        
        Args:
            config: Sandbox configuration
        """
        self.config = config or SandboxConfig.from_env()
        self._docker_available = None
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available
        
        try:
            import docker
            client = docker.from_env()
            client.ping()
            self._docker_available = True
        except Exception:
            self._docker_available = False
        
        return self._docker_available
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> dict:
        """Parse pytest output to extract test results.
        
        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest
            
        Returns:
            Dictionary with parsed results
        """
        result = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "error_type": None,
            "error_message": None,
            "line_number": None,
        }
        
        summary_pattern = r'(\d+) passed'
        match = re.search(summary_pattern, stdout)
        if match:
            result["tests_passed"] = int(match.group(1))
        
        failed_pattern = r'(\d+) failed'
        match = re.search(failed_pattern, stdout)
        if match:
            result["tests_failed"] = int(match.group(1))
        
        result["tests_run"] = result["tests_passed"] + result["tests_failed"]
        
        combined = stdout + stderr
        
        syntax_pattern = r'SyntaxError: (.+)'
        match = re.search(syntax_pattern, combined)
        if match:
            result["error_type"] = "syntax_error"
            result["error_message"] = match.group(1)
        
        import_pattern = r'(ModuleNotFoundError|ImportError): (.+)'
        match = re.search(import_pattern, combined)
        if match:
            result["error_type"] = "import_error"
            result["error_message"] = match.group(2)
        
        assertion_pattern = r'AssertionError: (.+)'
        match = re.search(assertion_pattern, combined)
        if match and not result["error_type"]:
            result["error_type"] = "assertion_error"
            result["error_message"] = match.group(1)
        
        line_pattern = r'line (\d+)'
        match = re.search(line_pattern, combined)
        if match:
            result["line_number"] = int(match.group(1))
        
        if not result["error_type"] and result["tests_failed"] > 0:
            result["error_type"] = "test_failure"
            fail_match = re.search(r'FAILED (.+)', stdout)
            if fail_match:
                result["error_message"] = f"Test failed: {fail_match.group(1)}"
            else:
                result["error_message"] = f"{result['tests_failed']} test(s) failed"
        
        return result
    
    def _parse_coverage(self, stdout: str) -> Optional[float]:
        """Parse coverage percentage from pytest-cov output.
        
        Args:
            stdout: Standard output
            
        Returns:
            Coverage percentage or None
        """
        pattern = r'TOTAL\s+\d+\s+\d+\s+(\d+)%'
        match = re.search(pattern, stdout)
        if match:
            return float(match.group(1))
        return None

    def _parse_coverage_gaps(self, stdout: str) -> Optional[str]:
        """Parse missing line numbers from pytest-cov term-missing output.

        Looks for the 'Missing' column in coverage output, e.g.:
            source_module   30      5    83%   12, 15-18, 23

        Returns:
            Comma-separated missing lines string, or None
        """
        pattern = r'source_module\s+\d+\s+\d+\s+\d+%\s+(.+)'
        match = re.search(pattern, stdout)
        if match:
            return match.group(1).strip()
        return None
    
    def _execute_docker(
        self,
        source_code: str,
        test_code: str,
        workdir: Path,
    ) -> ExecutionResult:
        """Execute tests using Docker.
        
        Args:
            source_code: Source code to test
            test_code: Generated test code
            workdir: Working directory with files
            
        Returns:
            Execution result
        """
        import docker
        
        client = docker.from_env()
        
        try:
            client.images.get(self.config.image_name)
        except docker.errors.ImageNotFound:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="docker_error",
                error_message=f"Docker image '{self.config.image_name}' not found. "
                              f"Please build it with: docker build -t {self.config.image_name} -f docker/Dockerfile .",
            )
        
        try:
            container = client.containers.run(
                self.config.image_name,
                command=[
                    "pytest", "-v", "--tb=short",
                    "--cov=source_module", "--cov-report=term-missing",
                    "test_generated.py",
                ],
                volumes={
                    str(workdir.absolute()): {
                        "bind": "/workspace",
                        "mode": "ro",
                    }
                },
                working_dir="/workspace",
                network_disabled=self.config.network_disabled,
                mem_limit=self.config.memory_limit,
                cpu_period=100000,
                cpu_quota=int(self.config.cpu_limit * 100000),
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
            )
            
            stdout = container.decode("utf-8") if isinstance(container, bytes) else str(container)
            stderr = ""
            exit_code = 0
            
        except docker.errors.ContainerError as e:
            stdout = e.stderr.decode("utf-8") if e.stderr else ""
            stderr = str(e)
            exit_code = e.exit_status
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="docker_error",
                error_message=str(e),
            )
        
        parsed = self._parse_pytest_output(stdout, stderr)
        coverage = self._parse_coverage(stdout)
        coverage_gaps = self._parse_coverage_gaps(stdout)
        
        success = exit_code == 0 and parsed["tests_failed"] == 0 and parsed["error_type"] is None
        
        return ExecutionResult(
            success=success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            error_type=parsed["error_type"],
            error_message=parsed["error_message"],
            line_number=parsed["line_number"],
            tests_run=parsed["tests_run"],
            tests_passed=parsed["tests_passed"],
            tests_failed=parsed["tests_failed"],
            coverage=coverage,
            coverage_gaps=coverage_gaps,
        )
    
    def _execute_subprocess(
        self,
        source_code: str,
        test_code: str,
        workdir: Path,
    ) -> ExecutionResult:
        """Execute tests using subprocess (fallback when Docker unavailable).
        
        Args:
            source_code: Source code to test
            test_code: Generated test code
            workdir: Working directory with files
            
        Returns:
            Execution result
        """
        import subprocess
        
        try:
            result = subprocess.run(
                [
                    "pytest", "-v", "--tb=short",
                    "--cov=source_module", "--cov-report=term-missing",
                    "test_generated.py",
                ],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
            
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="timeout",
                error_message=f"Execution timed out after {self.config.timeout} seconds",
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="environment_error",
                error_message="pytest not found. Please install pytest.",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="execution_error",
                error_message=str(e),
            )
        
        parsed = self._parse_pytest_output(stdout, stderr)
        coverage = self._parse_coverage(stdout)
        coverage_gaps = self._parse_coverage_gaps(stdout)
        
        success = exit_code == 0 and parsed["tests_failed"] == 0 and parsed["error_type"] is None
        
        return ExecutionResult(
            success=success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            error_type=parsed["error_type"],
            error_message=parsed["error_message"],
            line_number=parsed["line_number"],
            tests_run=parsed["tests_run"],
            tests_passed=parsed["tests_passed"],
            tests_failed=parsed["tests_failed"],
            coverage=coverage,
            coverage_gaps=coverage_gaps,
        )
    
    def _fix_imports(self, test_code: str) -> str:
        """Fix import statements to use source_module instead of original paths.
        
        Args:
            test_code: Generated test code
            
        Returns:
            Test code with fixed imports
        """
        lines = test_code.split('\n')
        fixed_lines = []
        has_source_import = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('from ') and ' import ' in stripped:
                if 'source_module' in stripped:
                    has_source_import = True
                    fixed_lines.append(line)
                elif stripped.startswith('from pytest') or stripped.startswith('from typing'):
                    fixed_lines.append(line)
                else:
                    has_source_import = True
                    fixed_lines.append('from source_module import *')
            elif stripped.startswith('import ') and not stripped.startswith('import pytest'):
                if 'source_module' in stripped:
                    has_source_import = True
                    fixed_lines.append(line)
                else:
                    has_source_import = True
                    fixed_lines.append('from source_module import *')
            else:
                fixed_lines.append(line)
        
        result = '\n'.join(fixed_lines)
        
        if not has_source_import:
            result = 'from source_module import *\n\n' + result
        
        return result
    
    def execute(
        self,
        source_code: str,
        test_code: str,
    ) -> ExecutionResult:
        """Execute generated tests against source code.
        
        Args:
            source_code: The source code to test
            test_code: The generated test code
            
        Returns:
            Execution result with pass/fail status and diagnostics
        """
        workdir = Path(tempfile.mkdtemp(prefix="llm_agent_sandbox_"))
        
        try:
            source_file = workdir / "source_module.py"
            source_file.write_text(source_code, encoding="utf-8")
            
            test_file = workdir / "test_generated.py"
            
            test_code = self._fix_imports(test_code)
            
            test_file.write_text(test_code, encoding="utf-8")
            
            if self._check_docker():
                return self._execute_docker(source_code, test_code, workdir)
            else:
                return self._execute_subprocess(source_code, test_code, workdir)
        
        finally:
            try:
                shutil.rmtree(workdir)
            except Exception:
                pass
