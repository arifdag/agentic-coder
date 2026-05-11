"""Docker sandbox for secure test execution."""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..config import SandboxConfig

_STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", set()))

_COMMON_TEST_AND_THIRD_PARTY = {
    "pytest",
    "unittest",
    "mock",
    "typing",
    "hypothesis",
    "coverage",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    "websockets",
    "pydantic",
    "attrs",
    "dataclasses",
    "click",
    "typer",
    "rich",
    "flask",
    "django",
    "fastapi",
    "starlette",
    "werkzeug",
    "jinja2",
    "sqlalchemy",
    "alembic",
    "redis",
    "pymongo",
    "psycopg2",
    "pymysql",
    "yaml",
    "toml",
    "msgpack",
    "lxml",
    "bs4",
    "beautifulsoup4",
    "PIL",
    "cv2",
    "scikit-learn",
    "freezegun",
    "responses",
    "faker",
    "factory_boy",
    "vcr",
    "langchain",
    "langchain_core",
    "langchain_groq",
    "langgraph",
}


def _is_known_import(module: str) -> bool:
    """Return True if the module is stdlib or a known third-party package."""
    root = module.split(".")[0]
    return root in _STDLIB_MODULES or root in _COMMON_TEST_AND_THIRD_PARTY


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
    coverage_gaps: Optional[str] = Field(
        default=None, description="Uncovered lines from term-missing"
    )
    branch_coverage: Optional[float] = Field(default=None, description="Branch coverage percentage")
    coverage_data: Optional[dict] = Field(default=None, description="Parsed coverage.json data")
    repo_setup_pass: Optional[bool] = Field(
        default=None, description="Whether repo-context dependency setup succeeded"
    )
    infrastructure_pass: Optional[bool] = Field(
        default=None, description="Whether repo-context infrastructure was usable"
    )


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

        summary_pattern = r"(\d+) passed"
        match = re.search(summary_pattern, stdout)
        if match:
            result["tests_passed"] = int(match.group(1))

        failed_pattern = r"(\d+) failed"
        match = re.search(failed_pattern, stdout)
        if match:
            result["tests_failed"] = int(match.group(1))

        result["tests_run"] = result["tests_passed"] + result["tests_failed"]

        combined = stdout + stderr

        syntax_pattern = r"SyntaxError: (.+)"
        match = re.search(syntax_pattern, combined)
        if match:
            result["error_type"] = "syntax_error"
            result["error_message"] = match.group(1)

        import_pattern = r"(ModuleNotFoundError|ImportError): (.+)"
        match = re.search(import_pattern, combined)
        if match:
            result["error_type"] = "import_error"
            result["error_message"] = match.group(2)

        assertion_pattern = r"AssertionError: (.+)"
        match = re.search(assertion_pattern, combined)
        if match and not result["error_type"]:
            result["error_type"] = "assertion_error"
            result["error_message"] = match.group(1)

        line_pattern = r"line (\d+)"
        match = re.search(line_pattern, combined)
        if match:
            result["line_number"] = int(match.group(1))

        # "collected 0 items" — pytest couldn't find any test functions.
        # This is a distinct failure mode from tests_failed > 0 and needs
        # to be surfaced to the repair LLM so it knows the test file is
        # structurally wrong (missing test_* functions, import errors
        # before collection, etc.).
        if result["tests_run"] == 0 and not result["error_type"]:
            result["error_type"] = "no_tests_collected"
            collected_msg = ""
            # Walk stdout backwards to find actionable info. Skip coverage
            # table rows, summary bars, and the "no tests ran" footer.
            _COV_SKIP = {"Name", "TOTAL", "source_module.py"}
            for line in reversed(stdout.splitlines()):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("="):
                    continue
                if stripped.startswith("-"):
                    continue
                if "no tests ran" in stripped.lower():
                    continue
                if stripped.split()[0] in _COV_SKIP:
                    continue
                # First meaningful line (could be an import error, syntax
                # error, or the "collected 0 items" line itself).
                collected_msg = stripped
                break
            result["error_message"] = collected_msg or "No test functions were collected by pytest"

        if not result["error_type"] and result["tests_failed"] > 0:
            result["error_type"] = "test_failure"
            fail_match = re.search(r"FAILED (.+)", stdout)
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
        pattern = r"TOTAL\s+\d+\s+\d+\s+(\d+)%"
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
        pattern = r"source_module\s+\d+\s+\d+\s+\d+%\s+(.+)"
        match = re.search(pattern, stdout)
        if match:
            return match.group(1).strip()
        return None

    def _parse_coverage_json(self, workdir: Path) -> tuple[Optional[float], Optional[dict]]:
        """Parse branch coverage and raw coverage data from coverage.py JSON output."""
        path = workdir / "coverage.json"
        if not path.exists():
            return None, None
        try:
            data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, None

        branch_coverage = None
        totals = data.get("totals")
        if isinstance(totals, dict):
            raw = totals.get("percent_covered_branches")
            if raw is None:
                raw = totals.get("percent_branches")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                branch_coverage = float(raw)
        return branch_coverage, data

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

        # Use detach=True so we can retrieve stdout and stderr separately
        # after the container exits. The default run(detach=False) path loses
        # stdout when the container exits non-zero (it raises ContainerError
        # with only stderr populated), which swallows pytest's failure output.
        container = None
        try:
            # Enforce the wall-clock budget *inside* the container with
            # pytest-timeout so the container self-terminates. docker-py's
            # ``container.wait(timeout=...)`` is an HTTP request timeout, not
            # a container runtime limit, so we give it generous headroom on
            # top of the pytest budget.
            test_timeout = max(10, int(self.config.timeout))
            http_wait_timeout = test_timeout + 60

            container = client.containers.run(
                self.config.image_name,
                command=[
                    "pytest",
                    "-v",
                    "--tb=short",
                    "-p",
                    "no:cacheprovider",
                    f"--timeout={test_timeout}",
                    "--timeout-method=thread",
                    "--cov=source_module",
                    "--cov-branch",
                    "--cov-report=term-missing",
                    "--cov-report=json:coverage.json",
                    "test_generated.py",
                ],
                volumes={
                    str(workdir.absolute()): {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                },
                working_dir="/workspace",
                network_disabled=self.config.network_disabled,
                mem_limit=self.config.memory_limit,
                cpu_period=100000,
                cpu_quota=int(self.config.cpu_limit * 100000),
                detach=True,
                stdout=True,
                stderr=True,
            )

            try:
                wait_result = container.wait(timeout=http_wait_timeout)
                exit_code = int(wait_result.get("StatusCode", 1))
            except Exception as wait_exc:
                # HTTP timeout or daemon hiccup: kill the container and
                # surface a useful error rather than leaking it.
                try:
                    container.kill()
                except Exception:
                    pass
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    error_type="docker_timeout",
                    error_message=f"container.wait timed out after {http_wait_timeout}s: {wait_exc}",
                )

            stdout_bytes = container.logs(stdout=True, stderr=False) or b""
            stderr_bytes = container.logs(stdout=False, stderr=True) or b""
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="docker_error",
                error_message=str(e),
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        parsed = self._parse_pytest_output(stdout, stderr)
        coverage = self._parse_coverage(stdout)
        coverage_gaps = self._parse_coverage_gaps(stdout)
        branch_coverage, coverage_data = self._parse_coverage_json(workdir)

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
            branch_coverage=branch_coverage,
            coverage_data=coverage_data,
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
                    "pytest",
                    "-v",
                    "--tb=short",
                    "-p",
                    "no:cacheprovider",
                    "--cov=source_module",
                    "--cov-branch",
                    "--cov-report=term-missing",
                    "--cov-report=json:coverage.json",
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
        branch_coverage, coverage_data = self._parse_coverage_json(workdir)

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
            branch_coverage=branch_coverage,
            coverage_data=coverage_data,
        )

    def _fix_imports(self, test_code: str) -> str:
        """Rewrite imports that reference the generated source module.

        Strategy: only rewrite imports whose root module is NOT a known
        stdlib/third-party package. Everything else (unittest.mock, tempfile,
        os, json, numpy, ...) is preserved untouched.

        Implemented with AST surgery (not line regex) so multi-line
        parenthesized imports like::

            from foo import (
                bar,
                baz,
            )

        get rewritten as a single logical unit instead of leaving dangling
        continuation lines that later raise IndentationError.
        """
        import ast

        try:
            tree = ast.parse(test_code)
        except SyntaxError:
            # If the LLM produced unparseable code, fall back to a safe stub
            # with just the wildcard import — the sandbox will surface the
            # original syntax error via pytest anyway.
            return "from source_module import *\n\n" + test_code

        has_source_import = False
        replacements: list = []  # (start_line, end_line, new_text)

        def _loc(node: ast.AST) -> tuple:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            return start, end

        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if root == "source_module":
                    has_source_import = True
                    continue
                if _is_known_import(module):
                    continue
                # Unknown module -> redirect to wildcard from source_module
                has_source_import = True
                start, end = _loc(node)
                if start is not None:
                    replacements.append((start, end, "from source_module import *"))
                continue

            if isinstance(node, ast.Import):
                # `import a, b as c` -> keep known roots, replace unknown ones
                kept = []
                unknown = False
                for alias in node.names:
                    module = alias.name
                    root = module.split(".")[0]
                    if root == "source_module":
                        has_source_import = True
                        kept.append(alias)
                    elif _is_known_import(module):
                        kept.append(alias)
                    else:
                        unknown = True
                if not unknown:
                    continue
                has_source_import = True
                new_lines: list = []
                for alias in kept:
                    seg = f"import {alias.name}"
                    if alias.asname:
                        seg += f" as {alias.asname}"
                    new_lines.append(seg)
                new_lines.append("from source_module import *")
                start, end = _loc(node)
                if start is not None:
                    replacements.append((start, end, "\n".join(new_lines)))

        if not replacements:
            if not has_source_import:
                return "from source_module import *\n\n" + test_code
            return test_code

        # Apply replacements bottom-up so line numbers stay valid.
        source_lines = test_code.splitlines()
        for start, end, new_text in sorted(replacements, key=lambda r: -r[0]):
            source_lines[start - 1 : end] = [new_text]

        result = "\n".join(source_lines)
        if not has_source_import and "from source_module import *" not in result:
            result = "from source_module import *\n\n" + result
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
