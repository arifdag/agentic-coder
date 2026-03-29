"""Playwright sandbox for executing generated UI / E2E tests."""

import os
import re
import tempfile
import shutil
from typing import Optional
from pathlib import Path

from ..config import UITestConfig
from .sandbox import ExecutionResult


class UITestExecutor:
    """Execute Playwright tests in an isolated Docker container."""

    def __init__(self, config: Optional[UITestConfig] = None):
        self.config = config or UITestConfig.from_env()
        self._docker_available: Optional[bool] = None

    def _check_docker(self) -> bool:
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
        result = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "error_type": None,
            "error_message": None,
            "line_number": None,
        }

        match = re.search(r'(\d+) passed', stdout)
        if match:
            result["tests_passed"] = int(match.group(1))

        match = re.search(r'(\d+) failed', stdout)
        if match:
            result["tests_failed"] = int(match.group(1))

        result["tests_run"] = result["tests_passed"] + result["tests_failed"]

        combined = stdout + stderr

        match = re.search(r'SyntaxError: (.+)', combined)
        if match:
            result["error_type"] = "syntax_error"
            result["error_message"] = match.group(1)

        match = re.search(r'(ModuleNotFoundError|ImportError): (.+)', combined)
        if match and not result["error_type"]:
            result["error_type"] = "import_error"
            result["error_message"] = match.group(2)

        match = re.search(r'TimeoutError', combined)
        if match and not result["error_type"]:
            result["error_type"] = "timeout_error"
            timeout_detail = re.search(r'TimeoutError:?\s*(.*)', combined)
            result["error_message"] = (
                timeout_detail.group(1).strip() if timeout_detail else "Playwright timeout"
            )

        match = re.search(r'(locator\.\w+|page\.\w+): (.+)', combined)
        if match and not result["error_type"]:
            result["error_type"] = "locator_error"
            result["error_message"] = match.group(0)

        match = re.search(r'AssertionError: (.+)', combined)
        if match and not result["error_type"]:
            result["error_type"] = "assertion_error"
            result["error_message"] = match.group(1)

        match = re.search(r'line (\d+)', combined)
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

    def _execute_docker(
        self,
        test_code: str,
        workdir: Path,
        target_url: Optional[str] = None,
        html_content: Optional[str] = None,
    ) -> ExecutionResult:
        import docker

        client = docker.from_env()

        try:
            client.images.get(self.config.playwright_image)
        except docker.errors.ImageNotFound:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="docker_error",
                error_message=(
                    f"Docker image '{self.config.playwright_image}' not found. "
                    f"Build it with: docker build -t {self.config.playwright_image} "
                    f"-f docker/Dockerfile.playwright ."
                ),
            )

        env_vars: dict[str, str] = {}
        if target_url:
            env_vars["BASE_URL"] = target_url
        if html_content:
            env_vars.setdefault("BASE_URL", "http://localhost:8080")

        network_disabled = not self.config.network_enabled
        if target_url and target_url.startswith("http"):
            network_disabled = False

        try:
            container = client.containers.run(
                self.config.playwright_image,
                command=[
                    "pytest", "-v", "--tb=short",
                    "--browser", "chromium",
                    "test_generated.py",
                ],
                volumes={
                    str(workdir.absolute()): {
                        "bind": "/workspace",
                        "mode": "ro",
                    }
                },
                working_dir="/workspace",
                environment=env_vars,
                network_disabled=network_disabled,
                mem_limit=self.config.memory_limit,
                cpu_period=100000,
                cpu_quota=100000,
                detach=True,
            )

            result = container.wait(timeout=self.config.timeout)
            exit_code = result.get("StatusCode", -1)
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            container.remove(force=True)

        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="docker_error",
                error_message=str(e),
            )

        parsed = self._parse_pytest_output(stdout, stderr)
        success = (
            exit_code == 0
            and parsed["tests_failed"] == 0
            and parsed["error_type"] is None
        )

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
        )

    def _execute_subprocess(
        self,
        test_code: str,
        workdir: Path,
        target_url: Optional[str] = None,
        html_content: Optional[str] = None,
    ) -> ExecutionResult:
        import subprocess
        import threading

        env = os.environ.copy()
        server_proc = None

        if html_content:
            env["BASE_URL"] = "http://localhost:8080"
            site_dir = workdir / "site"
            server_proc = subprocess.Popen(
                ["python", "-m", "http.server", "8080", "--directory", str(site_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import time
            time.sleep(1)
        elif target_url:
            env["BASE_URL"] = target_url

        try:
            result = subprocess.run(
                [
                    "pytest", "-v", "--tb=short",
                    "--browser", "chromium",
                    "test_generated.py",
                ],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env=env,
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
                error_message="pytest not found. Install pytest and pytest-playwright.",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="execution_error",
                error_message=str(e),
            )
        finally:
            if server_proc:
                server_proc.terminate()
                server_proc.wait(timeout=5)

        parsed = self._parse_pytest_output(stdout, stderr)
        success = (
            exit_code == 0
            and parsed["tests_failed"] == 0
            and parsed["error_type"] is None
        )

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
        )

    def execute(
        self,
        test_code: str,
        target_url: Optional[str] = None,
        html_content: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute generated Playwright tests.

        Provide either ``target_url`` for a live application or ``html_content``
        for a static page (served via a local HTTP server inside the sandbox).
        """
        workdir = Path(tempfile.mkdtemp(prefix="llm_agent_pw_sandbox_"))

        try:
            test_file = workdir / "test_generated.py"
            test_file.write_text(test_code, encoding="utf-8")

            if html_content:
                site_dir = workdir / "site"
                site_dir.mkdir()
                (site_dir / "index.html").write_text(html_content, encoding="utf-8")

            if self._check_docker():
                return self._execute_docker(
                    test_code, workdir,
                    target_url=target_url,
                    html_content=html_content,
                )
            else:
                return self._execute_subprocess(
                    test_code, workdir,
                    target_url=target_url,
                    html_content=html_content,
                )
        finally:
            try:
                shutil.rmtree(workdir)
            except Exception:
                pass
