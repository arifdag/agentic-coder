"""Node.js sandbox for secure Jest test execution."""

import json
import os
import re
import tempfile
import shutil
from typing import Optional
from pathlib import Path

from .sandbox import ExecutionResult


class JsSandboxConfig:
    """Minimal config holder resolved from the real config at import time."""
    pass


class JsSandboxExecutor:
    """Execute Jest tests in an isolated Docker container or local subprocess."""

    def __init__(self, config=None):
        self.config = config
        self._docker_available = None

    @property
    def image_name(self) -> str:
        if self.config and hasattr(self.config, "image_name"):
            return self.config.image_name
        return "llm-agent-node"

    @property
    def timeout(self) -> int:
        if self.config and hasattr(self.config, "timeout"):
            return self.config.timeout
        return 60

    @property
    def memory_limit(self) -> str:
        if self.config and hasattr(self.config, "memory_limit"):
            return self.config.memory_limit
        return "512m"

    @property
    def network_disabled(self) -> bool:
        if self.config and hasattr(self.config, "network_disabled"):
            return self.config.network_disabled
        return True

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

    def _parse_jest_output(self, stdout: str, stderr: str) -> dict:
        """Parse Jest output to extract test results."""
        result = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "error_type": None,
            "error_message": None,
            "line_number": None,
        }

        combined = stdout + "\n" + stderr

        # Jest summary line: Tests: 2 passed, 1 failed, 3 total
        m = re.search(r'Tests:\s+(?:(\d+)\s+failed,\s*)?(?:(\d+)\s+passed,\s*)?(\d+)\s+total', combined)
        if m:
            result["tests_failed"] = int(m.group(1) or 0)
            result["tests_passed"] = int(m.group(2) or 0)
            result["tests_run"] = int(m.group(3) or 0)

        # Fallback: count PASS/FAIL lines
        if result["tests_run"] == 0:
            pass_count = len(re.findall(r'✓|✔|PASS', combined))
            fail_count = len(re.findall(r'✕|✘|FAIL', combined))
            result["tests_passed"] = pass_count
            result["tests_failed"] = fail_count
            result["tests_run"] = pass_count + fail_count

        # Syntax errors
        m = re.search(r'SyntaxError:\s*(.+)', combined)
        if m:
            result["error_type"] = "syntax_error"
            result["error_message"] = m.group(1).strip()

        # Module not found
        m = re.search(r"Cannot find module\s+'([^']+)'", combined)
        if m and not result["error_type"]:
            result["error_type"] = "import_error"
            result["error_message"] = f"Cannot find module '{m.group(1)}'"

        # Reference / Type errors
        m = re.search(r'(ReferenceError|TypeError):\s*(.+)', combined)
        if m and not result["error_type"]:
            result["error_type"] = m.group(1).lower()
            result["error_message"] = m.group(2).strip()

        # Line number
        m = re.search(r':(\d+):\d+\)', combined)
        if m:
            result["line_number"] = int(m.group(1))

        if not result["error_type"] and result["tests_failed"] > 0:
            result["error_type"] = "test_failure"
            result["error_message"] = f"{result['tests_failed']} test(s) failed"

        return result

    def _fix_imports(self, test_code: str) -> str:
        """Rewrite import paths to point to ./source_module."""
        # require('./anything') -> require('./source_module')
        test_code = re.sub(
            r"""require\s*\(\s*['"]\.\/[^'"]+['"]\s*\)""",
            "require('./source_module')",
            test_code,
        )
        # import ... from './anything' -> import ... from './source_module'
        test_code = re.sub(
            r"""(from\s+)['"]\.\/[^'"]+['"]""",
            r"\1'./source_module'",
            test_code,
        )
        return test_code

    def _write_package_json(self, workdir: Path):
        """Write a minimal package.json so Jest can resolve."""
        pkg = {
            "name": "sandbox-test",
            "version": "1.0.0",
            "scripts": {"test": "jest --verbose --no-cache"},
        }
        (workdir / "package.json").write_text(json.dumps(pkg), encoding="utf-8")

    def _execute_docker(
        self,
        source_code: str,
        test_code: str,
        workdir: Path,
    ) -> ExecutionResult:
        import docker

        client = docker.from_env()

        try:
            client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="docker_error",
                error_message=(
                    f"Docker image '{self.image_name}' not found. "
                    f"Build with: docker build -t {self.image_name} -f docker/Dockerfile.node ."
                ),
            )

        try:
            container = client.containers.run(
                self.image_name,
                command=["npx", "jest", "--verbose", "--no-cache"],
                volumes={
                    str(workdir.absolute()): {"bind": "/workspace", "mode": "rw"}
                },
                working_dir="/workspace",
                network_disabled=self.network_disabled,
                mem_limit=self.memory_limit,
                remove=False,
                detach=True,
            )

            exit_info = container.wait(timeout=self.timeout)
            exit_code = exit_info.get("StatusCode", -1)

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

        parsed = self._parse_jest_output(stdout, stderr)
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
        )

    def _execute_subprocess(
        self,
        source_code: str,
        test_code: str,
        workdir: Path,
    ) -> ExecutionResult:
        import subprocess

        try:
            result = subprocess.run(
                ["npx", "jest", "--verbose", "--no-cache"],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=(os.name == "nt"),
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="timeout",
                error_message=f"Execution timed out after {self.timeout} seconds",
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="environment_error",
                error_message="npx/jest not found. Install Node.js and Jest.",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="execution_error",
                error_message=str(e),
            )

        parsed = self._parse_jest_output(stdout, stderr)
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
        )

    def execute(
        self,
        source_code: str,
        test_code: str,
    ) -> ExecutionResult:
        """Execute generated Jest tests against source code."""
        workdir = Path(tempfile.mkdtemp(prefix="llm_agent_js_sandbox_"))

        try:
            source_file = workdir / "source_module.js"
            source_file.write_text(source_code, encoding="utf-8")

            test_code = self._fix_imports(test_code)

            test_file = workdir / "source_module.test.js"
            test_file.write_text(test_code, encoding="utf-8")

            self._write_package_json(workdir)

            if self._check_docker():
                return self._execute_docker(source_code, test_code, workdir)
            else:
                return self._execute_subprocess(source_code, test_code, workdir)
        finally:
            try:
                shutil.rmtree(workdir)
            except Exception:
                pass
