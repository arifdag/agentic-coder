"""Repo-context executor for running generated tests inside real project checkouts.

Executes Python pytest tests within a cloned/copied project worktree rather
than the single-file sandbox, preserving import structures and enabling
coverage against the real module tree.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .sandbox import ExecutionResult, SandboxExecutor

log = logging.getLogger(__name__)

_DEFAULT_REPOS_CACHE = Path(".cache") / "llm_agent" / "repos"
_GENERATED_DIR = ".llm_agent_generated"
_TEST_FILE = "test_generated.py"


def _tail(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


class RepoContextExecutor:
    """Execute generated tests against a real project checkout or worktree."""

    def __init__(
        self,
        repo_setup: str = "auto",
        gate_policy: str = "strict",
        pytest_timeout: Optional[int] = None,
    ):
        self.repo_setup = repo_setup
        self.gate_policy = gate_policy
        self._pytest_timeout = pytest_timeout
        self._last_repo_error: Optional[str] = None
        self._parser = SandboxExecutor()
        self._repos_cache = Path(
            os.getenv("LLM_AGENT_REPO_CACHE", str(_DEFAULT_REPOS_CACHE))
        ).resolve()

    def execute(
        self,
        source_code: str,
        test_code: str,
        metadata: Dict[str, Any],
    ) -> ExecutionResult:
        """Run generated tests inside a repo checkout.

        Args:
            source_code: The target source code (used as fallback metadata).
            test_code: Generated pytest test code.
            metadata: Dict with repo execution fields.
                Remote: ``repo``, ``base_commit``, ``code_file``, ``test_file``
                Local: ``project_root``, ``project_name``, ``target_file``,
                ``dependency_files``

        Returns:
            ``ExecutionResult`` compatible with sandbox metrics.
        """
        temp_repo: Optional[Path] = None
        try:
            project_root = metadata.get("project_root")
            repo_url = metadata.get("repo")
            base_commit = metadata.get("base_commit")

            if project_root:
                src_path = Path(project_root)
                if not src_path.exists():
                    return ExecutionResult(
                        success=False,
                        exit_code=-1,
                        error_type="infrastructure_error",
                        error_message=f"Local project root not found: {project_root}",
                        infrastructure_pass=False,
                    )
                temp_repo = self._copy_local(src_path)
            elif repo_url:
                temp_repo = self._clone_remote(repo_url, base_commit)
                if temp_repo is None:
                    message = self._last_repo_error or (
                        f"Failed to clone/checkout repo: {repo_url} @ {base_commit}"
                    )
                    return ExecutionResult(
                        success=False,
                        exit_code=-1,
                        error_type="infrastructure_error",
                        error_message=message,
                        infrastructure_pass=False,
                    )
            else:
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    error_type="infrastructure_error",
                    error_message="No project_root or repo in metadata for repo-context execution",
                    infrastructure_pass=False,
                )

            setup_ok, setup_msg = self._setup_dependencies(temp_repo)
            # prepare=strict, auto=best-effort, off/reuse=skip
            if self.repo_setup == "prepare":
                if not setup_ok:
                    return ExecutionResult(
                        success=False,
                        exit_code=-1,
                        error_type="infrastructure_error",
                        error_message=setup_msg,
                        repo_setup_pass=False,
                        infrastructure_pass=False,
                    )
                repo_setup_pass = True
            elif self.repo_setup in ("off", "reuse"):
                repo_setup_pass = None  # skipped, not a failure
            else:  # auto
                repo_setup_pass = setup_ok if setup_ok else False

            gen_dir = temp_repo / _GENERATED_DIR
            gen_dir.mkdir(parents=True, exist_ok=True)
            test_path = gen_dir / _TEST_FILE
            test_path.write_text(test_code, encoding="utf-8")

            result = self._run_pytest(temp_repo, test_path, metadata)
            result.repo_setup_pass = repo_setup_pass
            if repo_setup_pass is False and setup_msg:
                setup_details = f"[repo setup warning]\n{setup_msg}\n"
                result.stderr = f"{setup_details}\n{result.stderr or ''}"
            return result

        finally:
            if temp_repo is not None:
                try:
                    shutil.rmtree(temp_repo, ignore_errors=True)
                except Exception:
                    pass

    def _copy_local(self, src: Path) -> Path:
        dest = Path(tempfile.mkdtemp(prefix="llm_agent_repo_"))
        ignore = shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".pytest_cache", "node_modules"
        )
        shutil.copytree(src, dest / "project", ignore=ignore)
        return dest / "project"

    def _clone_remote(self, repo_url: str, base_commit: Optional[str]) -> Optional[Path]:
        self._last_repo_error = None
        if not shutil.which("git"):
            self._last_repo_error = "git not available for repo-context execution"
            log.warning(self._last_repo_error)
            return None

        self._repos_cache.mkdir(parents=True, exist_ok=True)
        clone_url = self._normalize_repo_url(repo_url)
        repo_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", clone_url.rstrip("/"))
        repo_name = repo_name.removesuffix(".git") or "repo"
        cache_path = self._repos_cache / repo_name

        def _git_safe(args: List[str], timeout: int) -> subprocess.CompletedProcess:
            safe_args = ["git", "-c", f"safe.directory={str(cache_path)}"] + args[1:]
            return subprocess.run(
                safe_args,
                capture_output=True,
                timeout=timeout,
                check=False,
            )

        def _diagnostic(step: str, proc: subprocess.CompletedProcess) -> str:
            stderr = proc.stderr.decode(errors="replace") if proc.stderr else ""
            command = " ".join(str(part) for part in getattr(proc, "args", []) or [])
            return (
                f"Git {step} failed for {clone_url} at cache {cache_path}"
                f"{f' (commit {base_commit})' if base_commit else ''}. "
                f"Command: {command or step}. "
                f"Return code: {proc.returncode}. "
                f"Stderr: {_tail(stderr)}"
            )

        def _remove_cache() -> None:
            try:
                shutil.rmtree(cache_path, ignore_errors=True)
            except Exception:
                pass

        def _try_clone_or_fetch(retrying: bool = False) -> Optional[Path]:
            try:
                if cache_path.exists():
                    # Verify it looks like a git repo
                    git_dir = cache_path / ".git"
                    if not git_dir.exists() or not git_dir.is_dir():
                        message = f"Cache {cache_path} for {clone_url} is not a git repo"
                        if not retrying:
                            log.warning("%s; removing and retrying", message)
                            _remove_cache()
                            return _try_clone_or_fetch(retrying=True)
                        self._last_repo_error = message
                        log.warning("%s after retry", message)
                        return None

                    proc = _git_safe(
                        ["git", "-C", str(cache_path), "fetch", "origin"],
                        timeout=60,
                    )
                    if proc.returncode != 0:
                        message = _diagnostic("fetch", proc)
                        if not retrying:
                            log.warning(
                                "Git fetch failed for %s; removing cache and retrying", cache_path
                            )
                            _remove_cache()
                            return _try_clone_or_fetch(retrying=True)
                        self._last_repo_error = message
                        log.warning(message)
                        return None
                else:
                    proc = _git_safe(
                        ["git", "clone", clone_url, str(cache_path)],
                        timeout=120,
                    )
                    if proc.returncode != 0:
                        message = _diagnostic("clone", proc)
                        if not retrying:
                            log.warning("Git clone failed for %s; retrying once", cache_path)
                            _remove_cache()
                            return _try_clone_or_fetch(retrying=True)
                        self._last_repo_error = message
                        log.warning(message)
                        return None

                if base_commit:
                    proc = _git_safe(
                        ["git", "-C", str(cache_path), "checkout", base_commit],
                        timeout=60,
                    )
                    if proc.returncode != 0:
                        message = _diagnostic("checkout", proc)
                        if not retrying:
                            log.warning(
                                "Git checkout failed for %s @ %s; retrying once",
                                cache_path,
                                base_commit,
                            )
                            _remove_cache()
                            return _try_clone_or_fetch(retrying=True)
                        self._last_repo_error = message
                        log.warning(message)
                        return None
            except subprocess.TimeoutExpired as exc:
                message = (
                    f"Git operation timed out for {clone_url} at cache {cache_path}"
                    f"{f' (commit {base_commit})' if base_commit else ''}"
                )
                if not retrying:
                    cmd = getattr(exc, "cmd", ["?"])
                    op = cmd[1] if len(cmd) > 1 else "operation"
                    log.warning("Git %s timed out for %s; retrying once", op, cache_path)
                    _remove_cache()
                    return _try_clone_or_fetch(retrying=True)
                self._last_repo_error = message
                log.warning("%s after retry", message)
                return None
            return cache_path

        cache = _try_clone_or_fetch()
        if cache is None:
            return None

        dest = Path(tempfile.mkdtemp(prefix="llm_agent_repo_"))
        ignore = shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".pytest_cache", "node_modules"
        )
        shutil.copytree(cache_path, dest / "project", ignore=ignore)
        return dest / "project"

    @staticmethod
    def _normalize_repo_url(repo_url: str) -> str:
        value = repo_url.strip()
        if value.startswith(("http://", "https://", "git@")):
            return value
        if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value):
            return f"https://github.com/{value}.git"
        return value

    def _setup_dependencies(self, repo_root: Path) -> tuple[bool, str]:
        if self.repo_setup not in ("auto", "prepare"):
            return True, ""

        errors: List[str] = []
        cwd = str(repo_root)

        setup_py = repo_root / "setup.py"
        pyproject = repo_root / "pyproject.toml"
        if setup_py.exists() or pyproject.exists():
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", "."],
                    cwd=cwd,
                    capture_output=True,
                    timeout=120,
                )
                if proc.returncode != 0:
                    msg = proc.stderr.decode(errors="replace")[:500]
                    errors.append(f"pip install -e . failed: {msg}")
            except subprocess.TimeoutExpired:
                errors.append("pip install -e . timed out")

        req_files = sorted(repo_root.glob("requirements*.txt"))

        for req in req_files:
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req)],
                    cwd=cwd,
                    capture_output=True,
                    timeout=120,
                )
                if proc.returncode != 0:
                    msg = proc.stderr.decode(errors="replace")[:500]
                    errors.append(f"pip install -r {req.name} failed: {msg}")
            except subprocess.TimeoutExpired:
                errors.append(f"pip install -r {req.name} timed out")

        if errors:
            return False, "; ".join(errors)
        return True, ""

    def _run_pytest(
        self,
        repo_root: Path,
        test_path: Path,
        metadata: Dict[str, Any],
    ) -> ExecutionResult:
        target_file = metadata.get("target_file") or metadata.get("code_file")
        cov_target: Optional[str] = None
        if target_file:
            target_path = repo_root / target_file
            if target_path.exists():
                rel = target_path.relative_to(repo_root).as_posix()
                if rel.endswith(".py"):
                    cov_target = rel[:-3].replace("/", ".")
                else:
                    cov_target = rel.replace("/", ".")
            else:
                cov_target = "."
        else:
            cov_target = "."

        cmd: List[str] = [
            sys.executable,
            "-m",
            "pytest",
            str(test_path.relative_to(repo_root)),
            "--tb=short",
            "-q",
        ]

        coverage_json_path = repo_root / "coverage.json"
        try:
            import pytest_cov  # noqa: F401

            has_coverage = True
        except ImportError:
            has_coverage = False

        if has_coverage:
            cmd.extend(
                [
                    f"--cov={cov_target}",
                    "--cov-branch",
                    "--cov-report=term-missing",
                    f"--cov-report=json:{coverage_json_path}",
                ]
            )

        # Resolve timeout: per-case metadata overrides constructor value
        effective_timeout: int = 120
        meta_timeout = metadata.get("repo_pytest_timeout") or metadata.get("pytest_timeout")
        if meta_timeout is not None:
            try:
                effective_timeout = int(meta_timeout)
                if effective_timeout <= 0:
                    effective_timeout = 120
            except (ValueError, TypeError):
                effective_timeout = self._pytest_timeout or 120
        else:
            effective_timeout = self._pytest_timeout or 120

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                error_type="infrastructure_error",
                error_message=f"pytest timed out after {effective_timeout}s",
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                infrastructure_pass=False,
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        parsed = self._parser._parse_pytest_output(stdout, stderr)
        tests_run = parsed.get("tests_run", 0)
        tests_passed = parsed.get("tests_passed", 0)
        tests_failed = parsed.get("tests_failed", 0)
        error_type = parsed.get("error_type")
        error_message = parsed.get("error_message")
        line_number = parsed.get("line_number")

        coverage_val: Optional[float] = None
        branch_cov: Optional[float] = None
        coverage_data: Optional[dict] = None

        if has_coverage and coverage_json_path.exists():
            try:
                with open(coverage_json_path, "r", encoding="utf-8") as f:
                    coverage_data = json.load(f)
                if coverage_data and "totals" in coverage_data:
                    totals = coverage_data["totals"]
                    cov_pct = totals.get("percent_covered")
                    if isinstance(cov_pct, (int, float)):
                        coverage_val = float(cov_pct)
                    branch_pct = totals.get("percent_covered_branches")
                    if isinstance(branch_pct, (int, float)):
                        branch_cov = float(branch_pct)
            except Exception:
                pass

        if coverage_val is None:
            coverage_val = self._parser._parse_coverage(stdout)

        success = proc.returncode == 0 and tests_failed == 0 and not error_type
        if error_type and error_type not in ("test_failure", "no_tests_collected"):
            success = False

        return ExecutionResult(
            success=success,
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            error_type=error_type,
            error_message=error_message,
            line_number=line_number,
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            coverage=coverage_val,
            coverage_gaps=self._parser._parse_coverage_gaps(stdout),
            branch_coverage=branch_cov,
            coverage_data=coverage_data,
            repo_setup_pass=None,
            infrastructure_pass=True,
        )
