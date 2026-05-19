"""Gate 1: Static Application Security Testing (SAST) with Semgrep and Bandit."""

import ast
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from .models import Finding, GateResult, Severity

# CWE/Bandit codes that are noise on generated *test* code and should never
# block the gate (e.g. Bandit B101 flags every `assert` statement, which is
# the whole point of a test file; CWE-703 is its CWE mapping).
_IGNORED_CODES = {"CWE-703", "B101"}

_SENSITIVE_NAME_RE = re.compile(r"(api[_-]?key|secret|token|password|passwd|credential)", re.I)
_SECRET_VALUE_RE = re.compile(r"(sk-live-|AKIA|secret|token|key|[A-Za-z0-9_/\-+=]{16,})", re.I)


class SastAnalyzer:
    """Run SAST tools (Semgrep + Bandit) on generated code."""

    # Severities considered blocking (gate fails if any such finding survives
    # the LLM judge's false-positive filter).
    BLOCKING_SEVERITIES = {Severity.ERROR, Severity.WARNING}

    def __init__(
        self,
        semgrep_rules: str = "auto",
        bandit_enabled: bool = True,
        timeout: int = 60,
    ):
        self.semgrep_rules = semgrep_rules
        self.bandit_enabled = bandit_enabled
        self.timeout = timeout

    def _run_semgrep(self, file_path: Path) -> List[Finding]:
        """Run Semgrep on a file and return findings."""
        findings: List[Finding] = []

        try:
            result = subprocess.run(
                [
                    "semgrep",
                    "scan",
                    "--config",
                    self.semgrep_rules,
                    "--json",
                    "--quiet",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.stdout.strip():
                data = json.loads(result.stdout)
                for item in data.get("results", []):
                    severity_raw = item.get("extra", {}).get("severity", "WARNING").upper()
                    severity_map = {
                        "ERROR": Severity.ERROR,
                        "WARNING": Severity.WARNING,
                        "INFO": Severity.INFO,
                    }
                    severity = severity_map.get(severity_raw, Severity.WARNING)

                    cwe_list = item.get("extra", {}).get("metadata", {}).get("cwe", [])
                    cwe_code = cwe_list[0] if cwe_list else item.get("check_id", "")

                    findings.append(
                        Finding(
                            severity=severity,
                            code=str(cwe_code),
                            message=item.get("extra", {}).get("message", item.get("check_id", "")),
                            line=item.get("start", {}).get("line"),
                            file=str(file_path.name),
                            suggestion=item.get("extra", {}).get("fix", None),
                        )
                    )

        except FileNotFoundError:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    message="Semgrep not installed — skipping. Install with: pip install semgrep",
                )
            )
        except subprocess.TimeoutExpired:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    message=f"Semgrep timed out after {self.timeout}s",
                )
            )
        except (json.JSONDecodeError, KeyError):
            pass

        return findings

    def _run_bandit(self, file_path: Path) -> List[Finding]:
        """Run Bandit on a file and return findings."""
        findings: List[Finding] = []

        if not self.bandit_enabled:
            return findings

        try:
            result = subprocess.run(
                [
                    "bandit",
                    "-f",
                    "json",
                    "-q",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout.strip()
            if output:
                data = json.loads(output)
                for item in data.get("results", []):
                    severity_raw = item.get("issue_severity", "MEDIUM").upper()
                    severity_map = {
                        "HIGH": Severity.ERROR,
                        "MEDIUM": Severity.WARNING,
                        "LOW": Severity.INFO,
                    }
                    severity = severity_map.get(severity_raw, Severity.WARNING)

                    cwe_info = item.get("issue_cwe", {})
                    cwe_id = (
                        f"CWE-{cwe_info.get('id', '')}"
                        if cwe_info.get("id")
                        else item.get("test_id", "")
                    )

                    findings.append(
                        Finding(
                            severity=severity,
                            code=cwe_id,
                            message=item.get("issue_text", ""),
                            line=item.get("line_number"),
                            file=str(file_path.name),
                            suggestion=None,
                        )
                    )

        except FileNotFoundError:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    message="Bandit not installed — skipping. Install with: pip install bandit",
                )
            )
        except subprocess.TimeoutExpired:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    message=f"Bandit timed out after {self.timeout}s",
                )
            )
        except (json.JSONDecodeError, KeyError):
            pass

        return findings

    def _filename_for_language(self, language: Optional[str]) -> str:
        """Pick a file extension that matches the source language."""
        if language in ("javascript", "js"):
            return "generated_code.js"
        if language in ("typescript", "ts"):
            return "generated_code.ts"
        return "generated_code.py"

    def _is_python(self, filename: str) -> bool:
        return filename.endswith(".py")

    def _line_for_match(self, code: str, start: int) -> int:
        return code.count("\n", 0, start) + 1

    def _finding(
        self,
        *,
        code: str,
        message: str,
        line: Optional[int],
        filename: str,
        suggestion: str,
        severity: Severity = Severity.ERROR,
    ) -> Finding:
        return Finding(
            severity=severity,
            code=code,
            message=message,
            line=line,
            file=filename,
            suggestion=suggestion,
        )

    def _python_builtin_findings(self, code: str, filename: str) -> List[Finding]:
        findings: List[Finding] = []
        try:
            tree = ast.parse(code or "")
        except SyntaxError:
            return findings

        def target_names(node: ast.AST) -> list[str]:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                return []
            names: list[str] = []
            for target in targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
                elif isinstance(target, ast.Attribute):
                    names.append(target.attr)
            return names

        def constant_texts(node: ast.AST) -> list[str]:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return [node.value]
            if isinstance(node, ast.JoinedStr):
                return [part.value for part in node.values if isinstance(part, ast.Constant)]
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                return constant_texts(node.left) + constant_texts(node.right)
            return []

        def call_name(node: ast.Call) -> str:
            func = node.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                parts = [func.attr]
                value = func.value
                while isinstance(value, ast.Attribute):
                    parts.append(value.attr)
                    value = value.value
                if isinstance(value, ast.Name):
                    parts.append(value.id)
                return ".".join(reversed(parts))
            return ""

        def names_loaded(node: ast.AST) -> set[str]:
            return {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            }

        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for name in target_names(node):
                    if not _SENSITIVE_NAME_RE.search(name):
                        continue
                    for value in constant_texts(getattr(node, "value", None)):
                        if _SECRET_VALUE_RE.search(value):
                            findings.append(
                                self._finding(
                                    code="CWE-798",
                                    message="Hardcoded credential or secret assigned in source.",
                                    line=getattr(node, "lineno", None),
                                    filename=filename,
                                    suggestion="Load secrets from configuration or a secret manager.",
                                )
                            )
                            break

            if isinstance(node, ast.Call):
                name = call_name(node)
                if name == "eval":
                    findings.append(
                        self._finding(
                            code="CWE-95",
                            message="Dynamic eval executes untrusted code.",
                            line=getattr(node, "lineno", None),
                            filename=filename,
                            suggestion="Replace eval with a parser or an explicit allowlisted operation.",
                        )
                    )
                if name in {"pickle.load", "pickle.loads"}:
                    findings.append(
                        self._finding(
                            code="CWE-502",
                            message="Pickle deserialization can execute attacker-controlled payloads.",
                            line=getattr(node, "lineno", None),
                            filename=filename,
                            suggestion="Use a safe serialization format for untrusted data.",
                        )
                    )
                if name in {"hashlib.md5", "md5"}:
                    findings.append(
                        self._finding(
                            code="CWE-328",
                            message="MD5 is a weak hash algorithm for security-sensitive data.",
                            line=getattr(node, "lineno", None),
                            filename=filename,
                            suggestion="Use a modern password hashing function or SHA-256 where appropriate.",
                        )
                    )
                if name.startswith("subprocess.") and any(
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                ):
                    findings.append(
                        self._finding(
                            code="CWE-78",
                            message="Subprocess executes a shell command with shell=True.",
                            line=getattr(node, "lineno", None),
                            filename=filename,
                            suggestion="Pass an argument list with shell=False and validate user input.",
                        )
                    )

            if isinstance(node, ast.Return):
                value = node.value
                texts = "".join(constant_texts(value))
                has_interpolation = isinstance(value, ast.JoinedStr) and any(
                    isinstance(part, ast.FormattedValue) for part in value.values
                )
                if has_interpolation and "Location:" in texts:
                    findings.append(
                        self._finding(
                            code="CWE-601",
                            message="Redirect response uses an unvalidated target in the Location header.",
                            line=getattr(node, "lineno", None),
                            filename=filename,
                            suggestion="Validate redirect destinations against an allowlist.",
                        )
                    )
                if has_interpolation and "<" in texts and ">" in texts:
                    findings.append(
                        self._finding(
                            code="CWE-79",
                            message="HTML response interpolates unsanitized user input.",
                            line=getattr(node, "lineno", None),
                            filename=filename,
                            suggestion="Escape untrusted data before rendering HTML.",
                        )
                    )

        for func in [
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]:
            params = {arg.arg for arg in list(func.args.args) + list(func.args.kwonlyargs)}
            tainted_path_vars: set[str] = set()
            tainted_sql_vars: set[str] = set()
            for node in ast.walk(func):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                    if isinstance(node, ast.Assign):
                        text = " ".join(constant_texts(node.value)).upper()
                        if (
                            any(
                                keyword in text
                                for keyword in ("SELECT", "INSERT", "UPDATE", "DELETE")
                            )
                            and names_loaded(node.value) & params
                        ):
                            for name in target_names(node):
                                tainted_sql_vars.add(name)
                    continue
                if call_name(node.value) == "os.path.join":
                    arg_names = {arg.id for arg in node.value.args if isinstance(arg, ast.Name)}
                    if arg_names & params:
                        for name in target_names(node):
                            tainted_path_vars.add(name)
            for node in ast.walk(func):
                if not isinstance(node, ast.Call) or call_name(node) != "open" or not node.args:
                    continue
                first = node.args[0]
                if isinstance(first, ast.Name) and first.id in tainted_path_vars:
                    findings.append(
                        self._finding(
                            code="CWE-22",
                            message="File open uses a path built from unsanitized user input.",
                            line=getattr(node, "lineno", None),
                            filename=filename,
                            suggestion="Normalize and validate paths remain inside the intended base directory.",
                        )
                    )
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                name = call_name(node)
                if name.endswith(".execute") and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Name) and first.id in tainted_sql_vars:
                        findings.append(
                            self._finding(
                                code="CWE-89",
                                message="SQL query is built by concatenating user-controlled input.",
                                line=getattr(node, "lineno", None),
                                filename=filename,
                                suggestion="Use parameterized SQL queries instead of string concatenation.",
                            )
                        )
                if (
                    name in {"urllib.request.urlopen", "requests.get", "requests.post"}
                    and node.args
                ):
                    first = node.args[0]
                    if isinstance(first, ast.Name) and first.id in params:
                        findings.append(
                            self._finding(
                                code="CWE-918",
                                message="Outbound request uses a user-controlled URL.",
                                line=getattr(node, "lineno", None),
                                filename=filename,
                                suggestion="Validate URLs against an allowlist before fetching.",
                            )
                        )

        return findings

    def _javascript_builtin_findings(self, code: str, filename: str) -> List[Finding]:
        findings: List[Finding] = []

        for match in re.finditer(r"\beval\s*\(", code or ""):
            findings.append(
                self._finding(
                    code="CWE-95",
                    message="Dynamic eval executes untrusted JavaScript.",
                    line=self._line_for_match(code, match.start()),
                    filename=filename,
                    suggestion="Replace eval with an explicit parser or allowlisted operation.",
                )
            )

        pollution_pattern = re.compile(
            r"for\s*\(\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s+in\s+"
            r"([A-Za-z_$][\w$]*)\s*\)[\s\S]{0,600}?"
            r"[A-Za-z_$][\w$]*\s*\[\s*\1\s*\]\s*=",
            re.MULTILINE,
        )
        for match in pollution_pattern.finditer(code or ""):
            findings.append(
                self._finding(
                    code="CWE-1321",
                    message="Dynamic recursive merge writes attacker-controlled keys into an object.",
                    line=self._line_for_match(code, match.start()),
                    filename=filename,
                    suggestion="Reject __proto__, prototype, and constructor keys before assigning.",
                )
            )

        return findings

    def _builtin_findings(
        self,
        code: str,
        filename: str,
        language: Optional[str] = None,
    ) -> List[Finding]:
        lang = (language or "").lower()
        if lang in ("javascript", "js", "typescript", "ts") or filename.endswith((".js", ".ts")):
            return self._javascript_builtin_findings(code, filename)
        return self._python_builtin_findings(code, filename)

    def analyze(
        self,
        code: str,
        filename: str = "generated_code.py",
        language: Optional[str] = None,
    ) -> GateResult:
        """Run SAST analysis on code.

        Args:
            code: Source code to analyze
            filename: Virtual filename for context
            language: Optional language hint (python, javascript, typescript)

        Returns:
            GateResult with findings from Semgrep and (for Python) Bandit
        """
        if language:
            filename = self._filename_for_language(language)

        tmpdir = Path(tempfile.mkdtemp(prefix="sast_"))
        file_path = tmpdir / filename

        try:
            file_path.write_text(code, encoding="utf-8")

            semgrep_findings = self._run_semgrep(file_path)

            bandit_findings = self._run_bandit(file_path) if self._is_python(filename) else []

            builtin_findings = self._builtin_findings(code, filename, language)

            all_findings = semgrep_findings + bandit_findings + builtin_findings

            # Drop noise findings that are spurious on generated test code.
            all_findings = [f for f in all_findings if (f.code or "").upper() not in _IGNORED_CODES]

            has_blocking = any(
                f.severity in self.BLOCKING_SEVERITIES
                for f in all_findings
                if "not installed" not in f.message
            )

            return GateResult(
                gate_name="sast",
                passed=not has_blocking,
                findings=all_findings,
            )
        finally:
            import shutil

            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
