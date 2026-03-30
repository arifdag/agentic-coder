"""Gate 1: Static Application Security Testing (SAST) with Semgrep and Bandit."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

from .models import GateResult, Finding, Severity


class SastAnalyzer:
    """Run SAST tools (Semgrep + Bandit) on generated code."""

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
                    "--config", self.semgrep_rules,
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

                    findings.append(Finding(
                        severity=severity,
                        code=str(cwe_code),
                        message=item.get("extra", {}).get("message", item.get("check_id", "")),
                        line=item.get("start", {}).get("line"),
                        file=str(file_path.name),
                        suggestion=item.get("extra", {}).get("fix", None),
                    ))

        except FileNotFoundError:
            findings.append(Finding(
                severity=Severity.INFO,
                message="Semgrep not installed — skipping. Install with: pip install semgrep",
            ))
        except subprocess.TimeoutExpired:
            findings.append(Finding(
                severity=Severity.WARNING,
                message=f"Semgrep timed out after {self.timeout}s",
            ))
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
                    "-f", "json",
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
                    cwe_id = f"CWE-{cwe_info.get('id', '')}" if cwe_info.get("id") else item.get("test_id", "")

                    findings.append(Finding(
                        severity=severity,
                        code=cwe_id,
                        message=item.get("issue_text", ""),
                        line=item.get("line_number"),
                        file=str(file_path.name),
                        suggestion=None,
                    ))

        except FileNotFoundError:
            findings.append(Finding(
                severity=Severity.INFO,
                message="Bandit not installed — skipping. Install with: pip install bandit",
            ))
        except subprocess.TimeoutExpired:
            findings.append(Finding(
                severity=Severity.WARNING,
                message=f"Bandit timed out after {self.timeout}s",
            ))
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

            bandit_findings = (
                self._run_bandit(file_path) if self._is_python(filename) else []
            )

            all_findings = semgrep_findings + bandit_findings

            has_errors = any(
                f.severity == Severity.ERROR
                for f in all_findings
                if "not installed" not in f.message
            )

            return GateResult(
                gate_name="sast",
                passed=not has_errors,
                findings=all_findings,
            )
        finally:
            import shutil
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
