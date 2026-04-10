"""Custom security benchmark loader.

Ships with the repo under ``data/benchmarks/security_suite/``.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR

log = logging.getLogger(__name__)


class CustomSecurityDataset:
    """Loads the curated security-vulnerability benchmark."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._root = data_dir / "security_suite"

    @property
    def name(self) -> str:
        return "security"

    @property
    def language(self) -> Optional[str]:
        return None  # mixed Python + JS

    def download(self) -> Path:
        return self._root  # ships with repo, nothing to fetch

    def load(self) -> List[BenchmarkCase]:
        manifest_path = self._root / "manifest.json"
        if not manifest_path.exists():
            log.warning("Security suite manifest not found at %s", manifest_path)
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases: List[BenchmarkCase] = []

        for entry in manifest.get("cases", []):
            file_path = self._root / entry["file"]
            if not file_path.exists():
                continue
            code = file_path.read_text(encoding="utf-8")
            cases.append(BenchmarkCase(
                id=f"sec-{file_path.stem}",
                code=code,
                language=entry.get("language", "python"),
                expected_cwe=entry.get("cwe"),
                metadata={"vuln": entry.get("vuln", ""), "file": entry["file"]},
                user_request="Generate comprehensive unit tests",
            ))

        log.info("Loaded %d cases from custom security suite", len(cases))
        return cases
