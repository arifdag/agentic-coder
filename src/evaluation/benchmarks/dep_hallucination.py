"""Dependency hallucination benchmark loader.

Ships with the repo under ``data/benchmarks/dep_hallucination/``.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR

log = logging.getLogger(__name__)


class DepHallucinationDataset:
    """Loads the curated phantom-package benchmark."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._root = data_dir / "dep_hallucination"

    @property
    def name(self) -> str:
        return "dep_hallucination"

    @property
    def language(self) -> Optional[str]:
        return None  # mixed

    def download(self) -> Path:
        return self._root

    def load(self) -> List[BenchmarkCase]:
        manifest_path = self._root / "manifest.json"
        if not manifest_path.exists():
            log.warning("Dep hallucination manifest not found at %s", manifest_path)
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases: List[BenchmarkCase] = []

        for entry in manifest.get("cases", []):
            file_path = self._root / entry["file"]
            if not file_path.exists():
                continue
            code = file_path.read_text(encoding="utf-8")
            cases.append(BenchmarkCase(
                id=f"dep-{file_path.stem}",
                code=code,
                language=entry.get("language", "python"),
                metadata={
                    "phantom_packages": entry.get("phantom_packages", []),
                    "valid_packages": entry.get("valid_packages", []),
                    "file": entry["file"],
                },
                user_request="Generate comprehensive unit tests",
            ))

        log.info("Loaded %d cases from dep hallucination benchmark", len(cases))
        return cases
