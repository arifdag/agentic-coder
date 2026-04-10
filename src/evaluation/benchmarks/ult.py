"""ULT (UnLeakedTestBench) benchmark loader.

Dataset: 3,909 Python functions with high cyclomatic complexity.
Source: https://github.com/huangd1999/UnLeakedTestBench
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR, ensure_repo

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/huangd1999/UnLeakedTestBench.git"
LOCAL_DIR_NAME = "UnLeakedTestBench"


class ULTDataset:
    """Loads the ULT benchmark."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._root = data_dir / LOCAL_DIR_NAME

    @property
    def name(self) -> str:
        return "ult"

    @property
    def language(self) -> Optional[str]:
        return "python"

    def download(self) -> Path:
        return ensure_repo(REPO_URL, self._root)

    def load(self) -> List[BenchmarkCase]:
        self.download()
        cases: List[BenchmarkCase] = []

        # ULT stores data as JSONL in data/ULT.jsonl (or similar)
        for jsonl in sorted(self._root.rglob("*.jsonl")):
            for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                code = entry.get("code") or entry.get("function") or entry.get("prompt", "")
                if not code:
                    continue

                cases.append(BenchmarkCase(
                    id=f"ult-{i}",
                    code=code,
                    language="python",
                    metadata=entry,
                    user_request="Generate comprehensive unit tests",
                ))
            if cases:
                break  # use first JSONL found

        # Fallback: look for .py files under a dataset/ folder
        if not cases:
            for py_file in sorted(self._root.rglob("*.py"))[:500]:
                code = py_file.read_text(encoding="utf-8", errors="replace")
                if len(code) < 20:
                    continue
                cases.append(BenchmarkCase(
                    id=f"ult-{py_file.stem}",
                    code=code,
                    language="python",
                    user_request="Generate comprehensive unit tests",
                ))

        log.info("Loaded %d cases from ULT benchmark", len(cases))
        return cases
