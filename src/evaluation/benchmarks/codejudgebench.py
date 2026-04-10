"""CodeJudgeBench loader.

Dataset: LLM judge evaluation pairs (code, tests, human judgment).
Source: HuggingFace dataset ``mattymchen/codejudgebench`` (Apache 2.0).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR

log = logging.getLogger(__name__)

HF_DATASET_ID = "mattymchen/codejudgebench"
LOCAL_DIR_NAME = "codejudgebench"


class CodeJudgeBenchDataset:
    """Loads the CodeJudgeBench testgen subset."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._root = data_dir / LOCAL_DIR_NAME

    @property
    def name(self) -> str:
        return "codejudgebench"

    @property
    def language(self) -> Optional[str]:
        return None  # mixed

    def download(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        marker = self._root / ".downloaded"
        if marker.exists():
            return self._root

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=HF_DATASET_ID,
                repo_type="dataset",
                local_dir=str(self._root),
            )
        except ImportError:
            log.warning(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
            return self._root
        except Exception as exc:
            log.warning("Failed to download CodeJudgeBench: %s", exc)
            return self._root

        marker.write_text("ok")
        return self._root

    def load(self) -> List[BenchmarkCase]:
        self.download()
        cases: List[BenchmarkCase] = []

        # Look for the testgen subset (JSONL or Parquet)
        for jsonl in sorted(self._root.rglob("*.jsonl")):
            if "testgen" not in jsonl.name.lower() and "testgen" not in str(jsonl.parent).lower():
                continue
            for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code = entry.get("code") or entry.get("prompt", "")
                if not code:
                    continue
                cases.append(BenchmarkCase(
                    id=f"cjb-{i}",
                    code=code,
                    language=entry.get("language", "python").lower(),
                    metadata=entry,
                    user_request="Generate comprehensive unit tests",
                ))

        # Fallback: any JSONL in root
        if not cases:
            for jsonl in sorted(self._root.rglob("*.jsonl"))[:3]:
                for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    code = entry.get("code") or entry.get("prompt", "")
                    if not code:
                        continue
                    cases.append(BenchmarkCase(
                        id=f"cjb-{i}",
                        code=code,
                        language=entry.get("language", "python").lower(),
                        metadata=entry,
                        user_request="Generate comprehensive unit tests",
                    ))

        log.info("Loaded %d cases from CodeJudgeBench", len(cases))
        return cases
