"""TestGenEval benchmark loader.

Datasets: kjain14/testgenevallite and kjain14/testgeneval (Hugging Face).
Source: https://huggingface.co/datasets/kjain14/testgenevallite
"""

import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR

log = logging.getLogger(__name__)

DATASET_IDS = {
    "testgeneval_lite": "kjain14/testgenevallite",
    "testgeneval": "kjain14/testgeneval",
}

# Metadata keys to preserve explicitly.
_METADATA_KEYS = (
    "repo",
    "base_commit",
    "version",
    "instance_id",
    "code_file",
    "test_file",
    "baseline_covs",
    "id",
    "dataset_id",
    "split",
)


class TestGenEvalDataset:
    """Loads the TestGenEval benchmark from Hugging Face datasets."""

    def __init__(
        self, data_dir: Path = DEFAULT_DATA_DIR, name: str = "testgeneval_lite", split: str = "test"
    ):
        self._data_dir = data_dir
        if name not in DATASET_IDS:
            raise ValueError(
                f"Unknown TestGenEval variant '{name}'. Choose from: {', '.join(DATASET_IDS)}"
            )
        self._name = name
        self._dataset_id = DATASET_IDS[name]
        self._split = split

    @property
    def name(self) -> str:
        return self._name

    @property
    def language(self) -> Optional[str]:
        return "python"

    def load(self) -> List[BenchmarkCase]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' library is required to load TestGenEval. "
                "Install it with: pip install datasets"
            ) from exc

        log.info("Loading %s (split=%s) from Hugging Face", self._dataset_id, self._split)
        try:
            cache_dir = self._data_dir / self._name
            cache_dir.mkdir(parents=True, exist_ok=True)
            ds = load_dataset(self._dataset_id, split=self._split, cache_dir=str(cache_dir))
        except Exception as exc:
            log.warning("Failed to load %s split=%s: %s", self._dataset_id, self._split, exc)
            return []

        cases: List[BenchmarkCase] = []
        for i, row in enumerate(ds):
            row_dict = dict(row) if hasattr(row, "keys") else row
            code_src = ""
            preds = row_dict.get("preds_context")
            if isinstance(preds, dict):
                code_src = preds.get("code_src") or ""
            if not code_src:
                code_src = row_dict.get("code_src") or ""
            if not code_src or not isinstance(code_src, str):
                log.debug("Skipping row %d: missing code_src", i)
                continue

            metadata = {k: row_dict[k] for k in _METADATA_KEYS if k in row_dict}
            metadata["dataset_id"] = self._dataset_id
            metadata["split"] = self._split

            # Build a stable case id.
            instance_id = metadata.get("instance_id")
            case_id = f"{self._name}-{instance_id}" if instance_id else f"{self._name}-{i}"

            user_request = (
                f"Generate comprehensive pytest unit tests for the provided code file "
                f"({metadata.get('code_file', 'unknown')})."
            )

            cases.append(
                BenchmarkCase(
                    id=case_id,
                    code=code_src,
                    language="python",
                    metadata=metadata,
                    user_request=user_request,
                )
            )

        log.info("Loaded %d cases from %s", len(cases), self._dataset_id)
        return cases
