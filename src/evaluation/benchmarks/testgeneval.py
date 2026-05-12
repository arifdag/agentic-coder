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
    "preds_context",
    "test_patch",
    "patch",
    "baseline_covs",
    "id",
    "dataset_id",
    "split",
)


def _infer_import_module(code_file: object) -> Optional[str]:
    """Infer the Python import module path from a code_file path.

    Examples:
        django/db/models/base.py   -> django.db.models.base
        sklearn/preprocessing/_label.py -> sklearn.preprocessing._label
        src/mypkg/core.py           -> mypkg.core
        pkg/__init__.py            -> pkg
    """
    if not isinstance(code_file, str) or not code_file.strip():
        return None

    path = code_file.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("src/"):
        path = path[len("src/") :]
    if not path.endswith(".py"):
        return None

    path = path[:-3]
    if path.endswith("/__init__"):
        path = path[: -len("/__init__")]

    module = ".".join(part for part in path.split("/") if part)
    if not module or module.startswith("."):
        return None
    return module


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
            metadata["execution_context"] = "repo"
            # Ensure repo execution metadata is present even when fields are empty/null
            metadata.setdefault("repo", row_dict.get("repo"))
            metadata["code_file"] = metadata.get("code_file") or row_dict.get("code_file")
            metadata["test_file"] = metadata.get("test_file") or row_dict.get("test_file")
            metadata["base_commit"] = metadata.get("base_commit") or row_dict.get("base_commit")

            import_module = _infer_import_module(metadata.get("code_file"))
            if import_module:
                metadata["import_module"] = import_module

            # Build a stable case id.
            instance_id = metadata.get("instance_id")
            case_id = f"{self._name}-{instance_id}" if instance_id else f"{self._name}-{i}"

            code_file = metadata.get("code_file", "unknown")
            import_hint = (
                f" Import from the real repo module `{import_module}`, not source_module."
                if import_module
                else ""
            )
            user_request = (
                f"Generate comprehensive pytest unit tests for the provided code file "
                f"({code_file})."
                f"{import_hint}"
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
