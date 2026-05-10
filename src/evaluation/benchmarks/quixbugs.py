"""QuixBugs benchmark loader.

Dataset: classic program-repair benchmark with paired buggy and correct Python programs.
Source: https://github.com/jkoppel/QuixBugs
"""

import ast
import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR, ensure_repo

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/jkoppel/QuixBugs.git"
LOCAL_DIR_NAME = "QuixBugs"


class QuixBugsDataset:
    """Loads the QuixBugs benchmark by pairing correct and buggy Python sources."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._root = data_dir / LOCAL_DIR_NAME

    @property
    def name(self) -> str:
        return "quixbugs"

    @property
    def language(self) -> Optional[str]:
        return "python"

    def download(self) -> Path:
        return ensure_repo(REPO_URL, self._root)

    @staticmethod
    def _infer_target(source: str, filename: str) -> Optional[str]:
        """Best-effort target/function hint from filename or top-level definitions."""
        stem = Path(filename).stem
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return stem
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.lower() == stem.lower():
                    return node.name
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return stem

    def load(self) -> List[BenchmarkCase]:
        self.download()
        cases: List[BenchmarkCase] = []

        correct_dir = self._root / "correct_python_programs"
        buggy_dir = self._root / "python_programs"

        if not correct_dir.exists() or not buggy_dir.exists():
            log.warning(
                "QuixBugs repo missing expected directories: %s / %s", correct_dir, buggy_dir
            )
            return cases

        def is_program(path: Path) -> bool:
            return not path.stem.endswith("_test") and path.stem != "node"

        correct_files = {p.stem: p for p in sorted(correct_dir.glob("*.py")) if is_program(p)}
        buggy_files = {p.stem: p for p in sorted(buggy_dir.glob("*.py")) if is_program(p)}

        for stem, correct_path in correct_files.items():
            buggy_path = buggy_files.get(stem)
            if buggy_path is None:
                log.debug("No buggy counterpart for %s; skipping", stem)
                continue

            try:
                correct_src = correct_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                log.warning("Failed to read correct source %s: %s", correct_path, exc)
                continue

            try:
                buggy_src = buggy_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                log.warning("Failed to read buggy source %s: %s", buggy_path, exc)
                continue

            if not correct_src.strip() or not buggy_src.strip():
                log.debug("Empty source for %s; skipping", stem)
                continue

            target = self._infer_target(correct_src, stem)
            metadata = {
                "program": stem,
                "benchmark": "quixbugs",
                "buggy_code": buggy_src,
                "target": target,
            }

            cases.append(
                BenchmarkCase(
                    id=f"quixbugs-{stem}",
                    code=correct_src,
                    language="python",
                    metadata=metadata,
                    user_request="Generate comprehensive pytest unit tests",
                )
            )

        log.info("Loaded %d cases from QuixBugs benchmark", len(cases))
        return cases
