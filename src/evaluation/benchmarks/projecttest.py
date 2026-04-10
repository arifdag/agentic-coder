"""ProjectTest benchmark loader.

Dataset: 20 Python projects + 20 JavaScript projects (multi-file).
Source: https://github.com/YiboWANG214/ProjectTest
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..models import BenchmarkCase
from .utils import DEFAULT_DATA_DIR, ensure_repo

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/YiboWANG214/ProjectTest.git"
LOCAL_DIR_NAME = "ProjectTest"


class ProjectTestDataset:
    """Loads the ProjectTest benchmark."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, language_filter: Optional[str] = None):
        self._root = data_dir / LOCAL_DIR_NAME
        self._lang_filter = language_filter

    @property
    def name(self) -> str:
        return "projecttest"

    @property
    def language(self) -> Optional[str]:
        return self._lang_filter

    def download(self) -> Path:
        return ensure_repo(REPO_URL, self._root)

    def _detect_language(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext in (".js", ".jsx", ".ts", ".tsx"):
            return "javascript"
        return "python"

    def load(self) -> List[BenchmarkCase]:
        self.download()
        cases: List[BenchmarkCase] = []

        # ProjectTest organises projects under data/{python,javascript}/
        for lang_dir in ("python", "javascript", "Python", "JavaScript"):
            base = self._root / "data" / lang_dir
            if not base.exists():
                base = self._root / lang_dir
            if not base.exists():
                continue

            for project_dir in sorted(base.iterdir()):
                if not project_dir.is_dir():
                    continue
                lang = "javascript" if "javascript" in lang_dir.lower() else "python"
                if self._lang_filter and lang != self._lang_filter:
                    continue

                exts = ("*.py",) if lang == "python" else ("*.js", "*.ts")
                project_code_parts = []
                for ext in exts:
                    for src in sorted(project_dir.rglob(ext)):
                        if "test" in src.name.lower() or "node_modules" in str(src):
                            continue
                        project_code_parts.append(
                            f"# --- {src.relative_to(project_dir)} ---\n"
                            + src.read_text(encoding="utf-8", errors="replace")
                        )
                if not project_code_parts:
                    continue

                cases.append(BenchmarkCase(
                    id=f"pt-{lang}-{project_dir.name}",
                    code="\n\n".join(project_code_parts),
                    language=lang,
                    metadata={"project": project_dir.name},
                    user_request="Generate comprehensive unit tests",
                ))

        # Fallback: scan root for JSONL
        if not cases:
            for jsonl in sorted(self._root.rglob("*.jsonl")):
                for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    code = entry.get("code") or entry.get("source", "")
                    if not code:
                        continue
                    lang = entry.get("language", "python").lower()
                    cases.append(BenchmarkCase(
                        id=f"pt-{i}",
                        code=code,
                        language=lang,
                        metadata=entry,
                        user_request="Generate comprehensive unit tests",
                    ))

        log.info("Loaded %d cases from ProjectTest benchmark", len(cases))
        return cases
