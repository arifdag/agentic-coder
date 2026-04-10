"""Shared helpers for benchmark loaders."""

from pathlib import Path
import subprocess
import logging

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/benchmarks")


def ensure_repo(repo_url: str, dest: Path) -> Path:
    """Clone *repo_url* into *dest* if it doesn't already exist.

    Returns the destination path.
    """
    if dest.exists() and any(dest.iterdir()):
        log.info("Repo already present at %s -- skipping clone", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    log.info("Cloning %s -> %s", repo_url, dest)

    try:
        from git import Repo
        Repo.clone_from(repo_url, str(dest), depth=1)
    except ImportError:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True,
            capture_output=True,
        )

    return dest
