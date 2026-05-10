"""Benchmark dataset loaders."""

from pathlib import Path
from typing import Dict

from .codejudgebench import CodeJudgeBenchDataset
from .custom_security import CustomSecurityDataset
from .cweval import CWEvalDataset
from .dep_hallucination import DepHallucinationDataset
from .projecttest import ProjectTestDataset
from .quixbugs import QuixBugsDataset
from .testgeneval import TestGenEvalDataset
from .ult import ULTDataset

__all__ = [
    "ULTDataset",
    "ProjectTestDataset",
    "CWEvalDataset",
    "CodeJudgeBenchDataset",
    "CustomSecurityDataset",
    "DepHallucinationDataset",
    "TestGenEvalDataset",
    "QuixBugsDataset",
    "get_dataset",
    "DATASET_REGISTRY",
]


def _build_registry(data_dir: Path) -> Dict[str, object]:
    return {
        "ult": ULTDataset(data_dir),
        "projecttest": ProjectTestDataset(data_dir),
        "cweval": CWEvalDataset(data_dir),
        "codejudgebench": CodeJudgeBenchDataset(data_dir),
        "security": CustomSecurityDataset(data_dir),
        "dep_hallucination": DepHallucinationDataset(data_dir),
        "testgeneval_lite": TestGenEvalDataset(data_dir, name="testgeneval_lite"),
        "testgeneval": TestGenEvalDataset(data_dir, name="testgeneval"),
        "quixbugs": QuixBugsDataset(data_dir),
    }


DATASET_REGISTRY = _build_registry(Path("data/benchmarks"))


def get_dataset(name: str, data_dir: Path = Path("data/benchmarks")):
    """Return a dataset loader by short name."""
    registry = _build_registry(data_dir)
    if name not in registry:
        raise ValueError(f"Unknown benchmark '{name}'. Choose from: {', '.join(sorted(registry))}")
    return registry[name]
