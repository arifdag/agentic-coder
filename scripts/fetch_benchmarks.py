"""Fetch benchmark datasets used by the evaluation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks import get_dataset  # noqa: E402

FETCH_CHOICES = [
    "ult",
    "projecttest",
    "cweval",
    "codejudgebench",
    "security",
    "dep_hallucination",
    "testgeneval_lite",
    "testgeneval",
    "quixbugs",
]


def _touch_manifest(data_dir: Path, benchmark: str, count: int) -> None:
    manifest_dir = data_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{benchmark}.txt"
    manifest.write_text(f"benchmark={benchmark}\ncases={count}\n", encoding="utf-8")


def fetch_one(name: str, data_dir: Path) -> bool:
    print(f"[fetch] {name}")
    try:
        dataset = get_dataset(name, data_dir=data_dir)
        if hasattr(dataset, "download"):
            dataset.download()
        cases = dataset.load()
        if not cases:
            print(f"[error] {name}: loaded zero cases")
            return False
        _touch_manifest(data_dir, name, len(cases))
    except Exception as exc:  # noqa: BLE001 - CLI should summarize all failures
        print(f"[error] {name}: {exc}")
        return False
    print(f"[ok] {name}: {len(cases)} case(s)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=[*FETCH_CHOICES, "all"],
        default="all",
        help="Benchmark to fetch. Use 'all' for practical paper-lite dependencies.",
    )
    parser.add_argument("--data-dir", default="data/benchmarks", help="Benchmark data root.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    names = FETCH_CHOICES if args.benchmark == "all" else [args.benchmark]
    ok = True
    for name in names:
        ok = fetch_one(name, data_dir) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
