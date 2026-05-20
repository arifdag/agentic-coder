"""Regenerate only missing official TestGenEval predictions.

This script is intentionally separate from the normal windowed scorer. The
windowed scorer can reuse and score existing predictions, but it does not have a
"generate only failed predictions" mode. This utility fills that gap by reading
an existing TestGenEval run directory, finding task IDs that have no prediction,
and appending newly recovered predictions to the same predictions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.unit_test import UnitTestAgent  # noqa: E402
from src.config import Config, get_role_llm  # noqa: E402
from src.evaluation.benchmarks import get_dataset  # noqa: E402
from src.main import _generate_official_testgeneval_prediction  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected manifest list in {path}")
    return data


def _write_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _prediction_ids(predictions_path: Path) -> set[str]:
    ids: set[str] = set()
    for row in _read_jsonl(predictions_path):
        pred_id = row.get("id")
        if isinstance(pred_id, str) and pred_id:
            ids.add(pred_id)
    return ids


def _task_ids(tasks_path: Path) -> list[str]:
    ids: list[str] = []
    for row in _read_jsonl(tasks_path):
        task_id = row.get("id")
        if isinstance(task_id, str) and task_id:
            ids.append(task_id)
    return ids


def _missing_prediction_entries(
    manifest: list[dict[str, Any]],
    task_ids: list[str],
    existing_prediction_ids: set[str],
) -> list[dict[str, Any]]:
    """Return manifest-like entries for tasks without a prediction record."""
    manifest_by_id = {
        entry.get("official_id"): entry
        for entry in manifest
        if isinstance(entry.get("official_id"), str)
    }
    missing: list[dict[str, Any]] = []
    for task_id in task_ids:
        if task_id in existing_prediction_ids:
            continue
        entry = manifest_by_id.get(task_id)
        if entry is None:
            entry = {
                "case_index": None,
                "case_id": None,
                "official_id": task_id,
                "status": "failed",
                "error": "missing prediction and manifest entry",
            }
            manifest.append(entry)
            manifest_by_id[task_id] = entry
        missing.append(entry)
    return missing


def _backup(path: Path, stamp: str) -> None:
    if path.is_file():
        shutil.copy2(path, path.with_name(f"{path.name}.bak_{stamp}"))


def _case_map(benchmark: str, data_dir: Path) -> dict[str, Any]:
    dataset = get_dataset(benchmark, data_dir=data_dir)
    cases = dataset.load()
    return {case.metadata.get("id") or case.id: case for case in cases}


def _print_missing_summary(missing: list[dict[str, Any]]) -> None:
    print(f"Missing predictions: {len(missing)}")
    if not missing:
        return
    print("Failure reasons:")
    reasons = Counter(str(entry.get("error") or "<none>") for entry in missing)
    for reason, count in reasons.most_common(20):
        print(f"{count:4}  {reason[:180]}")
    print("\nMissing official IDs:")
    for entry in missing:
        print(entry["official_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", default="testgeneval", choices=["testgeneval", "testgeneval_lite"]
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Existing benchmark run directory containing predictions.jsonl and generation_manifest.json",
    )
    parser.add_argument("--provider", default=None, help="Override provider, e.g. ollama")
    parser.add_argument("--model-name", default="llm-agent-gdr")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum missing predictions to try"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only list missing predictions")
    parser.add_argument(
        "--no-backup", action="store_true", help="Do not backup predictions/manifest first"
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit nonzero if any selected prediction still fails to regenerate",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    predictions_path = run_dir / "predictions.jsonl"
    tasks_path = run_dir / "official_tasks.jsonl"
    manifest_path = run_dir / "generation_manifest.json"

    if not predictions_path.is_file():
        raise SystemExit(f"Missing predictions file: {predictions_path}")
    if not tasks_path.is_file():
        raise SystemExit(f"Missing official tasks file: {tasks_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"Missing generation manifest: {manifest_path}")

    manifest = _read_manifest(manifest_path)
    existing_ids = _prediction_ids(predictions_path)
    missing = _missing_prediction_entries(manifest, _task_ids(tasks_path), existing_ids)
    if args.limit is not None:
        missing = missing[: args.limit]

    _print_missing_summary(missing)
    if args.dry_run or not missing:
        return 0

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _backup(predictions_path, stamp)
        _backup(manifest_path, stamp)

    config = Config.load(provider=args.provider)
    data_dir = args.data_dir or Path(config.evaluation.data_dir)
    cases_by_id = _case_map(args.benchmark, data_dir)
    agent = UnitTestAgent(get_role_llm(config, "coding"))

    recovered = 0
    still_failed = 0
    with predictions_path.open("a", encoding="utf-8", newline="\n") as pred_file:
        for entry in missing:
            official_id = entry["official_id"]
            case = cases_by_id.get(official_id)
            if case is None:
                entry["status"] = "failed"
                entry["error"] = "missing case in dataset load"
                still_failed += 1
                _write_manifest(manifest_path, manifest)
                print(f"[missing-case] {official_id}")
                continue

            print(f"[generate] {official_id}")
            try:
                test_code = _generate_official_testgeneval_prediction(
                    agent,
                    case,
                    max_attempts=args.max_attempts,
                )
                record = {
                    "id": official_id,
                    "instance_id": case.metadata.get("instance_id") or official_id,
                    "model_name_or_path": args.model_name,
                    "preds": {"full": [test_code]},
                }
                pred_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                pred_file.flush()
                existing_ids.add(official_id)
                entry["status"] = "ok"
                entry["error"] = None
                recovered += 1
                print(f"[ok] {official_id}")
            except Exception as exc:  # noqa: BLE001 - preserve per-case failure and continue
                entry["status"] = "failed"
                entry["error"] = str(exc)
                still_failed += 1
                print(f"[failed] {official_id}: {exc}")
            _write_manifest(manifest_path, manifest)

    print(f"Recovered: {recovered}")
    print(f"Still failed: {still_failed}")
    print(f"Prediction total now: {len(existing_ids)}")
    return 1 if args.strict_exit and still_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
