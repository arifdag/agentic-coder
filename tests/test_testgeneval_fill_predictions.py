"""Tests for failed-only TestGenEval prediction filling helpers."""

from scripts.fill_testgeneval_predictions import _missing_prediction_entries


def test_missing_prediction_entries_uses_task_order_and_skips_existing():
    manifest = [
        {
            "case_index": 0,
            "case_id": "testgeneval-a",
            "official_id": "a-1",
            "status": "ok",
            "error": None,
        },
        {
            "case_index": 1,
            "case_id": "testgeneval-b",
            "official_id": "b-1",
            "status": "failed",
            "error": "Generated test code is empty",
        },
    ]

    missing = _missing_prediction_entries(
        manifest,
        ["a-1", "b-1", "c-1"],
        {"a-1"},
    )

    assert [entry["official_id"] for entry in missing] == ["b-1", "c-1"]
    assert missing[0]["error"] == "Generated test code is empty"
    assert missing[1]["error"] == "missing prediction and manifest entry"
    assert manifest[-1]["official_id"] == "c-1"
