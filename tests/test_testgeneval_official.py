"""Tests for the official TestGenEval bridge."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.evaluation.models import BenchmarkCase
from src.evaluation.testgeneval_official import run_official_bridge


def _official_repo(tmp_path):
    repo = tmp_path / "official"
    repo.mkdir()
    (repo / "run_evaluation.py").write_text("# fake eval\n", encoding="utf-8")
    (repo / "generate_report.py").write_text("# fake report\n", encoding="utf-8")
    return repo


def _case(case_id: str = "testgeneval_lite-django__django-1") -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        code="def target():\n    return 1\n",
        language="python",
        metadata={"id": "row-1", "instance_id": "django__django-1"},
    )


def test_official_bridge_writes_prediction_jsonl_and_manifest(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "generate_report.py" in cmd:
            out_dir = tmp_path / "out" / "testgeneval_lite" / "official_reports"
            (out_dir / "llm-agent-gdr_summary.json").write_text(
                json.dumps({"resolved": 1}),
                encoding="utf-8",
            )
            (out_dir / "llm-agent-gdr_report.json").write_text(
                json.dumps({"instances": []}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = run_official_bridge(
        benchmark="testgeneval_lite",
        cases=[_case()],
        output_dir=tmp_path / "out",
        model_name="llm-agent-gdr",
        official_repo_dir=_official_repo(tmp_path),
        generate=lambda case: "from django.example import target\n\ndef test_target():\n    assert target() == 1\n",
    )

    records = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "id": "row-1",
            "instance_id": "django__django-1",
            "model_name_or_path": "llm-agent-gdr",
            "preds": {
                "full": [
                    "from django.example import target\n\n"
                    "def test_target():\n"
                    "    assert target() == 1\n"
                ]
            },
        }
    ]
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))[0]["status"] == "ok"
    assert result.summary_copied is not None
    assert result.report_copied is not None
    assert len(calls) == 2


def test_official_bridge_omits_generation_failures_and_skips_eval(tmp_path):
    result = run_official_bridge(
        benchmark="testgeneval_lite",
        cases=[_case()],
        output_dir=tmp_path / "out",
        model_name="llm-agent-gdr",
        official_repo_dir=_official_repo(tmp_path),
        generate=lambda case: (_ for _ in ()).throw(RuntimeError("provider down")),
    )

    assert result.predictions_path.read_text(encoding="utf-8") == ""
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest[0]["status"] == "failed"
    assert "provider down" in manifest[0]["error"]
    assert result.counts == {"written": 0, "failed": 1, "total_attempted": 1}
    assert result.commands_run == []
    assert "No predictions were written" in result.errors[0]


def test_official_bridge_requires_official_scripts(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing required files"):
        run_official_bridge(
            benchmark="testgeneval_lite",
            cases=[_case()],
            output_dir=tmp_path / "out",
            model_name="llm-agent-gdr",
            official_repo_dir=tmp_path,
            generate=lambda case: "def test_target():\n    assert True\n",
        )


def test_official_bridge_command_arguments(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    run_official_bridge(
        benchmark="testgeneval_lite",
        cases=[_case()],
        output_dir=tmp_path / "out",
        model_name="model-x",
        official_repo_dir=_official_repo(tmp_path),
        namespace="kdjain",
        timeout=900,
        num_processes=2,
        skip_mutation=True,
        skip_existing=True,
        generate=lambda case: "def test_target():\n    assert True\n",
    )

    eval_cmd = calls[0]
    assert "run_evaluation.py" in eval_cmd
    assert "--predictions_path" in eval_cmd
    assert "--log_dir" in eval_cmd
    assert "--swe_bench_tasks" in eval_cmd
    assert "kjain14/testgenevallite" in eval_cmd
    assert "--namespace" in eval_cmd
    assert "kdjain" in eval_cmd
    assert "--timeout" in eval_cmd
    assert "900" in eval_cmd
    assert "--num_processes" in eval_cmd
    assert "2" in eval_cmd
    assert "--skip_existing" in eval_cmd
    assert "--skip_mutation" in eval_cmd

    report_cmd = calls[1]
    assert "generate_report.py" in report_cmd
    assert "--output_dir" in report_cmd
