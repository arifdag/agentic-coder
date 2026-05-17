"""Tests for the official TestGenEval bridge."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.models import BenchmarkCase
from src.evaluation.testgeneval_official import (
    _docker_image_for_task,
    run_official_bridge,
    run_official_bridge_windowed,
    validate_official_prediction,
    wrap_django_official_prediction,
)
from src.main import _generate_official_testgeneval_prediction


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
        metadata={
            "id": "row-1",
            "instance_id": "django__django-1",
            "repo": "django/django",
            "version": "5.0",
            "base_commit": "abc123",
            "code_file": "django/example.py",
            "test_file": "tests/test_example.py",
            "preds_context": {"code_src": "def target():\n    return 1\n", "last": ""},
            "test_patch": "diff --git a/tests/test_example.py b/tests/test_example.py\n",
            "patch": "diff --git a/django/example.py b/django/example.py\n",
            "baseline_covs": {"line_coverage": 10.0},
        },
    )


def _valid_prediction() -> str:
    return (
        "from django.example import target\n\n"
        "def test_target_returns_one():\n"
        "    assert target() == 1\n"
    )


def _window_case(row_id: str, instance_id: str, repo: str, version: str) -> BenchmarkCase:
    return BenchmarkCase(
        id=f"testgeneval_lite-{instance_id}",
        code="def target():\n    return 1\n",
        language="python",
        metadata={
            "id": row_id,
            "instance_id": instance_id,
            "repo": repo,
            "version": version,
            "base_commit": "abc123",
            "code_file": "django/example.py",
            "test_file": "tests/test_example.py",
            "preds_context": {"code_src": "def target():\n    return 1\n", "last": ""},
            "test_patch": "",
            "patch": "",
            "baseline_covs": {},
        },
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
        generate=lambda case: _valid_prediction(),
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
                    "def test_target_returns_one():\n"
                    "    assert target() == 1\n"
                ]
            },
        }
    ]
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))[0]["status"] == "ok"
    tasks = [
        json.loads(line) for line in result.tasks_path.read_text(encoding="utf-8").splitlines()
    ]
    assert tasks[0]["id"] == "row-1"
    assert tasks[0]["instance_id"] == "django__django-1"
    assert tasks[0]["repo"] == "django/django"
    assert tasks[0]["baseline_covs"] == {"line_coverage": 10.0}
    assert result.summary_copied is not None
    assert result.report_copied is not None
    assert len(calls) == 2


def test_official_windowed_groups_by_image_and_deletes_after(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    cases = [
        _window_case("row-1", "django__django-1", "django/django", "5.0"),
        _window_case("row-2", "django__django-2", "django/django", "5.0"),
        _window_case("row-3", "psf__requests-1", "psf/requests", "2.31"),
    ]

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "generate_report.py" in cmd:
            output_arg = cmd[cmd.index("--output_dir") + 1]
            out_dir = Path(output_arg.rstrip("/"))
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "llm-agent-gdr_summary.json").write_text(
                json.dumps({"full_pass_at_1": 1.0}),
                encoding="utf-8",
            )
            (out_dir / "llm-agent-gdr_report.json").write_text(
                json.dumps({"with_logs": ["row-1", "row-2", "row-3"]}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = run_official_bridge_windowed(
        benchmark="testgeneval_lite",
        cases=cases,
        output_dir=tmp_path / "out",
        model_name="llm-agent-gdr",
        official_repo_dir=_official_repo(tmp_path),
        namespace="kdjain",
        window_images=1,
        num_processes_per_image=2,
        delete_images_after=True,
        skip_mutation=True,
        generate=lambda case: _valid_prediction(),
    )

    eval_cmds = [cmd for cmd in calls if "run_evaluation.py" in cmd]
    delete_cmds = [cmd for cmd in calls if cmd[:3] == ["docker", "image", "rm"]]
    report_cmds = [cmd for cmd in calls if "generate_report.py" in cmd]

    assert result.counts["image_count"] == 2
    assert result.counts["window_count"] == 2
    assert [window["prediction_count"] for window in result.counts["windows"]] == [2, 1]
    assert len(eval_cmds) == 2
    assert len(delete_cmds) == 2
    assert len(report_cmds) == 1
    assert "kdjain/swe-bench-django__django-testbed:5.0" in result.counts["deleted_images"]
    assert "kdjain/swe-bench-psf__requests-testbed:2.31" in result.counts["deleted_images"]
    assert all("--skip_mutation" in cmd for cmd in eval_cmds)
    assert all("2" == cmd[cmd.index("--num_processes") + 1] for cmd in eval_cmds)
    assert result.summary_copied is not None
    assert result.report_copied is not None


def test_official_windowed_image_name_uses_repo_and_version():
    task = {"repo": "django/django", "version": "5.0"}

    assert _docker_image_for_task(task, "kdjain") == "kdjain/swe-bench-django__django-testbed:5.0"


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
            generate=lambda case: _valid_prediction(),
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
        generate=lambda case: _valid_prediction(),
    )

    eval_cmd = calls[0]
    assert "run_evaluation.py" in eval_cmd
    assert "--predictions_path" in eval_cmd
    predictions_arg = eval_cmd[eval_cmd.index("--predictions_path") + 1]
    assert "\\" not in predictions_arg
    assert "--log_dir" in eval_cmd
    log_dir_arg = eval_cmd[eval_cmd.index("--log_dir") + 1]
    assert "\\" not in log_dir_arg
    assert log_dir_arg.endswith("/")
    assert "--swe_bench_tasks" in eval_cmd
    tasks_arg = eval_cmd[eval_cmd.index("--swe_bench_tasks") + 1]
    assert tasks_arg.endswith("official_tasks.jsonl")
    assert "\\" not in tasks_arg
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
    output_dir_arg = report_cmd[report_cmd.index("--output_dir") + 1]
    assert "\\" not in output_dir_arg
    assert output_dir_arg.endswith("/")


def test_official_bridge_reuses_existing_predictions(monkeypatch, tmp_path):
    bench_out = tmp_path / "out" / "testgeneval_lite"
    reports_dir = bench_out / "official_reports"
    bench_out.mkdir(parents=True)
    reports_dir.mkdir()
    (bench_out / "predictions.jsonl").write_text(
        json.dumps(
            {
                "id": "row-1",
                "instance_id": "django__django-1",
                "model_name_or_path": "llm-agent-gdr",
                "preds": {"full": [_valid_prediction()]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bench_out / "official_tasks.jsonl").write_text(
        json.dumps({"id": "row-1", "instance_id": "django__django-1"}) + "\n",
        encoding="utf-8",
    )
    (bench_out / "generation_manifest.json").write_text(
        json.dumps(
            [
                {"status": "ok"},
                {"status": "failed", "error": "Generated test code is empty"},
            ]
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, **kwargs):
        if "generate_report.py" in cmd:
            (reports_dir / "llm-agent-gdr_summary.json").write_text(
                json.dumps({"full_pass_at_1": 1.0}),
                encoding="utf-8",
            )
            (reports_dir / "llm-agent-gdr_report.json").write_text(
                json.dumps({"with_logs": ["row-1"]}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = run_official_bridge(
        benchmark="testgeneval_lite",
        cases=[],
        output_dir=tmp_path / "out",
        model_name="llm-agent-gdr",
        official_repo_dir=_official_repo(tmp_path),
        generate=lambda case: (_ for _ in ()).throw(AssertionError("should not generate")),
        reuse_predictions=True,
    )

    assert result.counts["written"] == 1
    assert result.counts["failed"] == 1
    assert result.counts["total_attempted"] == 2
    assert len(result.commands_run) == 2
    assert result.summary_copied is not None


def test_official_bridge_stages_log_dir_when_output_path_has_spaces(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    output_dir = tmp_path / "out with space"
    reports_dir = output_dir / "testgeneval_lite" / "official_reports"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "generate_report.py" in cmd:
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "llm-agent-gdr_summary.json").write_text(
                json.dumps({"full_pass_at_1": 1.0}),
                encoding="utf-8",
            )
            (reports_dir / "llm-agent-gdr_report.json").write_text(
                json.dumps({"with_logs": ["row-1"]}),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = run_official_bridge(
        benchmark="testgeneval_lite",
        cases=[_case()],
        output_dir=output_dir,
        model_name="llm-agent-gdr",
        official_repo_dir=_official_repo(tmp_path),
        generate=lambda case: _valid_prediction(),
    )

    assert result.staging_logs_dir is not None
    assert result.counts["staging_logs_used"] is True
    eval_cmd = calls[0]
    log_dir_arg = eval_cmd[eval_cmd.index("--log_dir") + 1]
    assert "out with space" not in log_dir_arg
    assert " " not in log_dir_arg


def test_official_bridge_uses_fallback_when_report_cli_fails(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        if "generate_report.py" in cmd:
            return SimpleNamespace(returncode=1, stdout="path error", stderr="KeyError")
        logs_dir = tmp_path / "out" / "testgeneval_lite" / "official_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "django__django-1.llm-agent-gdr.full.eval.log").write_text(
            "Tests Errored\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def fake_fallback(
        official_repo_dir,
        predictions_path,
        tasks_path,
        logs_dir,
        reports_dir,
        model_name,
    ):
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / f"{model_name}_summary.json").write_text(
            json.dumps({"full_pass_at_1": 0.0}),
            encoding="utf-8",
        )
        (reports_dir / f"{model_name}_report.json").write_text(
            json.dumps({"with_logs": ["django__django-1"]}),
            encoding="utf-8",
        )
        (reports_dir / f"{model_name}_full.json").write_text(
            json.dumps({"django__django-1": {"full": {}}}),
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "src.evaluation.testgeneval_official._write_fallback_report_outputs",
        fake_fallback,
    )

    result = run_official_bridge(
        benchmark="testgeneval_lite",
        cases=[_case("testgeneval_lite-django__django-1")],
        output_dir=tmp_path / "out",
        model_name="llm-agent-gdr",
        official_repo_dir=_official_repo(tmp_path),
        generate=lambda case: _valid_prediction(),
    )

    assert result.returncodes == [0, 1]
    assert result.errors == []
    assert result.counts["report_fallback_used"] is True
    assert result.summary_copied is not None
    assert result.report_copied is not None


def test_validate_official_prediction_rejects_unusable_outputs():
    with pytest.raises(ValueError, match="empty"):
        validate_official_prediction("   ")
    with pytest.raises(ValueError, match="valid Python"):
        validate_official_prediction("Here are the tests you requested.")
    with pytest.raises(ValueError, match="pytest test"):
        validate_official_prediction("from django.example import target\n")
    with pytest.raises(ValueError, match="dummy assertions"):
        validate_official_prediction("def test_placeholder():\n    assert True\n")
    with pytest.raises(ValueError, match="must not import or use pytest"):
        validate_official_prediction(
            "import pytest\n\n"
            "def test_error_path():\n"
            "    with pytest.raises(ValueError):\n"
            "        raise ValueError('x')\n"
        )
    with pytest.raises(ValueError, match="not test classes"):
        validate_official_prediction(
            "class TestTarget:\n"
            "    def test_target_returns_one(self):\n"
            "        assert target() == 1\n"
        )


def test_official_bridge_rejects_empty_generation_and_skips_eval(tmp_path):
    result = run_official_bridge(
        benchmark="testgeneval_lite",
        cases=[_case()],
        output_dir=tmp_path / "out",
        model_name="llm-agent-gdr",
        official_repo_dir=_official_repo(tmp_path),
        generate=lambda case: "",
    )

    assert result.predictions_path.read_text(encoding="utf-8") == ""
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest[0]["status"] == "failed"
    assert "empty" in manifest[0]["error"]
    assert result.counts == {"written": 0, "failed": 1, "total_attempted": 1}
    assert result.commands_run == []
    assert "No predictions were written" in result.errors[0]


def test_official_generation_retries_with_validation_feedback():
    class FakeAgent:
        def __init__(self):
            self.calls = []

        def generate_testgeneval(self, **kwargs):
            self.calls.append(kwargs)
            content = "" if len(self.calls) == 1 else _valid_prediction()
            return SimpleNamespace(test_code=content)

    agent = FakeAgent()

    case = _case()
    case.metadata["repo"] = "example/project"

    result = _generate_official_testgeneval_prediction(agent, case)

    assert result == _valid_prediction()
    assert len(agent.calls) == 2
    assert agent.calls[0]["feedback"] is None
    assert agent.calls[1]["feedback"] == "Generated test code is empty"


def test_django_official_prediction_wraps_functions_in_simple_testcase():
    wrapped = wrap_django_official_prediction(
        "from django.db.migrations.serializer import BaseSerializer\n\n"
        "def test_base_serializer_raises():\n"
        "    try:\n"
        "        BaseSerializer(1).serialize()\n"
        "    except NotImplementedError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('expected error')\n\n"
        "def test_nested_helper_class():\n"
        "    class Helper:\n"
        "        def method(self):\n"
        "            return 1\n"
        "    assert Helper().method() == 1\n"
    )

    assert "from django.test import SimpleTestCase" in wrapped
    assert "class TestsHarness(SimpleTestCase):" in wrapped
    assert "    def test_base_serializer_raises(self):" in wrapped
    assert "    def test_nested_helper_class(self):" in wrapped
    assert "        def method(self):" in wrapped
    validate_official_prediction(wrapped, allow_test_classes=True)
