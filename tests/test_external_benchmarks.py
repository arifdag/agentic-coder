"""Tests for external benchmark loaders and bug-detection metrics."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from src.config import EvalConfig
from src.evaluation.benchmarks import get_dataset
from src.evaluation.benchmarks.quixbugs import QuixBugsDataset
from src.evaluation.benchmarks.testgeneval import (
    TestGenEvalDataset as _TestGenEvalDataset,
)
from src.evaluation.benchmarks.testgeneval import (
    _infer_import_module,
)
from src.evaluation.models import BenchmarkCase, EvalMetrics, EvalResult
from src.evaluation.runner import BenchmarkRunner


def test_testgeneval_loader_extracts_code_and_metadata(monkeypatch, tmp_path):
    rows = [
        {
            "repo": "django/django",
            "base_commit": "abc123",
            "version": "5.0",
            "instance_id": "django__django-1",
            "code_file": "django/example.py",
            "test_file": "tests/test_example.py",
            "baseline_covs": {"line_coverage": 10.0},
            "id": "row-1",
            "preds_context": {"code_src": "def target():\n    return 1\n"},
            "test_src": "def test_human_reference(): pass",
        }
    ]
    fake_module = types.SimpleNamespace(load_dataset=lambda *a, **k: rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    dataset = _TestGenEvalDataset(data_dir=tmp_path, name="testgeneval_lite")
    cases = dataset.load()

    assert dataset.name == "testgeneval_lite"
    assert len(cases) == 1
    assert cases[0].code == "def target():\n    return 1\n"
    assert cases[0].language == "python"
    assert cases[0].metadata["repo"] == "django/django"
    assert cases[0].metadata["dataset_id"] == "kjain14/testgenevallite"
    assert cases[0].metadata["import_module"] == "django.example"
    assert "django.example" in (cases[0].user_request or "")
    assert "source_module" in (cases[0].user_request or "")
    assert "test_human_reference" not in (cases[0].user_request or "")


def test_testgeneval_infers_import_modules_from_code_file():
    assert _infer_import_module("django/db/models/base.py") == "django.db.models.base"
    assert _infer_import_module("sklearn/preprocessing/_label.py") == (
        "sklearn.preprocessing._label"
    )
    assert _infer_import_module("src/mypkg/core.py") == "mypkg.core"
    assert _infer_import_module("pkg/__init__.py") == "pkg"
    assert _infer_import_module(r"src\mypkg\core.py") == "mypkg.core"
    assert _infer_import_module("README.md") is None


def test_quixbugs_loader_pairs_fixed_and_buggy_sources(tmp_path):
    root = tmp_path / "QuixBugs"
    correct = root / "correct_python_programs"
    buggy = root / "python_programs"
    correct.mkdir(parents=True)
    buggy.mkdir(parents=True)
    (correct / "foo.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (buggy / "foo.py").write_text("def foo():\n    return 0\n", encoding="utf-8")

    dataset = QuixBugsDataset(data_dir=tmp_path)
    dataset.download = lambda: root
    cases = dataset.load()

    assert len(cases) == 1
    assert cases[0].id == "quixbugs-foo"
    assert cases[0].code == "def foo():\n    return 1\n"
    assert cases[0].metadata["buggy_code"] == "def foo():\n    return 0\n"
    assert cases[0].metadata["target"] == "foo"


def test_external_benchmarks_are_registered():
    for name in ("testgeneval_lite", "testgeneval", "quixbugs"):
        dataset = get_dataset(name)
        assert dataset.name == name


def test_eval_metrics_aggregate_coverage_gain_and_bug_detection():
    results = [
        EvalResult(
            case_id="a",
            coverage_metrics={"coverage_gain": "12.5"},
            bug_metrics={"eligible": True, "bug_detected": True},
        ),
        EvalResult(
            case_id="b",
            coverage_metrics={"coverage_gain": -2.5},
            bug_metrics={"eligible": True, "bug_detected": False},
        ),
    ]

    metrics = EvalMetrics.from_results(results, dataset_name="external")

    assert metrics.avg_coverage_gain == pytest.approx(5.0)
    assert metrics.bug_detection_rate == pytest.approx(0.5)


def test_benchmark_runner_bug_detection_uses_buggy_code(monkeypatch, tmp_path):
    class FakeSandboxExecutor:
        def __init__(self, config):
            self.config = config

        def execute(self, source_code, test_code):
            assert source_code == "def target():\n    return 0\n"
            assert "test_target" in test_code
            return SimpleNamespace(success=False)

    monkeypatch.setattr("src.verification.sandbox.SandboxExecutor", FakeSandboxExecutor)

    config = SimpleNamespace(
        evaluation=EvalConfig(quality_mode="fast"),
        sandbox=SimpleNamespace(),
        pipeline=SimpleNamespace(max_retries=0),
    )
    dataset = SimpleNamespace(name="fake", load=lambda: [])
    runner = BenchmarkRunner(config=config, dataset=dataset, results_dir=str(tmp_path))
    case = BenchmarkCase(
        id="bug",
        code="def target():\n    return 1\n",
        language="python",
        metadata={"buggy_code": "def target():\n    return 0\n", "target": "target"},
    )

    state = {
        "status": "success",
        "generated_tests": (
            "from source_module import target\n\n"
            "def test_target():\n"
            "    assert target() == 1\n"
        ),
        "test_functions": ["test_target"],
        "sandbox_tests_run": 1,
        "sandbox_tests_passed": 1,
        "verification_report": {"gates": [{"gate_name": "sandbox", "passed": True}]},
        "retry_count": 0,
    }

    result = runner._run_one(case, lambda **kwargs: state)

    assert result.bug_metrics["eligible"] is True
    assert result.bug_metrics["fixed_passed"] is True
    assert result.bug_metrics["buggy_failed"] is True
    assert result.bug_metrics["bug_detected"] is True


def test_benchmark_runner_scores_dep_hallucination_by_gate_oracle(tmp_path):
    config = SimpleNamespace(
        evaluation=EvalConfig(quality_mode="off"),
        sandbox=SimpleNamespace(),
        pipeline=SimpleNamespace(max_retries=0),
    )
    dataset = SimpleNamespace(name="dep_hallucination", load=lambda: [])
    runner = BenchmarkRunner(config=config, dataset=dataset, results_dir=str(tmp_path))
    case = BenchmarkCase(
        id="dep-phantom",
        code="import fakepkg\n",
        language="python",
        metadata={"phantom_packages": ["fakepkg"]},
    )
    state = {
        "status": "failed_after_retries",
        "generated_tests": "",
        "test_functions": [],
        "verification_report": {
            "gates": [
                {
                    "gate_name": "dependency",
                    "passed": False,
                    "findings": [
                        {
                            "severity": "error",
                            "code": "PHANTOM-PKG",
                            "message": "Package 'fakepkg' not found on PyPI.",
                        }
                    ],
                }
            ]
        },
        "retry_count": 0,
    }

    result = runner._run_one(case, lambda **kwargs: state)

    assert result.passed is True
    assert result.pipeline_state["pipeline_passed"] is False
    assert result.pipeline_state["benchmark_oracle"]["reason"] == "phantom_detected"


def test_benchmark_runner_failed_only_reuses_passing_results(monkeypatch, tmp_path):
    cases = [
        BenchmarkCase(id="pass", code="def ok():\n    return 1\n", language="python"),
        BenchmarkCase(id="fail", code="def bad():\n    return 0\n", language="python"),
    ]
    config = SimpleNamespace(
        evaluation=EvalConfig(quality_mode="off"),
        sandbox=SimpleNamespace(),
        pipeline=SimpleNamespace(max_retries=0),
    )
    dataset = SimpleNamespace(name="resume", load=lambda: cases)
    runner = BenchmarkRunner(config=config, dataset=dataset, results_dir=str(tmp_path))
    (runner.results_dir / "pass.json").write_text(
        EvalResult(case_id="pass", passed=True).model_dump_json(),
        encoding="utf-8",
    )
    (runner.results_dir / "fail.json").write_text(
        EvalResult(case_id="fail", passed=False).model_dump_json(),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_pipeline(**kwargs):
        calls.append(kwargs["code"])
        return {
            "status": "success",
            "generated_tests": "",
            "test_functions": [],
            "verification_report": {"gates": []},
            "retry_count": 0,
        }

    monkeypatch.setattr("src.graph.pipeline.run_pipeline", fake_pipeline)

    results = runner.run(failed_only=True)

    assert [result.case_id for result in results] == ["pass", "fail"]
    assert calls == [cases[1].code]
    assert results[0].passed is True
    assert results[1].passed is True


def test_benchmark_runner_skips_mutation_when_relevance_fails(monkeypatch, tmp_path):
    config = SimpleNamespace(
        evaluation=EvalConfig(quality_mode="full"),
        sandbox=SimpleNamespace(),
        pipeline=SimpleNamespace(max_retries=0),
    )
    dataset = SimpleNamespace(name="fake", load=lambda: [])
    runner = BenchmarkRunner(config=config, dataset=dataset, results_dir=str(tmp_path))
    case = BenchmarkCase(
        id="irrelevant",
        code="def target():\n    return 1\n",
        language="python",
        metadata={"target": "target"},
    )
    state = {
        "status": "success",
        "generated_tests": "def test_smoke():\n    assert True\n",
        "verification_report": {"gates": [{"gate_name": "sandbox", "passed": True}]},
    }
    monkeypatch.setattr(
        "src.evaluation.runner.generate_python_mutants",
        lambda source_code, limit: (_ for _ in ()).throw(AssertionError("mutation ran")),
    )

    payload = runner._build_quality_payload(
        case, state, state["verification_report"]["gates"], 100.0
    )

    assert payload["mutation_metrics"] == {
        "skipped": True,
        "skip_reason": "relevance_failed",
    }
