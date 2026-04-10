"""Tests for Phase 6: Evaluation & Benchmarking Infrastructure."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.models import (
    AblationConfig,
    BenchmarkCase,
    BenchmarkDataset,
    EvalMetrics,
    EvalResult,
)
from src.evaluation.benchmarks.custom_security import CustomSecurityDataset
from src.evaluation.benchmarks.dep_hallucination import DepHallucinationDataset
from src.evaluation.benchmarks import get_dataset
from src.evaluation.ablation import generate_variants
from src.evaluation.cost import CostAnalyzer
from src.config import EvalConfig


# ── Model tests ───────────────────────────────────────────────────────


class TestBenchmarkCase:
    def test_basic_creation(self):
        c = BenchmarkCase(id="t1", code="x=1", language="python")
        assert c.id == "t1"
        assert c.language == "python"
        assert c.metadata == {}

    def test_with_cwe(self):
        c = BenchmarkCase(id="t2", code="x", language="python", expected_cwe="CWE-89")
        assert c.expected_cwe == "CWE-89"


class TestEvalResult:
    def test_defaults(self):
        r = EvalResult(case_id="c1")
        assert r.passed is False
        assert r.error is None
        assert r.gate_results == []

    def test_full_result(self):
        r = EvalResult(
            case_id="c2",
            passed=True,
            elapsed_seconds=1.5,
            tests_run=5,
            tests_passed=5,
            coverage=87.5,
            iterations=2,
            gate_results=[{"gate_name": "sast", "passed": True}],
        )
        assert r.passed
        assert r.coverage == 87.5


class TestEvalMetrics:
    def test_from_empty(self):
        m = EvalMetrics.from_results([], dataset_name="empty")
        assert m.total == 0
        assert m.pass_rate == 0.0

    def test_from_results(self):
        results = [
            EvalResult(case_id="a", passed=True, elapsed_seconds=1.0, iterations=2,
                       gate_results=[{"gate_name": "sast", "passed": True}]),
            EvalResult(case_id="b", passed=False, elapsed_seconds=3.0, iterations=4,
                       gate_results=[{"gate_name": "sast", "passed": False}]),
        ]
        m = EvalMetrics.from_results(results, dataset_name="test")
        assert m.total == 2
        assert m.passed == 1
        assert m.failed == 1
        assert m.pass_rate == 0.5
        assert m.avg_iterations == 3.0
        assert m.avg_time == 2.0
        assert m.gate_pass_rates["sast"] == 0.5

    def test_to_markdown(self):
        m = EvalMetrics(dataset_name="demo", total=10, passed=7, failed=3, pass_rate=0.7,
                        avg_iterations=2.1, avg_time=1.5)
        md = m.to_markdown()
        assert "demo" in md
        assert "70.0%" in md


class TestAblationConfig:
    def test_creation(self):
        a = AblationConfig(name="test", sast_enabled=False, retry_budget=1)
        assert a.name == "test"
        assert a.sast_enabled is False
        assert a.retry_budget == 1


# ── Custom benchmark loader tests ────────────────────────────────────


class TestCustomSecurityDataset:
    def test_loads_from_repo_data(self):
        ds = CustomSecurityDataset(data_dir=Path("data/benchmarks"))
        cases = ds.load()
        assert len(cases) >= 10
        assert ds.name == "security"
        assert all(c.language in ("python", "javascript") for c in cases)
        cwe_cases = [c for c in cases if c.expected_cwe]
        assert len(cwe_cases) >= 10

    def test_missing_dir_returns_empty(self):
        ds = CustomSecurityDataset(data_dir=Path("/nonexistent/path"))
        cases = ds.load()
        assert cases == []


class TestDepHallucinationDataset:
    def test_loads_from_repo_data(self):
        ds = DepHallucinationDataset(data_dir=Path("data/benchmarks"))
        cases = ds.load()
        assert len(cases) >= 8
        assert ds.name == "dep_hallucination"
        has_phantoms = [c for c in cases if c.metadata.get("phantom_packages")]
        assert len(has_phantoms) >= 6

    def test_clean_cases_have_no_phantoms(self):
        ds = DepHallucinationDataset(data_dir=Path("data/benchmarks"))
        cases = ds.load()
        clean = [c for c in cases if "clean" in c.id]
        assert len(clean) >= 2
        for c in clean:
            assert c.metadata["phantom_packages"] == []


# ── Benchmark registry test ──────────────────────────────────────────


class TestBenchmarkRegistry:
    def test_get_known_datasets(self):
        for name in ("security", "dep_hallucination"):
            ds = get_dataset(name)
            assert hasattr(ds, "load")
            assert hasattr(ds, "name")

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_dataset("nonexistent_benchmark")


# ── Ablation variant generation tests ────────────────────────────────


class TestAblationVariants:
    def test_full_grid(self):
        variants = generate_variants()
        assert len(variants) == 2 * 2 * 2 * 4  # 32

    def test_single_axis(self):
        variants = generate_variants(axes=["retries"])
        assert len(variants) == 4
        budgets = {v.retry_budget for v in variants}
        assert budgets == {0, 1, 3, 5}

    def test_two_axes(self):
        variants = generate_variants(axes=["sast", "judge"])
        assert len(variants) == 4
        sast_vals = {v.sast_enabled for v in variants}
        assert sast_vals == {True, False}

    def test_variant_names_unique(self):
        variants = generate_variants()
        names = [v.name for v in variants]
        assert len(names) == len(set(names))


# ── CostAnalyzer tests ───────────────────────────────────────────────


class TestCostAnalyzer:
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            analyzer = CostAnalyzer(log_dir=d)
            runs = analyzer.analyze()
            assert runs == []

    def test_parses_audit_log(self):
        with tempfile.TemporaryDirectory() as d:
            log_data = {
                "source": "test_file.py",
                "entries": [
                    {"generated_artifact": "a" * 400, "repair_context": {}},
                    {"generated_artifact": "b" * 200, "repair_context": {"msg": "c" * 100}},
                ],
            }
            Path(d, "log1.json").write_text(json.dumps(log_data))

            analyzer = CostAnalyzer(log_dir=d, chars_per_token=4, cost_per_million_tokens=1.0)
            runs = analyzer.analyze()
            assert len(runs) == 1
            r = runs[0]
            assert r.iterations == 2
            assert r.total_chars == 700
            assert r.estimated_tokens == 175
            assert r.estimated_cost_usd == pytest.approx(175 / 1_000_000, rel=1e-6)

    def test_summarize(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                log_data = {"source": f"file_{i}", "entries": [{"generated_artifact": "x" * 100}]}
                Path(d, f"log_{i}.json").write_text(json.dumps(log_data))

            analyzer = CostAnalyzer(log_dir=d, chars_per_token=4)
            summary = analyzer.summarize()
            assert summary["total_runs"] == 3
            assert summary["total_iterations"] == 3
            assert summary["avg_iterations"] == 1.0

    def test_to_markdown(self):
        with tempfile.TemporaryDirectory() as d:
            log_data = {"source": "demo.py", "entries": [{"generated_artifact": "test"}]}
            Path(d, "log.json").write_text(json.dumps(log_data))

            analyzer = CostAnalyzer(log_dir=d)
            md = analyzer.to_markdown()
            assert "Cost Analysis" in md
            assert "demo.py" in md

    def test_nonexistent_dir(self):
        analyzer = CostAnalyzer(log_dir="/no/such/dir")
        assert analyzer.analyze() == []
        md = analyzer.to_markdown()
        assert "No audit logs found" in md


# ── BenchmarkDataset protocol test ───────────────────────────────────


class TestProtocol:
    def test_custom_security_satisfies_protocol(self):
        ds = CustomSecurityDataset()
        assert isinstance(ds, BenchmarkDataset)

    def test_dep_hallucination_satisfies_protocol(self):
        ds = DepHallucinationDataset()
        assert isinstance(ds, BenchmarkDataset)


# ── EvalConfig tests ─────────────────────────────────────────────────


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.data_dir == "data/benchmarks"
        assert cfg.results_dir == "eval_results"
        assert cfg.max_cases is None
        assert cfg.parallel == 1

    def test_from_env(self):
        with patch.dict("os.environ", {
            "EVAL_DATA_DIR": "/custom/data",
            "EVAL_RESULTS_DIR": "/custom/results",
            "EVAL_MAX_CASES": "50",
        }):
            cfg = EvalConfig.from_env()
            assert cfg.data_dir == "/custom/data"
            assert cfg.results_dir == "/custom/results"
            assert cfg.max_cases == 50
