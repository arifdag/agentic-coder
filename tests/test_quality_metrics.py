"""Tests for research-quality metric helpers."""

from src.evaluation.quality import (
    compute_gate_quality_metrics,
    compute_mutation_summary,
    compute_oracle_metrics,
    compute_relevance_metrics,
    generate_python_mutants,
    infer_primary_target,
)


def test_infer_primary_target_prefers_metadata():
    source = "def fallback(x):\n    return x\n"
    assert infer_primary_target(source, {"func_name": "target"}) == "target"


def test_oracle_metrics_detect_assertions_and_dummy_tests():
    strong = "def test_add():\n    assert add(2, 3) == 5\n"
    dummy = "def test_dummy():\n    assert 1 == 1\n"
    class_based = "class TestAdd:\n    def test_add(self):\n        assert add(2, 3) == 5\n"

    strong_metrics = compute_oracle_metrics(strong)
    dummy_metrics = compute_oracle_metrics(dummy)
    class_metrics = compute_oracle_metrics(class_based)

    assert strong_metrics["has_assertions"] is True
    assert strong_metrics["assertion_count"] == 1
    assert strong_metrics["dummy_test_flag"] is False
    assert class_metrics["test_count"] == 1
    assert class_metrics["assertions_per_test"] == 1
    assert dummy_metrics["dummy_test_flag"] is True


def test_relevance_metrics_detect_target_signals_and_gaming():
    source = "def add(a, b):\n    return a + b\n"
    relevant = "from source_module import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    unrelated = "def test_dummy():\n    assert 1 == 1\n"

    rel = compute_relevance_metrics(relevant, source, passed=True)
    game = compute_relevance_metrics(unrelated, source, passed=True)

    assert rel["relevance_pass"] is True
    assert rel["gaming_flag"] is False
    assert game["relevance_pass"] is False
    assert game["gaming_flag"] is True


def test_gate_quality_metrics_extract_safety_and_dependency_outcomes():
    gates = [
        {"gate_name": "sast", "passed": False},
        {"gate_name": "dependency", "passed": False},
    ]
    metrics = compute_gate_quality_metrics(
        gates,
        {"vuln": "sql-injection", "phantom_packages": ["fakepkg"]},
    )

    assert metrics["safety_metrics"]["expected_vulnerable"] is True
    assert metrics["safety_metrics"]["vulnerability_detected"] is True
    assert metrics["dependency_metrics"]["expected_phantom"] is True
    assert metrics["dependency_metrics"]["phantom_detected"] is True


def test_generate_python_mutants_is_bounded_and_deterministic():
    source = "def add(a, b):\n    if a > b:\n        return a + b\n    return 1\n"

    first = generate_python_mutants(source, limit=2)
    second = generate_python_mutants(source, limit=2)

    assert len(first) == 2
    assert first == second
    assert all("source_code" in mutant for mutant in first)


def test_compute_mutation_summary_counts_outcomes():
    mutants = [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]
    summary = compute_mutation_summary(mutants, [True, False])

    assert summary["mutants_killed"] == 1
    assert summary["mutants_survived"] == 1
    assert summary["mutants_uncovered"] == 1
    assert summary["mutation_score"] == 0.5
    assert summary["mutation_coverage"] == 2 / 3
