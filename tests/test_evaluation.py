"""Tests for Phase 6: Evaluation & Benchmarking Infrastructure."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import EvalConfig
from src.evaluation.ablation import generate_variants
from src.evaluation.benchmarks import get_dataset
from src.evaluation.benchmarks.custom_security import CustomSecurityDataset
from src.evaluation.benchmarks.dep_hallucination import DepHallucinationDataset
from src.evaluation.cost import CostAnalyzer
from src.evaluation.models import (
    AblationConfig,
    BenchmarkCase,
    BenchmarkDataset,
    EvalMetrics,
    EvalResult,
)

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
        assert r.coverage_metrics == {}
        assert r.mutation_metrics == {}

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
            coverage_metrics={"target_line_coverage": 90.0},
            mutation_metrics={"mutants_killed": 3, "mutants_survived": 1},
        )
        assert r.passed
        assert r.coverage == 87.5
        assert r.coverage_metrics["target_line_coverage"] == 90.0
        assert r.mutation_metrics["mutants_killed"] == 3


class TestEvalMetrics:
    def test_from_empty(self):
        m = EvalMetrics.from_results([], dataset_name="empty")
        assert m.total == 0
        assert m.pass_rate == 0.0

    def test_from_results(self):
        results = [
            EvalResult(
                case_id="a",
                passed=True,
                elapsed_seconds=1.0,
                iterations=2,
                gate_results=[{"gate_name": "sast", "passed": True}],
            ),
            EvalResult(
                case_id="b",
                passed=False,
                elapsed_seconds=3.0,
                iterations=4,
                gate_results=[{"gate_name": "sast", "passed": False}],
            ),
        ]
        m = EvalMetrics.from_results(results, dataset_name="test")
        assert m.total == 2
        assert m.passed == 1
        assert m.failed == 1
        assert m.pass_rate == 0.5
        assert m.avg_iterations == 3.0
        assert m.avg_time == 2.0
        assert m.gate_pass_rates["sast"] == 0.5

    def test_from_results_aggregates_execution_metrics(self):
        results = [
            EvalResult(
                case_id="repo-pass",
                execution_metrics={
                    "execution_context": "repo",
                    "eligible": True,
                    "infrastructure_pass": True,
                    "repo_setup_pass": True,
                },
            ),
            EvalResult(
                case_id="repo-fail",
                execution_metrics={
                    "execution_context": "repo",
                    "eligible": True,
                    "infrastructure_pass": False,
                    "repo_setup_pass": False,
                },
            ),
            EvalResult(
                case_id="single",
                execution_metrics={
                    "execution_context": "single-file",
                    "eligible": False,
                },
            ),
        ]

        m = EvalMetrics.from_results(results, dataset_name="exec")

        assert m.infrastructure_eligible_total == 2
        assert m.infrastructure_total == 2
        assert m.infrastructure_passed == 1
        assert m.infrastructure_pass_rate == pytest.approx(0.5)
        assert m.repo_setup_passed == 1
        assert m.repo_setup_pass_rate == pytest.approx(0.5)
        assert m.execution_context_counts["repo"] == 2
        assert m.execution_context_counts["single-file"] == 1

    def test_from_results_research_quality_metrics(self):
        results = [
            EvalResult(
                case_id="a",
                passed=True,
                coverage=80.0,
                coverage_metrics={
                    "line_coverage": 80.0,
                    "branch_coverage": 50.0,
                    "target_line_coverage": 90.0,
                    "target_branch_coverage": 75.0,
                },
                mutation_metrics={
                    "mutants_killed": 4,
                    "mutants_survived": 1,
                    "mutants_uncovered": 1,
                },
                relevance_metrics={"relevance_pass": True, "gaming_flag": False},
                reliability_metrics={"reliable": True, "flaky": False},
                oracle_metrics={"has_assertions": True, "assertion_count": 4, "test_count": 2},
                safety_metrics={"expected_vulnerable": True, "vulnerability_detected": True},
                dependency_metrics={"expected_phantom": True, "phantom_detected": True},
            ),
            EvalResult(
                case_id="b",
                passed=False,
                error="Error code: 429 rate limit",
                quality={"provider_error": True},
                coverage_metrics={
                    "line_coverage": 40.0,
                    "branch_coverage": 25.0,
                    "target_line_coverage": 50.0,
                    "target_branch_coverage": 25.0,
                },
                mutation_metrics={"mutants_killed": 1, "mutants_survived": 4},
                relevance_metrics={"relevance_pass": False, "gaming_flag": True},
                reliability_metrics={"reliable": False, "flaky": True},
                oracle_metrics={"has_assertions": False, "assertion_count": 0, "test_count": 1},
                safety_metrics={"expected_vulnerable": True, "vulnerability_detected": False},
                dependency_metrics={"expected_clean": True, "clean_accepted": True},
            ),
        ]

        m = EvalMetrics.from_results(results, dataset_name="quality")

        assert m.provider_error_count == 1
        assert m.clean_total == 1
        assert m.clean_pass_rate == 1.0
        assert m.avg_line_coverage == 60.0
        assert m.avg_branch_coverage == 37.5
        assert m.avg_target_line_coverage == 70.0
        assert m.avg_target_branch_coverage == 50.0
        assert m.mutation_score == pytest.approx(5 / 10)
        assert m.mutation_coverage == pytest.approx(10 / 11)
        assert m.relevance_pass_rate == 0.5
        assert m.gaming_rate == 0.5
        assert m.reliability_rate == 0.5
        assert m.flakiness_rate == 0.5
        assert m.assertion_presence_rate == 0.5
        assert m.avg_assertions_per_test == pytest.approx(4 / 3)
        assert m.sast_catch_rate == 0.5
        assert m.vulnerability_escape_rate == 0.5
        assert m.dependency_phantom_detection_rate == 1.0
        assert m.dependency_clean_acceptance_rate == 1.0

    def test_quality_denominators_ignore_clean_unlabeled_cases(self):
        results = [
            EvalResult(
                case_id="vuln",
                safety_metrics={"expected_vulnerable": True, "vulnerability_detected": True},
                dependency_metrics={"expected_phantom": True, "phantom_detected": True},
            ),
            EvalResult(
                case_id="clean",
                safety_metrics={"expected_vulnerable": False, "vulnerability_detected": False},
                dependency_metrics={
                    "expected_phantom": False,
                    "phantom_detected": False,
                    "expected_clean": False,
                    "clean_accepted": False,
                },
            ),
        ]

        m = EvalMetrics.from_results(results, dataset_name="quality")

        assert m.sast_catch_rate == 1.0
        assert m.dependency_phantom_detection_rate == 1.0
        assert m.dependency_clean_acceptance_rate is None

    def test_provider_errors_fall_back_to_pipeline_error_text(self):
        result = EvalResult(
            case_id="rate_limited",
            passed=False,
            quality={"provider_error": False},
            pipeline_state={"error_message": "Error code: 429 rate limit"},
        )

        m = EvalMetrics.from_results([result], dataset_name="quality")

        assert m.provider_error_count == 1
        assert m.clean_total == 0

    def test_to_markdown(self):
        m = EvalMetrics(
            dataset_name="demo",
            total=10,
            passed=7,
            failed=3,
            pass_rate=0.7,
            avg_iterations=2.1,
            avg_time=1.5,
        )
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


class TestProjectTestLoader:
    """Regression tests for ProjectTest multi-file concatenation.

    Bugs this guards against:
    - Intra-project imports (``from blackjack.base import Card``) survived
      concatenation and then failed with ``No module named 'blackjack'``.
    - ``from __future__ import X`` from a non-first file landed in the middle
      of the combined module, raising
      ``SyntaxError: from __future__ imports must occur at the beginning``.
    """

    def _build_fake_project(self, tmp_path, name: str, files: dict):
        """Create ``tmp_path/ProjectTest/dataset/Python/<name>/{files}``."""
        project_dir = tmp_path / "ProjectTest" / "dataset" / "Python" / name
        project_dir.mkdir(parents=True)
        for filename, content in files.items():
            (project_dir / filename).write_text(content, encoding="utf-8")
        return tmp_path

    def _build_fake_js_project(self, tmp_path, name: str, files: dict):
        """Create ``tmp_path/ProjectTest/dataset/JS/<name>/{files}``."""
        project_dir = tmp_path / "ProjectTest" / "dataset" / "JS" / name
        project_dir.mkdir(parents=True)
        for filename, content in files.items():
            (project_dir / filename).write_text(content, encoding="utf-8")
        return tmp_path

    def test_intra_project_imports_are_stripped(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_project(
            tmp_path,
            "blackjack",
            {
                "__init__.py": "from blackjack.base import Card as Card\n",
                "base.py": "class Card:\n    pass\n",
                "game.py": (
                    "from blackjack.base import Card\n"
                    "import blackjack.base\n"
                    "def deal():\n    return Card()\n"
                ),
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="python")
        ds.download = lambda: data_dir  # skip git clone
        cases = ds.load()
        assert len(cases) == 1
        combined = cases[0].code
        # intra-project imports must be gone
        assert "from blackjack.base" not in combined
        assert "import blackjack.base" not in combined
        # but the actual class definition is preserved
        assert "class Card" in combined

    def test_future_imports_are_hoisted(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_project(
            tmp_path,
            "mypkg",
            {
                "a.py": "def a(): return 1\n",  # no future import
                "b.py": ("from __future__ import unicode_literals\n" "def b(): return 2\n"),
                "c.py": (
                    "from __future__ import absolute_import, division\n" "def c(): return 3\n"
                ),
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="python")
        ds.download = lambda: data_dir
        cases = ds.load()
        assert len(cases) == 1
        combined = cases[0].code
        # The hoisted header must be the FIRST non-blank, non-comment line.
        # Otherwise Python will complain about future imports not being first.
        assert combined.startswith("from __future__ import")
        # All three features collected, dedup'd and sorted
        first_line = combined.splitlines()[0]
        assert "absolute_import" in first_line
        assert "division" in first_line
        assert "unicode_literals" in first_line
        # And the combined source must actually be parseable
        import ast

        ast.parse(combined)

    def test_external_imports_are_preserved(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_project(
            tmp_path,
            "myproj",
            {
                "core.py": (
                    "import numpy as np\n"
                    "from collections import Counter\n"
                    "def f(): return Counter()\n"
                ),
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="python")
        ds.download = lambda: data_dir
        cases = ds.load()
        assert len(cases) == 1
        combined = cases[0].code
        assert "import numpy as np" in combined
        assert "from collections import Counter" in combined

    def test_aliased_intra_project_import_becomes_local_alias(self, tmp_path):
        # Regression: `from pkg.dealer import BlackjackDealer as Dealer`
        # was being dropped entirely, so downstream `Dealer(...)` raised
        # NameError. Now it should become `Dealer = BlackjackDealer`.
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_project(
            tmp_path,
            "blackjack",
            {
                "dealer.py": "class BlackjackDealer:\n    pass\n",
                "game.py": (
                    "from blackjack.dealer import BlackjackDealer as Dealer\n"
                    "class Game:\n"
                    "    def init(self):\n"
                    "        self.dealer = Dealer()\n"
                ),
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="python")
        ds.download = lambda: data_dir
        cases = ds.load()
        combined = cases[0].code
        # The alias assignment is present
        assert "Dealer = BlackjackDealer" in combined
        # And the combined module actually runs (Dealer is defined at import
        # time so we can instantiate Game().init() without NameError).
        ns: dict = {}
        exec(compile(combined, "<combined>", "exec"), ns)
        g = ns["Game"]()
        g.init()
        assert isinstance(g.dealer, ns["BlackjackDealer"])

    def test_init_py_runs_last_so_reexport_aliases_resolve(self, tmp_path):
        # Regression: rlcard-style projects put re-exports in __init__.py:
        #     from blackjack.dealer import BlackjackDealer as Dealer
        # After rewriting this becomes ``Dealer = BlackjackDealer``, which
        # MUST run after ``class BlackjackDealer`` has been defined in
        # dealer.py. But __init__.py sorts alphabetically FIRST (``_`` < ``d``),
        # so the naive layout evaluates the alias before the class exists.
        # We fix this by always emitting __init__.py LAST.
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_project(
            tmp_path,
            "blackjack",
            {
                "__init__.py": (
                    "from blackjack.dealer import BlackjackDealer as Dealer\n"
                    "from blackjack.game import BlackjackGame as Game\n"
                ),
                "dealer.py": "class BlackjackDealer:\n    pass\n",
                "game.py": "class BlackjackGame:\n    pass\n",
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="python")
        ds.download = lambda: data_dir
        cases = ds.load()
        combined = cases[0].code
        # The combined module must actually execute cleanly.
        ns: dict = {}
        exec(compile(combined, "<combined>", "exec"), ns)
        assert ns["Dealer"] is ns["BlackjackDealer"]
        assert ns["Game"] is ns["BlackjackGame"]

    def test_submodule_as_namespace_is_synthesised(self, tmp_path):
        # Regression: `from pkg import utils; @utils.check_for_none`
        # failed with `NameError: 'utils' is not defined`. Now we synth
        # a SimpleNamespace and back-fill it with the top-level names that
        # came from ``utils.py``.
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_project(
            tmp_path,
            "fuzzy",
            {
                "utils.py": (
                    "def check_for_none(fn):\n"
                    "    def wrapped(*a, **kw):\n"
                    "        if any(x is None for x in a): return 0\n"
                    "        return fn(*a, **kw)\n"
                    "    return wrapped\n"
                ),
                "fuzz.py": (
                    "from fuzzy import utils\n"
                    "@utils.check_for_none\n"
                    "def ratio(a, b):\n"
                    "    return 100\n"
                ),
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="python")
        ds.download = lambda: data_dir
        cases = ds.load()
        combined = cases[0].code
        assert "SimpleNamespace" in combined
        assert "utils.check_for_none = check_for_none" in combined
        # The whole thing must execute cleanly
        ns: dict = {}
        exec(compile(combined, "<combined>", "exec"), ns)
        assert ns["ratio"]("a", "b") == 100
        assert ns["ratio"](None, "b") == 0

    def test_python_repo_metadata_uses_importable_package_root(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_project(
            tmp_path,
            "mypkg",
            {
                "__init__.py": "from mypkg.core import add\n",
                "core.py": "def add(a, b):\n    return a + b\n",
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="python")
        ds.download = lambda: data_dir
        case = ds.load()[0]

        repo_root = tmp_path / "ProjectTest" / "dataset" / "Python"
        assert case.metadata["execution_context"] == "repo"
        assert case.metadata["project_root"] == str(repo_root.resolve())
        assert case.metadata["package_root"] == "mypkg"
        assert case.metadata["pythonpath_entries"] == ["mypkg"]
        assert set(case.metadata["local_import_roots"]) >= {"mypkg", "core"}
        assert case.metadata["sast_source_nonblocking"] is True
        assert case.metadata["target_file"] == "mypkg/core.py"
        assert case.metadata["import_module"] == "mypkg.core"
        assert case.metadata["target_function"] == "add"

    def test_python_repo_metadata_handles_flat_project_modules(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_project(
            tmp_path,
            "flatproj",
            {
                "flatproj.py": "from helper import inc\n\ndef add_one(value):\n    return inc(value)\n",
                "helper.py": "def inc(value):\n    return value + 1\n",
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="python")
        ds.download = lambda: data_dir
        case = ds.load()[0]

        repo_root = tmp_path / "ProjectTest" / "dataset" / "Python" / "flatproj"
        assert case.metadata["project_root"] == str(repo_root.resolve())
        assert case.metadata["pythonpath_entries"] == []
        assert set(case.metadata["local_import_roots"]) >= {"flatproj", "helper"}
        assert case.metadata["sast_source_nonblocking"] is True
        assert case.metadata["target_file"] == "flatproj.py"
        assert case.metadata["import_module"] == "flatproj"
        assert case.metadata["target_function"] == "add_one"

    def test_python_repo_metadata_target_comes_from_selected_module(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_project(
            tmp_path,
            "stock",
            {
                "validate.py": "class Validator:\n    pass\n",
                "stock.py": "class Stock:\n    pass\n",
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="python")
        ds.download = lambda: data_dir
        case = ds.load()[0]

        assert case.metadata["target_file"] == "stock.py"
        assert case.metadata["import_module"] == "stock"
        assert case.metadata["target_function"] == "Stock"
        assert "class Stock" in case.metadata["target_source_code"]
        assert "class Validator" not in case.metadata["target_source_code"]

    def test_javascript_projecttest_flattens_to_commonjs(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_js_project(
            tmp_path,
            "widgets",
            {
                "main.js": (
                    "import { helper } from './helper.js'\n"
                    "export const useHelper = () => helper() + 1\n"
                ),
                "helper.js": "export const helper = () => 2\n",
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="javascript")
        ds.download = lambda: data_dir
        case = ds.load()[0]
        combined = case.code

        assert combined.startswith("// --- helper.js ---")
        assert "# ---" not in combined
        assert "import { helper }" not in combined
        assert "export const" not in combined
        assert "var helper = () => 2" in combined
        assert "var useHelper = () => helper() + 1" in combined
        assert "module.exports" in combined
        assert "helper" in combined
        assert "useHelper" in combined
        assert case.metadata["target_function"] == "helper"

    def test_javascript_comments_do_not_select_reserved_word_targets(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_js_project(
            tmp_path,
            "check",
            {
                "check.js": (
                    "// This helper is used with custom inheritance checks.\n"
                    "/* Do not treat prose like class with as a target. */\n"
                    "export const checkCustom = (value) => value === true\n"
                    "export const isSubclass = (value) => value && value.parent\n"
                ),
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="javascript")
        ds.download = lambda: data_dir
        case = ds.load()[0]

        assert case.metadata["target_function"] == "checkCustom"
        assert "with" not in case.code.rsplit("module.exports", 1)[1]

    def test_javascript_commonjs_requires_are_flattened_without_redeclaration(
        self,
        tmp_path,
    ):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_js_project(
            tmp_path,
            "circle",
            {
                "Shape.js": (
                    "function Shape(type) { this.type = type }\n" "module.exports = Shape\n"
                ),
                "Utils.js": (
                    "var Utils = { shallowClone: function shallowClone(obj) { return obj } }\n"
                    "module.exports = Utils\n"
                ),
                "vec2.js": "var vec2 = { create: function create() { return [0, 0] } }\n",
                "Circle.js": (
                    "var Shape = require('./Shape')\n"
                    ",    vec2 = require('./vec2')\n"
                    ",    shallowClone = require('./Utils').shallowClone;\n\n"
                    "function Circle(radius) {\n"
                    "  Shape.call(this, 'circle')\n"
                    "  this.radius = shallowClone({ radius: radius }).radius\n"
                    "}\n"
                    "module.exports = Circle\n"
                ),
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="javascript")
        ds.download = lambda: data_dir
        case = ds.load()[0]
        combined = case.code

        assert "require('./" not in combined
        assert "var Shape = require" not in combined
        assert "var vec2 = require" not in combined
        assert "var shallowClone = Utils.shallowClone;" in combined
        assert case.metadata["target_function"] == "Circle"

    def test_javascript_template_export_text_does_not_create_dollar_export(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_js_project(
            tmp_path,
            "validate",
            {
                "validate.js": (
                    "const message = `Use export const ${name} = Parent.subclass(name)`\n"
                    "export const validateSubclass = (value) => Boolean(value)\n"
                    "export const classesData = []\n"
                ),
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="javascript")
        ds.download = lambda: data_dir
        case = ds.load()[0]
        exports = case.code.rsplit("module.exports", 1)[1]

        assert case.metadata["target_function"] == "validateSubclass"
        assert "\n  $," not in exports
        assert "validateSubclass" in exports

    def test_javascript_duplicate_esm_helpers_parse_after_flattening(self, tmp_path):
        from src.evaluation.benchmarks.projecttest import ProjectTestDataset

        data_dir = self._build_fake_js_project(
            tmp_path,
            "synergy",
            {
                "helpers.js": (
                    "export const pascalToKebab = (value) => value.toLowerCase()\n"
                    "export const applyAttribute = (node, name) => pascalToKebab(name)\n"
                ),
                "attribute.js": (
                    "import { applyAttribute } from './helpers.js'\n"
                    "const pascalToKebab = (value) => value.replace('A', '-a')\n"
                    "export const attributeToProp = (name) => pascalToKebab(name)\n"
                ),
            },
        )
        ds = ProjectTestDataset(data_dir=data_dir, language_filter="javascript")
        ds.download = lambda: data_dir
        case = ds.load()[0]
        combined = case.code

        assert "const pascalToKebab" not in combined
        assert "var pascalToKebab" in combined
        assert "attributeToProp" in combined


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
        assert cfg.quality_mode == "fast"
        assert cfg.mutation_max_mutants == 25
        assert cfg.reliability_repeats == 0

    def test_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "EVAL_DATA_DIR": "/custom/data",
                "EVAL_RESULTS_DIR": "/custom/results",
                "EVAL_MAX_CASES": "50",
                "EVAL_QUALITY_MODE": "full",
                "EVAL_MUTATION_MAX_MUTANTS": "12",
                "EVAL_RELIABILITY_REPEATS": "3",
            },
        ):
            cfg = EvalConfig.from_env()
            assert cfg.data_dir == "/custom/data"
            assert cfg.results_dir == "/custom/results"
            assert cfg.max_cases == 50
            assert cfg.quality_mode == "full"
            assert cfg.mutation_max_mutants == 12
            assert cfg.reliability_repeats == 3

    def test_quality_config_normalization(self):
        cfg = EvalConfig(
            quality_mode="bad",
            mutation_max_mutants=-5,
            reliability_repeats=-1,
        )
        assert cfg.quality_mode == "fast"
        assert cfg.mutation_max_mutants == 0
        assert cfg.reliability_repeats == 0
