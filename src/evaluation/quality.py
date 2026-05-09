"""Research-quality metrics for generated test suites.

These helpers are deliberately side-effect free. The benchmark runner uses
them to attach quality signals to per-case JSON without changing whether the
pipeline accepts or rejects a case.
"""

from __future__ import annotations

import ast
import copy
from typing import Any


def infer_primary_target(source_code: str, metadata: dict | None = None) -> str | None:
    """Infer the function/class under test from benchmark metadata or source."""
    metadata = metadata or {}
    for key in ("target", "target_function", "func_name", "entry_point", "function_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    try:
        tree = ast.parse(source_code or "")
    except SyntaxError:
        return None

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
    return None


def _test_function_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            count += 1
    return count


def _is_unittest_assert_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr.startswith("assert")
    return False


def _is_pytest_raises_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "raises"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def _is_dummy_assert(node: ast.Assert) -> bool:
    test = node.test
    if isinstance(test, ast.Constant):
        return bool(test.value) is True
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        left = test.left
        right = test.comparators[0]
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            try:
                return bool(eval(compile(ast.Expression(test), "<metric>", "eval"))) is True
            except Exception:
                return False
    return False


def compute_oracle_metrics(test_code: str) -> dict[str, Any]:
    """Measure assertion/oracle strength using static Python AST signals."""
    if not test_code or not test_code.strip():
        return {
            "test_count": 0,
            "assertion_count": 0,
            "has_assertions": False,
            "assertions_per_test": None,
            "dummy_test_flag": True,
            "error": "empty test code",
        }

    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        return {
            "test_count": 0,
            "assertion_count": 0,
            "has_assertions": False,
            "assertions_per_test": None,
            "dummy_test_flag": True,
            "error": f"syntax error: {exc.msg}",
        }

    test_count = _test_function_count(tree)
    assert_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    call_asserts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (_is_unittest_assert_call(node) or _is_pytest_raises_call(node))
    ]
    assertion_count = len(assert_nodes) + len(call_asserts)
    dummy_asserts = sum(1 for node in assert_nodes if _is_dummy_assert(node))
    pass_only_tests = test_count > 0 and assertion_count == 0
    return {
        "test_count": test_count,
        "assertion_count": assertion_count,
        "has_assertions": assertion_count > 0,
        "assertions_per_test": assertion_count / test_count if test_count else None,
        "dummy_assertion_count": dummy_asserts,
        "dummy_test_flag": pass_only_tests
        or (assertion_count > 0 and dummy_asserts == assertion_count),
    }


def _imported_names_from_source_module(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "source_module"
        ):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _defines_name(tree: ast.AST, target: str) -> bool:
    target_lower = target.lower()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == target or node.name.lower() == target_lower:
                return True
    return False


def _calls_name(tree: ast.AST, target: str) -> bool:
    target_lower = target.lower()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id.lower() == target_lower:
                return True
            if isinstance(func, ast.Attribute) and func.attr.lower() == target_lower:
                return True
    return False


def compute_relevance_metrics(
    test_code: str,
    source_code: str,
    metadata: dict | None = None,
    passed: bool | None = None,
) -> dict[str, Any]:
    """Measure whether generated tests appear to target the requested code."""
    target = infer_primary_target(source_code, metadata)
    if not target:
        return {"eligible": False, "relevance_pass": None, "gaming_flag": False}
    try:
        tree = ast.parse(test_code or "")
    except SyntaxError as exc:
        return {
            "eligible": True,
            "target": target,
            "relevance_pass": False,
            "gaming_flag": bool(passed),
            "error": f"syntax error: {exc.msg}",
        }

    imported = _imported_names_from_source_module(tree)
    test_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    target_lower = target.lower()
    signals = {
        "imports_target": target in imported or target_lower in {n.lower() for n in imported},
        "calls_target": _calls_name(tree, target),
        "test_name_mentions_target": any(target_lower in name.lower() for name in test_names),
        "imports_source_module": bool(imported),
    }
    target_redefined = _defines_name(tree, target) and not signals["imports_target"]
    relevance_pass = any(signals.values()) and not target_redefined
    gaming_flag = bool(passed) and (not relevance_pass or target_redefined)
    return {
        "eligible": True,
        "target": target,
        "relevance_pass": relevance_pass,
        "gaming_flag": gaming_flag,
        "target_redefined": target_redefined,
        "target_signal_count": sum(1 for value in signals.values() if value),
        "signals": signals,
    }


def _gate_by_name(gate_results: list[dict], name: str) -> dict | None:
    return next((g for g in gate_results if g.get("gate_name") == name), None)


def compute_gate_quality_metrics(
    gate_results: list[dict],
    metadata: dict | None = None,
) -> dict[str, dict[str, Any]]:
    """Derive safety/dependency benchmark metrics from gate results and metadata."""
    metadata = metadata or {}
    sast_gate = _gate_by_name(gate_results, "sast")
    dep_gate = _gate_by_name(gate_results, "dependency")

    expected_vulnerable = bool(
        metadata.get("vuln") or metadata.get("cwe") or metadata.get("expected_cwe")
    )
    vulnerability_detected = bool(sast_gate and not sast_gate.get("passed", True))

    phantom_packages = metadata.get("phantom_packages") or []
    valid_packages = metadata.get("valid_packages") or []
    expected_phantom = bool(phantom_packages)
    expected_clean = bool(valid_packages) and not expected_phantom
    phantom_detected = bool(dep_gate and expected_phantom and not dep_gate.get("passed", True))
    clean_accepted = bool(dep_gate and expected_clean and dep_gate.get("passed", False))

    return {
        "safety_metrics": {
            "expected_vulnerable": expected_vulnerable,
            "vulnerability_detected": vulnerability_detected,
            "sast_passed": None if sast_gate is None else bool(sast_gate.get("passed")),
        },
        "dependency_metrics": {
            "expected_phantom": expected_phantom,
            "expected_clean": expected_clean,
            "phantom_detected": phantom_detected,
            "clean_accepted": clean_accepted,
            "dependency_passed": None if dep_gate is None else bool(dep_gate.get("passed")),
        },
    }


def _source_line_count(source_code: str) -> int:
    try:
        tree = ast.parse(source_code or "")
    except SyntaxError:
        return 0
    lines: set[int] = set()
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        if isinstance(lineno, int):
            lines.add(lineno)
    return len(lines)


def compute_coverage_metrics(
    coverage_data: dict | None,
    source_code: str,
    metadata: dict | None = None,
    line_coverage: float | None = None,
) -> dict[str, Any]:
    """Extract line, branch, and target coverage metrics from coverage.py JSON."""
    out: dict[str, Any] = {"line_coverage": line_coverage}
    if not coverage_data:
        return out

    totals = coverage_data.get("totals") if isinstance(coverage_data, dict) else None
    if isinstance(totals, dict):
        if isinstance(totals.get("percent_covered"), (int, float)):
            out["line_coverage"] = float(totals["percent_covered"])
        branch = totals.get("percent_covered_branches") or totals.get("percent_branches")
        if isinstance(branch, (int, float)):
            out["branch_coverage"] = float(branch)

    files = coverage_data.get("files") if isinstance(coverage_data, dict) else None
    source_entry = files.get("source_module.py") if isinstance(files, dict) else None
    if isinstance(source_entry, dict):
        summary = source_entry.get("summary")
        if isinstance(summary, dict):
            if isinstance(summary.get("percent_covered"), (int, float)):
                out["target_line_coverage"] = float(summary["percent_covered"])
            branch = summary.get("percent_covered_branches") or summary.get("percent_branches")
            if isinstance(branch, (int, float)):
                out["target_branch_coverage"] = float(branch)

    if "target_line_coverage" not in out and out.get("line_coverage") is not None:
        out["target_line_coverage"] = out["line_coverage"]
    if "target_branch_coverage" not in out and out.get("branch_coverage") is not None:
        out["target_branch_coverage"] = out["branch_coverage"]
    out["target"] = infer_primary_target(source_code, metadata)
    out["source_line_count"] = _source_line_count(source_code)
    return out


class _Mutator(ast.NodeTransformer):
    def __init__(self, target_index: int) -> None:
        self.target_index = target_index
        self.current_index = -1
        self.mutation_type: str | None = None
        self.lineno: int | None = None

    def _hit(self, node: ast.AST, mutation_type: str) -> bool:
        self.current_index += 1
        if self.current_index == self.target_index:
            self.mutation_type = mutation_type
            self.lineno = getattr(node, "lineno", None)
            return True
        return False

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        replacements = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.FloorDiv,
            ast.FloorDiv: ast.Mult,
            ast.Div: ast.Mult,
        }
        for src, dst in replacements.items():
            if isinstance(node.op, src) and self._hit(node, f"{src.__name__}->{dst.__name__}"):
                node.op = dst()
                return node
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        replacements = {
            ast.Gt: ast.GtE,
            ast.GtE: ast.Gt,
            ast.Lt: ast.LtE,
            ast.LtE: ast.Lt,
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
        }
        for src, dst in replacements.items():
            if isinstance(node.ops[0], src) and self._hit(node, f"{src.__name__}->{dst.__name__}"):
                node.ops[0] = dst()
                return node
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool) and self._hit(node, "bool_flip"):
            return ast.copy_location(ast.Constant(value=not node.value), node)
        if (
            isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and self._hit(node, "int_increment")
        ):
            return ast.copy_location(ast.Constant(value=node.value + 1), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._hit(node, "remove_not"):
            return node.operand
        return node


def generate_python_mutants(source_code: str, limit: int = 25) -> list[dict[str, Any]]:
    """Generate deterministic simple AST mutants for Python source."""
    try:
        tree = ast.parse(source_code or "")
    except SyntaxError:
        return []

    mutants: list[dict[str, Any]] = []
    target_index = 0
    max_mutants = max(0, int(limit))
    while len(mutants) < max_mutants:
        clone = copy.deepcopy(tree)
        mutator = _Mutator(target_index)
        mutated = mutator.visit(clone)
        if mutator.mutation_type is None:
            break
        ast.fix_missing_locations(mutated)
        mutants.append(
            {
                "id": f"mut-{target_index}",
                "mutation_type": mutator.mutation_type,
                "lineno": mutator.lineno,
                "source_code": ast.unparse(mutated),
            }
        )
        target_index += 1
    return mutants


def compute_mutation_summary(
    mutants: list[dict[str, Any]],
    outcomes: list[bool] | None = None,
) -> dict[str, Any]:
    """Summarize mutation outcomes; True means the mutant was killed."""
    total = len(mutants)
    if outcomes is None:
        return {
            "mutants_total": total,
            "mutants_sampled": total,
            "mutants_killed": 0,
            "mutants_survived": 0,
            "mutants_uncovered": total,
            "mutation_score": None,
            "mutation_coverage": 0.0 if total else None,
        }

    observed = outcomes[:total]
    killed = sum(1 for item in observed if item is True)
    survived = sum(1 for item in observed if item is False)
    uncovered = max(0, total - killed - survived)
    executable = killed + survived
    return {
        "mutants_total": total,
        "mutants_sampled": total,
        "mutants_killed": killed,
        "mutants_survived": survived,
        "mutants_uncovered": uncovered,
        "mutation_score": killed / executable if executable else None,
        "mutation_coverage": executable / total if total else None,
    }
