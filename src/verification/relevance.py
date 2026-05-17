"""Gate: target relevance validation for generated tests.

The gate is intentionally split into two checks:

* a static pre-sandbox check that catches obvious gaming cheaply; and
* a dynamic post-sandbox check that confirms the target lines executed when
  coverage.py JSON is available.

This keeps the Generate-Detect-Repair loop fast while avoiding the old failure
mode where a test name or a dummy import was enough to pass relevance.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePath
from typing import Any, Iterable, Optional

from .models import Finding, GateResult, Severity

_DEFAULT_SOURCE_MODULE = "source_module"


@dataclass(frozen=True)
class TargetInfo:
    """Inferred target definition in the source-under-test."""

    name: str
    kind: str = "unknown"
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    executable_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class RelevanceAnalysis:
    """Static relevance evidence extracted from generated test code."""

    eligible: bool
    passed: bool
    target: Optional[str]
    targets: tuple[str, ...]
    score: int
    strong_signal_count: int
    signals: dict[str, bool]
    negative_signals: dict[str, bool]
    test_count: int
    assertion_count: int
    dummy_assertion_count: int
    details: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _camel_split(name: str) -> list[str]:
    """Split CamelCase / snake_case into lowercased word tokens of len>=3."""
    tokens: set[str] = set()
    for tok in re.split(r"[_\W]+", name):
        if len(tok) >= 3:
            tokens.add(tok.lower())
    for tok in re.findall(r"[A-Z][a-z]+|[a-z]+", name):
        if len(tok) >= 3:
            tokens.add(tok.lower())
    return list(tokens)


def _candidate_keywords(target: str) -> list[str]:
    base = {target, target.lower()}
    base.update(_camel_split(target))
    return [k for k in base if k]


def _node_span(node: ast.AST) -> tuple[Optional[int], Optional[int]]:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if end is None:
        lines = [
            lineno
            for child in ast.walk(node)
            if isinstance((lineno := getattr(child, "lineno", None)), int)
        ]
        end = max(lines) if lines else start
    return start, end


def _node_executable_lines(node: ast.AST) -> tuple[int, ...]:
    lines = {
        lineno
        for child in ast.walk(node)
        if isinstance((lineno := getattr(child, "lineno", None)), int)
    }
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lineno = getattr(child, "lineno", None)
            if isinstance(lineno, int):
                lines.discard(lineno)
            for decorator in getattr(child, "decorator_list", []):
                decorator_line = getattr(decorator, "lineno", None)
                if isinstance(decorator_line, int):
                    lines.discard(decorator_line)
    return tuple(sorted(lines))


def _target_from_node(node: ast.AST) -> Optional[TargetInfo]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    start, end = _node_span(node)
    return TargetInfo(
        name=node.name,
        kind="class" if isinstance(node, ast.ClassDef) else "function",
        start_line=start,
        end_line=end,
        executable_lines=_node_executable_lines(node),
    )


def infer_target_infos(
    source_code: str,
    target_function: Optional[str] = None,
) -> list[TargetInfo]:
    """Infer source targets from an explicit name or the first top-level definition."""
    target_name = target_function.strip() if isinstance(target_function, str) else None
    try:
        tree = ast.parse(source_code or "")
    except SyntaxError:
        return [TargetInfo(name=target_name)] if target_name else []

    top_level = [
        info for node in ast.iter_child_nodes(tree) if (info := _target_from_node(node)) is not None
    ]
    if target_name:
        lowered = target_name.lower()
        matches = [info for info in top_level if info.name.lower() == lowered]
        return matches or [TargetInfo(name=target_name)]
    return top_level[:1]


def _imports_source_module(tree: ast.AST, source_module: str) -> tuple[bool, set[str]]:
    aliases: set[str] = set()
    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == source_module:
                    imported = True
                    aliases.add(alias.asname or source_module)
                elif alias.name.startswith(f"{source_module}."):
                    imported = True
                    aliases.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == source_module or mod.startswith(f"{source_module}."):
                imported = True
            for alias in node.names:
                if alias.name == source_module:
                    imported = True
                    aliases.add(alias.asname or source_module)
    return imported, aliases


def _imported_target_aliases(
    tree: ast.AST,
    source_module: str,
    targets: Iterable[str],
) -> set[str]:
    """Names brought in via ``from source_module import target``."""
    target_lowers = {target.lower() for target in targets}
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        mod = node.module or ""
        if mod != source_module and not mod.startswith(f"{source_module}."):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name.lower() in target_lowers:
                aliases.add(alias.asname or alias.name)
    return aliases


def _has_wildcard_source_import(tree: ast.AST, source_module: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if (node.module or "") != source_module:
            continue
        if any(alias.name == "*" for alias in node.names):
            return True
    return False


def _test_function_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                names.append(node.name)
    return names


def _redefined_target_names(tree: ast.AST, targets: Iterable[str]) -> bool:
    target_lowers = {target.lower() for target in targets}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.lower() in target_lowers:
                return True
    return False


def _is_unittest_assert_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr.startswith("assert")


def _is_pytest_raises_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "raises"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
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
                return bool(eval(compile(ast.Expression(test), "<relevance>", "eval"))) is True
            except Exception:
                return False
    return False


def _assertion_counts(tree: ast.AST) -> tuple[int, int]:
    assert_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    call_asserts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (_is_unittest_assert_call(node) or _is_pytest_raises_call(node))
    ]
    return len(assert_nodes) + len(call_asserts), sum(
        1 for node in assert_nodes if _is_dummy_assert(node)
    )


def _has_target_call_in_expr(
    expr: ast.expr,
    module_aliases: set[str],
    imported_aliases: set[str],
    targets: Iterable[str],
) -> bool:
    """Return True if *expr* contains a call to a target (direct or via module)."""
    target_lowers = {t.lower() for t in targets}
    imported_lowers = {a.lower() for a in imported_aliases}
    module_lowers = {a.lower() for a in module_aliases}
    for child in ast.walk(expr):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                if func.id.lower() in (imported_lowers | target_lowers):
                    return True
            elif isinstance(func, ast.Attribute):
                if func.attr.lower() in target_lowers:
                    return True
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id.lower() in module_lowers
                    and func.attr.lower() in target_lowers
                ):
                    return True
    return False


def _expr_references_target_or_module(
    expr: ast.expr,
    module_aliases: set[str],
    imported_aliases: set[str],
    targets: Iterable[str],
) -> bool:
    """Return True if *expr* references a module alias, imported alias, or target name."""
    target_lowers = {t.lower() for t in targets}
    imported_lowers = {a.lower() for a in imported_aliases}
    module_lowers = {a.lower() for a in module_aliases}
    ref_names = module_lowers | imported_lowers | target_lowers
    for child in ast.walk(expr):
        if isinstance(child, ast.Name):
            if child.id.lower() in ref_names:
                return True
        elif isinstance(child, ast.Attribute):
            if child.attr.lower() in target_lowers:
                return True
            if isinstance(child.value, ast.Name) and child.value.id.lower() in module_lowers:
                return True
    return False


def _generic_public_api_assertion_counts(
    tree: ast.AST,
    module_aliases: set[str],
    imported_aliases: set[str],
    targets: Iterable[str],
) -> int:
    """Count assertions that reference the target/module without calling it."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            if _has_target_call_in_expr(node.test, module_aliases, imported_aliases, targets):
                continue
            if _expr_references_target_or_module(
                node.test, module_aliases, imported_aliases, targets
            ):
                count += 1
        elif isinstance(node, ast.Call) and (
            _is_unittest_assert_call(node) or _is_pytest_raises_call(node)
        ):
            args = node.args + [kw.value for kw in getattr(node, "keywords", [])]
            if any(
                _has_target_call_in_expr(arg, module_aliases, imported_aliases, targets)
                for arg in args
            ):
                continue
            if any(
                _expr_references_target_or_module(arg, module_aliases, imported_aliases, targets)
                for arg in args
            ):
                count += 1
    return count


def _call_target_signals(
    tree: ast.AST,
    targets: Iterable[str],
    imported_aliases: set[str],
    module_aliases: set[str],
    wildcard_imported: bool,
) -> tuple[bool, bool, bool]:
    target_lowers = {target.lower() for target in targets}
    imported_lowers = {alias.lower() for alias in imported_aliases}
    direct_call = False
    module_call = False
    instantiated_target = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id.lower()
            if name in imported_lowers or (wildcard_imported and name in target_lowers):
                direct_call = True
                instantiated_target = True
            elif name in target_lowers:
                direct_call = True
                instantiated_target = True
        elif isinstance(func, ast.Attribute):
            value = func.value
            # Detect target calls through module aliases: sm.Target(), sm.func().
            if isinstance(value, ast.Name) and value.id in module_aliases:
                if func.attr.lower() in target_lowers:
                    module_call = True
                    instantiated_target = True

    return direct_call, module_call, instantiated_target


def _asserts_target_or_source(
    tree: ast.AST,
    targets: Iterable[str],
    imported_aliases: set[str],
    module_aliases: set[str],
) -> bool:
    """Check whether any assert expression references the target or source module alias."""
    target_lowers = {target.lower() for target in targets}
    imported_lowers = {alias.lower() for alias in imported_aliases}
    module_alias_set = {alias for alias in module_aliases}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for child in ast.walk(node.test):
            if isinstance(child, ast.Name):
                name_lower = child.id.lower()
                if name_lower in imported_lowers or name_lower in target_lowers:
                    return True
            elif isinstance(child, ast.Attribute):
                if (
                    isinstance(child.value, ast.Name)
                    and child.value.id in module_alias_set
                    and child.attr.lower() in target_lowers
                ):
                    return True
                if child.attr.lower() in target_lowers:
                    return True
    return False


def analyze_relevance(
    test_code: str,
    source_code: Optional[str] = None,
    target_function: Optional[str] = None,
    *,
    source_module: str = _DEFAULT_SOURCE_MODULE,
    min_relevance_signals: int = 1,
    gate_policy: str = "balanced",
) -> RelevanceAnalysis:
    """Analyze static test relevance using balanced anti-gaming evidence."""
    if not test_code or not test_code.strip():
        return RelevanceAnalysis(
            eligible=True,
            passed=False,
            target=target_function,
            targets=tuple([target_function] if target_function else []),
            score=0,
            strong_signal_count=0,
            signals={},
            negative_signals={"empty_tests": True},
            test_count=0,
            assertion_count=0,
            dummy_assertion_count=0,
            details="empty test_code",
            error="empty_tests",
        )

    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        return RelevanceAnalysis(
            eligible=True,
            passed=False,
            target=target_function,
            targets=tuple([target_function] if target_function else []),
            score=0,
            strong_signal_count=0,
            signals={},
            negative_signals={"syntax_error": True},
            test_count=0,
            assertion_count=0,
            dummy_assertion_count=0,
            details=str(exc),
            error="test_syntax_error",
        )

    targets = tuple(info.name for info in infer_target_infos(source_code or "", target_function))
    test_names = _test_function_names(tree)
    assertion_count, dummy_assertion_count = _assertion_counts(tree)
    imports_source, module_aliases = _imports_source_module(tree, source_module)
    wildcard_imported = _has_wildcard_source_import(tree, source_module)
    imported_aliases = _imported_target_aliases(tree, source_module, targets)
    asserts_target_or_source = _asserts_target_or_source(
        tree,
        targets,
        imported_aliases,
        module_aliases,
    )

    keywords: list[str] = []
    for target in targets:
        keywords.extend(_candidate_keywords(target))
    keywords = list(dict.fromkeys(keywords))
    test_name_references_target = bool(keywords) and any(
        any(keyword in name.lower() for keyword in keywords) for name in test_names
    )

    direct_call, module_call, instantiated_target = _call_target_signals(
        tree,
        targets,
        imported_aliases,
        module_aliases,
        wildcard_imported,
    )
    target_redefined = bool(targets) and _redefined_target_names(tree, targets)

    signals = {
        "imports_source_module": imports_source,
        "imports_target_name": bool(imported_aliases),
        "wildcard_source_import": wildcard_imported,
        "test_name_references_target": test_name_references_target,
        "calls_target_directly": direct_call,
        "calls_target_via_module": module_call,
        "instantiates_target": instantiated_target,
        "asserts_target_or_source": asserts_target_or_source,
    }
    generic_public_api_assertion_count = _generic_public_api_assertion_counts(
        tree, module_aliases, imported_aliases, targets
    )
    has_call_or_instantiation = direct_call or module_call or instantiated_target

    negative_signals = {
        "no_test_functions": not test_names,
        "no_assertions": assertion_count == 0,
        "dummy_assertions_only": assertion_count > 0 and dummy_assertion_count == assertion_count,
        "target_redefined": target_redefined and not bool(imported_aliases),
        "generic_public_api_gaming": (
            bool(targets)
            and not has_call_or_instantiation
            and generic_public_api_assertion_count > 0
        ),
    }

    strong_signal_count = sum(
        1
        for key in (
            "calls_target_directly",
            "calls_target_via_module",
            "instantiates_target",
            "asserts_target_or_source",
        )
        if signals[key]
    )
    score = (
        (2 if signals["calls_target_directly"] else 0)
        + (2 if signals["calls_target_via_module"] else 0)
        + (2 if signals["instantiates_target"] else 0)
        + (1 if signals["imports_target_name"] else 0)
        + (1 if signals["wildcard_source_import"] else 0)
        + (1 if signals["imports_source_module"] else 0)
        + (1 if signals["test_name_references_target"] else 0)
        + (1 if signals["asserts_target_or_source"] else 0)
    )

    if not targets:
        static_passed = imports_source and bool(test_names)
    else:
        # Balanced mode: accept wildcard source imports + real assertions
        # as sufficient only when the assertion/call references the target.
        balanced_wildcard_pass = (
            gate_policy == "balanced"
            and wildcard_imported
            and has_call_or_instantiation
            and not negative_signals.get("generic_public_api_gaming", False)
            and not negative_signals.get("no_assertions", True)
            and not negative_signals.get("dummy_assertions_only", True)
            and not negative_signals.get("target_redefined", True)
            and not negative_signals.get("no_test_functions", True)
            and assertion_count > 0
            and bool(test_names)
        )
        static_passed = (
            score >= max(1, int(min_relevance_signals))
            and strong_signal_count > 0
            and has_call_or_instantiation
            and not any(negative_signals.values())
        ) or balanced_wildcard_pass

    details = (
        f"score={score}; strong={strong_signal_count}; signals={json.dumps(signals, sort_keys=True)}; "
        f"negative={json.dumps(negative_signals, sort_keys=True)}; targets={list(targets)}"
    )
    return RelevanceAnalysis(
        eligible=True,
        passed=static_passed,
        target=targets[0] if targets else None,
        targets=targets,
        score=score,
        strong_signal_count=strong_signal_count,
        signals=signals,
        negative_signals=negative_signals,
        test_count=len(test_names),
        assertion_count=assertion_count,
        dummy_assertion_count=dummy_assertion_count,
        details=details,
    )


def _coverage_file_entry(
    coverage_data: dict | None,
    source_module: str,
    source_file_path: str | None = None,
) -> Optional[dict]:
    if not isinstance(coverage_data, dict):
        return None
    files = coverage_data.get("files")
    if not isinstance(files, dict):
        return None
    # Build candidate names to match in coverage data
    candidates = {f"{source_module}.py"}
    if source_file_path:
        candidates.add(PurePath(str(source_file_path).replace("\\", "/")).name)
        mod_from_path = source_file_path.replace("/", ".").replace("\\", ".").removesuffix(".py")
        candidates.add(f"{mod_from_path}.py")
        candidates.add(str(PurePath(str(source_file_path).replace("\\", "/"))))
    for path, entry in files.items():
        normalized_path = str(PurePath(str(path).replace("\\", "/")))
        normalized = PurePath(normalized_path)
        if (
            normalized.name in candidates
            or normalized_path in candidates
            or any(normalized_path.endswith(f"/{candidate}") for candidate in candidates)
        ):
            return entry if isinstance(entry, dict) else None
    # Fallback: match by stem of source_module if it contains dots
    if "." in source_module:
        stem = source_module.rsplit(".", 1)[-1]
        for path, entry in files.items():
            if PurePath(str(path).replace("\\", "/")).stem == stem:
                return entry if isinstance(entry, dict) else None
    return None


def compute_target_coverage(
    coverage_data: dict | None,
    source_code: str,
    target_function: Optional[str] = None,
    *,
    source_module: str = _DEFAULT_SOURCE_MODULE,
    source_file_path: str | None = None,
) -> dict[str, Any]:
    """Compute target-line and target-branch coverage from coverage.py JSON."""
    targets = infer_target_infos(source_code, target_function)
    target = targets[0] if targets else None
    out: dict[str, Any] = {
        "target": target.name if target else target_function,
        "target_kind": target.kind if target else None,
        "target_start_line": target.start_line if target else None,
        "target_end_line": target.end_line if target else None,
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "target_executed_line_count": 0,
        "target_executable_line_count": len(target.executable_lines) if target else 0,
    }
    if not target or not target.executable_lines:
        return out

    entry = _coverage_file_entry(coverage_data, source_module, source_file_path)
    if not entry:
        return out

    executed = {
        int(line)
        for line in entry.get("executed_lines", [])
        if isinstance(line, (int, float)) and not isinstance(line, bool)
    }
    target_lines = set(target.executable_lines)
    executed_target_lines = executed & target_lines
    out["target_executed_line_count"] = len(executed_target_lines)
    out["target_line_coverage"] = (
        len(executed_target_lines) / len(target_lines) * 100.0 if target_lines else None
    )

    executed_branches = entry.get("executed_branches") or []
    missing_branches = entry.get("missing_branches") or []
    target_executed_branches = {
        tuple(branch)
        for branch in executed_branches
        if isinstance(branch, list)
        and len(branch) == 2
        and isinstance(branch[0], int)
        and branch[0] in target_lines
    }
    target_missing_branches = {
        tuple(branch)
        for branch in missing_branches
        if isinstance(branch, list)
        and len(branch) == 2
        and isinstance(branch[0], int)
        and branch[0] in target_lines
    }
    target_branch_total = len(target_executed_branches | target_missing_branches)
    if target_branch_total:
        out["target_branch_coverage"] = len(target_executed_branches) / target_branch_total * 100.0
    return out


class RelevanceValidator:
    """Detect tests that don't actually exercise the function under test."""

    def __init__(
        self,
        source_module: str = _DEFAULT_SOURCE_MODULE,
        min_relevance_signals: int = 1,
        gate_policy: str = "balanced",
    ) -> None:
        self.source_module = source_module
        self.min_relevance_signals = max(1, int(min_relevance_signals))
        self.gate_policy = gate_policy if gate_policy in {"strict", "balanced"} else "balanced"

    def validate(
        self,
        test_code: str,
        target_function: Optional[str] = None,
        *,
        source_code: Optional[str] = None,
    ) -> GateResult:
        """Evaluate static relevance before sandbox execution."""
        analysis = analyze_relevance(
            test_code,
            source_code=source_code,
            target_function=target_function,
            source_module=self.source_module,
            min_relevance_signals=self.min_relevance_signals,
            gate_policy=self.gate_policy,
        )
        findings = self._analysis_findings(analysis)
        return GateResult(
            gate_name="relevance",
            passed=analysis.passed,
            findings=findings,
            details=analysis.details,
        )

    def validate_dynamic(
        self,
        test_code: str,
        source_code: str,
        coverage_data: dict | None,
        target_function: Optional[str] = None,
        source_file_path: str | None = None,
    ) -> GateResult:
        """Evaluate post-sandbox relevance using target coverage evidence."""
        analysis = analyze_relevance(
            test_code,
            source_code=source_code,
            target_function=target_function,
            source_module=self.source_module,
            min_relevance_signals=self.min_relevance_signals,
            gate_policy=self.gate_policy,
        )
        target_cov = compute_target_coverage(
            coverage_data,
            source_code,
            target_function=target_function,
            source_module=self.source_module,
            source_file_path=source_file_path,
        )
        findings = self._analysis_findings(analysis)
        target_line_coverage = target_cov.get("target_line_coverage")

        if target_cov.get("target") is None:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    code="target_unknown",
                    message="Could not infer a target definition for dynamic relevance.",
                )
            )
            passed = analysis.passed
        elif target_line_coverage is None:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    code="target_coverage_unavailable",
                    message="Coverage JSON did not contain executable target-line data.",
                )
            )
            passed = analysis.passed and analysis.strong_signal_count > 0
        elif float(target_line_coverage) <= 0.0:
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code="target_not_executed",
                    message=(
                        "Generated tests passed, but coverage shows zero executed "
                        "lines in the inferred target."
                    ),
                    suggestion="Call or instantiate the original target from source_module in an assertion.",
                )
            )
            passed = False
        else:
            # Dynamic coverage is stronger evidence than import shape;
            # positive target coverage passes even with static weak signals.
            passed = analysis.passed or (
                self.gate_policy == "balanced" and float(target_line_coverage) > 0.0
            )

        return GateResult(
            gate_name="target_relevance",
            passed=passed,
            findings=findings,
            details=f"{analysis.details}; target_coverage={json.dumps(target_cov, sort_keys=True)}",
        )

    def _analysis_findings(self, analysis: RelevanceAnalysis) -> list[Finding]:
        findings: list[Finding] = []
        negatives = analysis.negative_signals
        target = analysis.target
        if analysis.error == "empty_tests":
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code="empty_tests",
                    message="No test code was generated.",
                )
            )
        if analysis.error == "test_syntax_error":
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code="test_syntax_error",
                    message=f"Cannot parse test code: {analysis.details}",
                )
            )
        if negatives.get("no_test_functions"):
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code="no_test_functions",
                    message="No test_* functions or Test* methods were found.",
                    suggestion="Define at least one pytest-discoverable test function.",
                )
            )
        if negatives.get("no_assertions"):
            msg = (
                f"Generated tests for '{target}' do not contain assertions or pytest.raises checks."
                if target
                else "Generated tests do not contain assertions or pytest.raises checks."
            )
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code="no_assertions",
                    message=msg,
                    suggestion="Assert observable behavior of the original target.",
                )
            )
        if negatives.get("dummy_assertions_only"):
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code="dummy_assertions_only",
                    message="Generated tests only contain dummy assertions.",
                    suggestion="Replace dummy assertions with checks against source_module behavior.",
                )
            )
        if negatives.get("target_redefined"):
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code="target_shadowed_in_tests",
                    message=(
                        "The target appears to be redefined in the test file rather "
                        "than imported from source_module."
                    ),
                    suggestion="Import and exercise the original target from source_module.",
                )
            )
        if negatives.get("generic_public_api_gaming"):
            msg = (
                f"Generated tests for '{target}' appear to be generic public API smoke tests "
                f"(e.g., assert {self.source_module} is not None or dir({self.source_module}))."
                if target
                else "Generated tests appear to be generic public API smoke tests."
            )
            suggestion = (
                f"Call or instantiate '{target}' from '{self.source_module}' in an assertion."
                if target
                else f"Call or instantiate the target from '{self.source_module}' in an assertion."
            )
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code="generic_public_api_gaming",
                    message=msg,
                    suggestion=suggestion,
                )
            )
        if not analysis.passed and not findings:
            if target:
                findings.append(
                    Finding(
                        severity=Severity.ERROR,
                        code="tests_unrelated_to_source",
                        message=(
                            f"Tests import {self.source_module} but never call target {target}; "
                            f"rewrite to import {target} directly and assert on {target}(...)."
                        ),
                        suggestion=(
                            f"Import '{target}' from '{self.source_module}' and call it "
                            "inside an assertion."
                        ),
                    )
                )
            else:
                findings.append(
                    Finding(
                        severity=Severity.ERROR,
                        code="tests_unrelated_to_source",
                        message=(
                            "Generated tests do not provide strong evidence that they "
                            "exercise the original target."
                        ),
                        suggestion=(
                            f"Import the target from '{self.source_module}' and call it "
                            "inside an assertion."
                        ),
                    )
                )
        return findings
