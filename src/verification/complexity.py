"""AST-based heuristic validation of Big-O complexity claims."""

import ast
import re
from typing import List, Optional, Set, Tuple

from .models import GateResult, Finding, Severity


_COMPLEXITY_ORDER = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "O(2^n)", "O(n!)"]


def _normalize(expr: str) -> str:
    """Normalize a Big-O expression for comparison."""
    s = expr.strip().upper().replace(" ", "")
    s = s.replace("O(", "O(").replace("LOGN", "LOG N").replace("LOG(N)", "LOG N")
    s = re.sub(r'O\((\d+)\)', 'O(1)', s)
    return expr.strip()


def _rank(expr: str) -> int:
    """Return an ordinal rank for common Big-O expressions (higher = slower)."""
    normalized = expr.strip().lower().replace(" ", "")
    mapping = {
        "o(1)": 0,
        "o(logn)": 1, "o(log(n))": 1, "o(log n)": 1,
        "o(n)": 2,
        "o(nlogn)": 3, "o(nlog(n))": 3, "o(n log n)": 3, "o(nlog n)": 3,
        "o(n^2)": 4, "o(n**2)": 4,
        "o(n^3)": 5, "o(n**3)": 5,
        "o(2^n)": 6, "o(2**n)": 6,
        "o(n!)": 7,
    }
    return mapping.get(normalized, -1)


class _ComplexityVisitor(ast.NodeVisitor):
    """Walk an AST and collect structural signals relevant to complexity."""

    def __init__(self):
        self.max_loop_depth: int = 0
        self.has_recursion: bool = False
        self.has_sort_call: bool = False
        self.allocates_collections: bool = False

        self._current_loop_depth: int = 0
        self._function_names: Set[str] = set()
        self._inside_function: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._function_names.add(node.name)
        prev = self._inside_function
        self._inside_function = node.name
        self.generic_visit(node)
        self._inside_function = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_loop(self, node):
        self._current_loop_depth += 1
        if self._current_loop_depth > self.max_loop_depth:
            self.max_loop_depth = self._current_loop_depth
        self.generic_visit(node)
        self._current_loop_depth -= 1

    def visit_For(self, node):
        self._visit_loop(node)

    def visit_While(self, node):
        self._visit_loop(node)

    def visit_Call(self, node):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in ("sorted", "sort"):
            self.has_sort_call = True

        if self._inside_function and func_name == self._inside_function:
            self.has_recursion = True

        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.allocates_collections = True
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.allocates_collections = True
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.allocates_collections = True
        self.generic_visit(node)


def _infer_time_lower_bound(visitor: _ComplexityVisitor) -> Tuple[int, str]:
    """Return (rank, reason) for the inferred minimum time complexity."""
    if visitor.has_sort_call and visitor.max_loop_depth >= 1:
        return 3, "Code uses sort inside a loop → at least O(n log n)"
    if visitor.has_sort_call:
        return 3, "Code uses sorted()/sort() → at least O(n log n)"
    if visitor.max_loop_depth >= 3:
        return 5, f"Code has {visitor.max_loop_depth} nested loops → at least O(n^3)"
    if visitor.max_loop_depth == 2:
        return 4, f"Code has 2 nested loops → at least O(n^2)"
    if visitor.max_loop_depth == 1 or visitor.has_recursion:
        return 2, "Code has a loop or recursion → at least O(n)"
    return 0, "No loops or recursion detected"


class ComplexityValidator:
    """Validate Big-O claims using AST-based heuristics."""

    def validate(self, code: str, time_claim: str, space_claim: str) -> GateResult:
        """Compare claimed complexity against AST-inferred lower bounds."""
        findings: List[Finding] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return GateResult(
                gate_name="complexity",
                passed=True,
                findings=[],
                details=f"Could not parse code: {e}",
            )

        visitor = _ComplexityVisitor()
        visitor.visit(tree)

        inferred_rank, reason = _infer_time_lower_bound(visitor)
        claimed_rank = _rank(time_claim)

        if claimed_rank != -1 and inferred_rank > claimed_rank:
            findings.append(Finding(
                severity=Severity.ERROR,
                code="complexity_underestimate",
                message=(
                    f"Claimed time complexity {time_claim} appears too low. "
                    f"{reason}."
                ),
                suggestion=f"Consider revising to at least {_COMPLEXITY_ORDER[inferred_rank]}",
            ))

        if visitor.allocates_collections:
            space_rank = _rank(space_claim)
            if space_rank != -1 and space_rank < 2:
                findings.append(Finding(
                    severity=Severity.WARNING,
                    code="space_underestimate",
                    message=(
                        f"Claimed space complexity {space_claim} may be too low — "
                        f"code allocates dynamic collections."
                    ),
                    suggestion="Consider at least O(n) space",
                ))

        return GateResult(
            gate_name="complexity",
            passed=len([f for f in findings if f.severity == Severity.ERROR]) == 0,
            findings=findings,
            details=f"Loops depth={visitor.max_loop_depth}, sort={visitor.has_sort_call}, "
                    f"recursion={visitor.has_recursion}, collections={visitor.allocates_collections}",
        )

    def format_feedback(self, gate_result: GateResult) -> str:
        """Format complexity findings as structured feedback for repair."""
        if gate_result.passed:
            return ""
        lines = [f"AST analysis: {gate_result.details}"]
        for f in gate_result.findings:
            lines.append(f"  - {f.message}")
            if f.suggestion:
                lines.append(f"    Suggestion: {f.suggestion}")
        return "\n".join(lines)
