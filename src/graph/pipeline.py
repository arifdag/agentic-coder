"""LangGraph pipeline for the GDR (Generate-Detect-Repair) workflow.

Phase 2: Multi-gate verification with SAST, dependency checks, LLM judge,
sandboxed execution with coverage, and structured diagnostics.
Phase 4: Code explanation with LLM-as-judge and AST complexity validation.
Phase 5: Multi-language support (JS/TS via Jest).
"""

import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from ..agents.explanation import ExplanationAgent, ExplanationRepairContext
from ..agents.jest_test import JestTestAgent
from ..agents.router import Language, RouterAgent, TaskType
from ..agents.ui_test import UIRepairContext, UITestAgent
from ..agents.unit_test import RepairContext, UnitTestAgent
from ..config import Config, get_role_llm
from ..utils.logging import AuditLogger
from ..verification.complexity import ComplexityValidator
from ..verification.dependency import DependencyValidator, extract_imports, extract_js_imports
from ..verification.explanation_judge import ExplanationJudge
from ..verification.js_sandbox import JsSandboxExecutor
from ..verification.judge import SastJudge
from ..verification.models import Finding, GateResult, Severity, VerificationReport
from ..verification.relevance import RelevanceValidator
from ..verification.repo_context import RepoContextExecutor
from ..verification.sandbox import SandboxExecutor
from ..verification.sast import SastAnalyzer
from ..verification.ui_sandbox import UITestExecutor

JS_LANGUAGES = {Language.JAVASCRIPT.value, Language.TYPESCRIPT.value}


def _dependency_finding_packages(gate: GateResult) -> set[str]:
    """Extract package names from dependency-gate PHANTOM-PKG findings."""
    import re

    packages: set[str] = set()
    for finding in gate.findings:
        if finding.code != "PHANTOM-PKG":
            continue
        match = re.search(r"Package '([^']+)' not found", finding.message)
        if match:
            packages.add(match.group(1))
    return packages


def _import_names_for_language(code: str, language: str) -> set[str]:
    if language in JS_LANGUAGES:
        return extract_js_imports(code)
    return extract_imports(code)


def _repo_local_import_roots(metadata: dict) -> set[str]:
    """Collect import roots that belong to the repo under test."""
    roots: set[str] = set()
    for key in ("local_import_roots", "pythonpath_entries"):
        values = metadata.get(key) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            if isinstance(value, str) and value.strip():
                roots.add(Path(value.strip()).name)

    for key in ("package_root", "project_name", "import_module"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            roots.add(value.strip().replace("\\", "/").split("/")[0].split(".")[0])

    target_file = metadata.get("target_file") or metadata.get("code_file")
    if isinstance(target_file, str) and target_file.endswith(".py"):
        path = Path(target_file.replace("\\", "/"))
        roots.add(path.stem)
        if len(path.parts) > 1:
            roots.add(path.parts[0])

    scan_root_value = metadata.get("package_project_root") or metadata.get("project_root")
    if isinstance(scan_root_value, str) and scan_root_value.strip():
        scan_root = Path(scan_root_value)
        if scan_root.exists() and scan_root.is_dir():
            for src in scan_root.rglob("*.py"):
                rel_parts = set(src.relative_to(scan_root).parts[:-1])
                if "__pycache__" in rel_parts:
                    continue
                roots.add(src.stem)
                roots.update(part for part in rel_parts if part and part != "__pycache__")

    return {root for root in roots if root and root != "."}


def _nonblocking_source_sast_findings(findings: list[Finding]) -> list[Finding]:
    """Convert source-only SAST findings into informational diagnostics."""
    out: list[Finding] = []
    for finding in findings:
        message = f"[source non-blocking] {finding.message}"
        out.append(finding.model_copy(update={"severity": Severity.INFO, "message": message}))
    return out


def _sast_source_nonblocking(metadata: dict) -> bool:
    """Return True when source SAST is diagnostic-only for this benchmark case."""
    return bool(metadata.get("sast_source_nonblocking"))


def _pytest_test_names(code: str) -> List[str]:
    return re.findall(r"def\s+(test_\w+)\s*\(", code or "")


def _has_pytest_tests(code: str) -> bool:
    return bool(_pytest_test_names(code))


def _safe_test_name(name: str) -> str:
    safe = re.sub(r"\W+", "_", name).strip("_").lower()
    return safe or "target"


def _arg_value_expr(arg: ast.arg) -> str:
    name = arg.arg.lower()
    annotation = ""
    if arg.annotation is not None:
        try:
            annotation = ast.unparse(arg.annotation).lower()
        except Exception:  # noqa: BLE001 - best-effort fallback generation
            annotation = ""

    hint = f"{name} {annotation}"
    if any(token in hint for token in ("list", "tuple", "items", "values", "nums", "array")):
        return "[]"
    if any(token in hint for token in ("dict", "map", "mapping")):
        return "{}"
    if any(token in hint for token in ("str", "name", "text", "path", "key")):
        return "''"
    if any(token in hint for token in ("bool", "flag", "enabled", "is_", "has_")):
        return "False"
    if any(token in hint for token in ("float", "ratio", "percent")):
        return "0.0"
    return "0"


def _required_call_args(args: ast.arguments, *, skip_first: bool = False) -> List[str]:
    positional = list(args.posonlyargs) + list(args.args)
    if skip_first and positional:
        positional = positional[1:]

    required_positional = positional[: max(0, len(positional) - len(args.defaults))]
    call_args = [_arg_value_expr(arg) for arg in required_positional]
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is None:
            call_args.append(f"{arg.arg}={_arg_value_expr(arg)}")
    return call_args


def _target_call_shape(source_code: str, target_function: str) -> tuple[str, List[str]]:
    try:
        tree = ast.parse(source_code or "")
    except SyntaxError:
        return "function", []

    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == target_function
        ):
            return "function", _required_call_args(node.args)
        if isinstance(node, ast.ClassDef) and node.name == target_function:
            init = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__"
                ),
                None,
            )
            if init is None:
                return "class", []
            return "class", _required_call_args(init.args, skip_first=True)
    return "function", []


def _repair_fallback_test_code(
    source_code: str,
    target_function: Optional[str],
    import_module: Optional[str] = None,
) -> Optional[str]:
    if not target_function or not re.match(r"^[A-Za-z_]\w*$", target_function):
        return None

    kind, call_args = _target_call_shape(source_code, target_function)
    call = f"{target_function}({', '.join(call_args)})"
    test_name = _safe_test_name(target_function)
    import_path = import_module or "source_module"
    lines = [f"from {import_path} import {target_function}", ""]
    if kind == "class":
        lines.extend(
            [
                f"def test_{test_name}_repair_scaffold_instantiates_target():",
                f"    instance = {call}",
                f"    assert isinstance(instance, {target_function})",
            ]
        )
    else:
        lines.extend(
            [
                f"def test_{test_name}_repair_scaffold_calls_target():",
                f"    result = {call}",
                "    assert result is not None",
            ]
        )
    return "\n".join(lines) + "\n"


def _repair_mode_from_report(
    report: Optional[VerificationReport], error_type: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    codes = {error_type} if error_type else set()
    if report:
        for gate in report.gates:
            for finding in gate.findings:
                if finding.code:
                    codes.add(finding.code)

    if codes & {"no_tests_collected", "no_test_functions", "empty_tests"}:
        return (
            "no_tests_collected",
            "Discard the previous structure and emit top-level def test_* functions that import and call the target.",
        )
    if "generic_public_api_gaming" in codes:
        return (
            "generic_public_api_gaming",
            "Delete broad module API smoke checks and assert directly on calls to the requested target.",
        )
    if "target_not_executed" in codes:
        return (
            "target_line_coverage_zero",
            "Every test must directly call or instantiate the target so coverage executes target lines.",
        )
    if codes & {"assertion_error", "test_failure"}:
        return (
            "assertion_failure",
            "Preserve passing target-focused tests; remove or correct only failing assumptions and never blank the file.",
        )
    if "syntax_error" in codes:
        return (
            "syntax_error",
            "Rebuild a small syntactically valid test file; avoid multiline test names and huge generated suites.",
        )
    if "tests_unrelated_to_source" in codes:
        return (
            "tests_unrelated_to_source",
            "Rewrite tests to import the target directly and assert on target calls.",
        )
    return None, None


class AuditEntry(BaseModel):
    """Single audit log entry."""

    iteration: int
    timestamp: str
    generated_artifact: Optional[str] = None
    verification_report: Optional[dict] = None
    repair_context: Optional[dict] = None


class PipelineState(TypedDict):
    """State schema for the LangGraph pipeline."""

    code_input: str
    file_path: Optional[str]
    user_request: str

    # UI-test specific inputs
    target_url: Optional[str]
    html_content: Optional[str]
    description: Optional[str]
    target_function: Optional[str]

    routing_decision: Optional[dict]
    task_type: Optional[str]
    language: Optional[str]

    generated_tests: Optional[str]
    test_functions: Optional[List[str]]

    # Explanation-specific (Phase 4)
    generated_explanation: Optional[dict]

    # Multi-gate verification (Phase 2)
    gate_results: Optional[List[dict]]
    verification_report: Optional[dict]
    coverage_report: Optional[str]

    # Actual pytest pass/fail counts from the sandbox gate. Kept at the
    # top-level of the state so the evaluation runner can report real
    # partial-success numbers (e.g. "18/28 tests passed") instead of
    # collapsing multi-file projects to all-or-nothing. LangGraph will
    # drop keys not declared in the TypedDict, so they must live here.
    sandbox_tests_run: Optional[int]
    sandbox_tests_passed: Optional[int]
    sandbox_branch_coverage: Optional[float]
    sandbox_coverage_data: Optional[dict]
    repo_metadata: Optional[dict]
    execution_context: Optional[str]
    infrastructure_pass: Optional[bool]
    repo_setup_pass: Optional[bool]
    source_phantom_import: bool
    source_phantom_packages: List[str]
    test_phantom_packages: List[str]

    # Legacy single-gate fields kept for CLI compatibility
    verification_result: Optional[dict]
    verification_passed: bool
    error_type: Optional[str]
    error_message: Optional[str]

    retry_count: int
    max_retries: int
    audit_log: List[dict]
    final_output: Optional[str]
    status: str


def create_pipeline(config: Optional[Config] = None):
    """Create the LangGraph pipeline with multi-gate verification.

    Graph: router -> generate -> verify_static -> judge -> verify_sandbox
           -> aggregate -> (repair | output)
    """
    if config is None:
        config = Config.load()

    # Coding and judge may be different providers/models with independent
    # fallback chains. Both default to the same primary if the role-based
    # env vars aren't configured (legacy behavior).
    llm = get_role_llm(config, "coding")
    router_agent = RouterAgent()
    unit_test_agent = UnitTestAgent(llm)
    jest_test_agent = JestTestAgent(llm)
    ui_test_agent = UITestAgent(llm)
    explanation_agent = ExplanationAgent(llm)
    sandbox = SandboxExecutor(config.sandbox)
    js_sandbox = JsSandboxExecutor(config.js_sandbox)
    ui_sandbox = UITestExecutor(config.ui_test)
    audit_logger = AuditLogger(config.pipeline.audit_log_dir)

    sast_analyzer = (
        SastAnalyzer(
            semgrep_rules=config.sast.semgrep_rules,
            bandit_enabled=config.sast.bandit_enabled,
            timeout=config.sast.timeout,
        )
        if config.sast.enabled
        else None
    )

    dep_validator = (
        DependencyValidator(
            pypi_timeout=config.dependency.pypi_timeout,
        )
        if config.dependency.enabled
        else None
    )

    judge_llm = get_role_llm(config, "judge")

    sast_judge = SastJudge(judge_llm) if config.judge.enabled else None

    explanation_judge = ExplanationJudge(judge_llm) if config.explanation.judge_enabled else None
    complexity_validator = (
        ComplexityValidator() if config.explanation.complexity_check_enabled else None
    )

    # Anti-gaming relevance gate (opt-in via RELEVANCE_GATE_ENABLED).
    # Catches the failure mode where the LLM ignores the target function
    # and writes tests for fictional code -- discovered in the ULT
    # ablation, where ~33% of "passing" cases at k=5 had zero tests
    # referencing the target.
    relevance_validator = (
        RelevanceValidator(
            source_module=config.relevance.source_module,
            min_relevance_signals=config.relevance.min_signals,
            gate_policy=config.evaluation.gate_policy,
        )
        if config.relevance.enabled
        else None
    )

    # ── Nodes ──────────────────────────────────────────────────────────

    def router_node(state: PipelineState) -> PipelineState:
        routing = router_agent.route(
            code=state["code_input"],
            user_request=state["user_request"],
            file_path=state.get("file_path"),
        )
        return {
            **state,
            "routing_decision": routing.model_dump(),
            "task_type": routing.task_type.value,
            "language": state.get("language") or routing.language.value,
            "status": "routed",
        }

    def generate_node(state: PipelineState) -> PipelineState:
        task_type = state.get("task_type", TaskType.UNIT_TEST.value)

        if task_type == TaskType.EXPLANATION.value:
            expl = explanation_agent.generate(
                code=state["code_input"],
                file_path=state.get("file_path"),
            )
            expl_dict = expl.model_dump()
            expl_md = expl.to_markdown()

            audit_entry = AuditEntry(
                iteration=state["retry_count"] + 1,
                timestamp=datetime.now().isoformat(),
                generated_artifact=expl_md,
            )

            return {
                **state,
                "generated_tests": expl_md,
                "generated_explanation": expl_dict,
                "test_functions": [],
                "audit_log": state["audit_log"] + [audit_entry.model_dump()],
                "status": "generated",
            }

        if task_type == TaskType.UNIT_TEST.value:
            lang = state.get("language", Language.PYTHON.value)
            if lang in JS_LANGUAGES:
                result = jest_test_agent.generate(
                    code=state["code_input"],
                    file_path=state.get("file_path"),
                    target_function=state.get("target_function"),
                )
            else:
                repo_meta = state.get("repo_metadata") or {}
                result = unit_test_agent.generate(
                    code=state["code_input"],
                    file_path=state.get("file_path"),
                    import_module=repo_meta.get("import_module"),
                    target_function=state.get("target_function"),
                )
        elif task_type == TaskType.UI_TEST.value:
            ui_result = ui_test_agent.generate(
                description=state.get("description") or state.get("user_request", ""),
                target_url=state.get("target_url"),
                html_content=state.get("html_content"),
            )
            result = ui_result
        else:
            return {
                **state,
                "status": "error",
                "error_message": f"Task type '{task_type}' not yet implemented",
            }

        audit_entry = AuditEntry(
            iteration=state["retry_count"] + 1,
            timestamp=datetime.now().isoformat(),
            generated_artifact=result.test_code,
        )

        return {
            **state,
            "generated_tests": result.test_code,
            "test_functions": result.test_functions,
            "audit_log": state["audit_log"] + [audit_entry.model_dump()],
            "status": "generated",
        }

    def verify_static_node(state: PipelineState) -> PipelineState:
        """Run Gate 1 (SAST) and Gate 2 (dependency) concurrently.

        SAST is skipped for UI tests because Semgrep/Bandit rules target
        application code, not Playwright test scripts.
        Explanations bypass static analysis entirely.
        """
        task_type = state.get("task_type", TaskType.UNIT_TEST.value)

        if task_type == TaskType.EXPLANATION.value:
            return {
                **state,
                "gate_results": [],
                "verification_passed": True,
                "status": "static_checked",
            }

        if not state.get("generated_tests"):
            return {
                **state,
                "gate_results": [],
                "verification_passed": False,
                "error_type": "generation_error",
                "error_message": "No tests were generated",
                "status": "verification_failed",
            }

        is_ui = task_type == TaskType.UI_TEST.value
        lang = state.get("language", Language.PYTHON.value)
        repo_meta = state.get("repo_metadata") or {}

        test_code = state["generated_tests"]
        source_and_test = state["code_input"] + "\n\n" + test_code

        def run_sast():
            if is_ui:
                return GateResult(
                    gate_name="sast", passed=True, findings=[], details="Skipped for UI tests"
                )
            if sast_analyzer:
                if _sast_source_nonblocking(repo_meta):
                    test_result = sast_analyzer.analyze(test_code, language=lang)
                    source_result = sast_analyzer.analyze(state["code_input"], language=lang)
                    return GateResult(
                        gate_name="sast",
                        passed=test_result.passed,
                        findings=(
                            list(test_result.findings)
                            + _nonblocking_source_sast_findings(source_result.findings)
                        ),
                        details=(
                            "source_sast_nonblocking=true; "
                            f"test_sast_passed={test_result.passed}; "
                            f"source_sast_passed={source_result.passed}"
                        ),
                    )
                return sast_analyzer.analyze(source_and_test, language=lang)
            return GateResult(gate_name="sast", passed=True, findings=[])

        def run_deps():
            if not dep_validator:
                return GateResult(gate_name="dependency", passed=True, findings=[])
            # Validate BOTH the source code and the generated test code.
            # Previously only the test code was checked, which meant
            # phantom imports in the subject-under-test (the exact
            # thing the ``dep_hallucination`` benchmark exists to
            # surface) slipped through undetected.
            combined = (state["code_input"] or "") + "\n\n" + (test_code or "")
            return dep_validator.validate(
                combined,
                language=lang,
                extra_known=_repo_local_import_roots(repo_meta),
            )

        def run_relevance():
            if not relevance_validator or is_ui or lang in JS_LANGUAGES:
                return None
            # Tests for explanation tasks aren't subject to relevance.
            return relevance_validator.validate(
                test_code,
                target_function=state.get("target_function"),
                source_code=state["code_input"],
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            sast_future = pool.submit(run_sast)
            dep_future = pool.submit(run_deps)
            rel_future = pool.submit(run_relevance)
            sast_result = sast_future.result()
            dep_result = dep_future.result()
            rel_result = rel_future.result()

        detected_phantoms = _dependency_finding_packages(dep_result)
        source_phantoms = sorted(
            detected_phantoms & _import_names_for_language(state["code_input"] or "", lang)
        )
        test_phantoms = sorted(
            detected_phantoms & _import_names_for_language(test_code or "", lang)
        )
        if detected_phantoms:
            extra_details = (
                f"source_phantom_imports={source_phantoms}; "
                f"test_phantom_imports={test_phantoms}"
            )
            dep_result.details = (
                f"{dep_result.details}; {extra_details}" if dep_result.details else extra_details
            )

        gate_results = [sast_result.model_dump(), dep_result.model_dump()]
        static_passed = sast_result.passed and dep_result.passed
        if rel_result is not None:
            gate_results.append(rel_result.model_dump())
            static_passed = static_passed and rel_result.passed

        return {
            **state,
            "gate_results": gate_results,
            "verification_passed": static_passed,
            "source_phantom_import": bool(source_phantoms),
            "source_phantom_packages": source_phantoms,
            "test_phantom_packages": test_phantoms,
            "status": "static_checked",
        }

    def judge_node(state: PipelineState) -> PipelineState:
        """Run LLM judge: SAST triage for tests, rubric evaluation for explanations."""
        task_type = state.get("task_type", TaskType.UNIT_TEST.value)

        if task_type == TaskType.EXPLANATION.value:
            if not explanation_judge:
                return state
            expl_dict = state.get("generated_explanation")
            if not expl_dict:
                return state
            expl_json = json.dumps(expl_dict, indent=2)
            judge_gate = explanation_judge.verify(state["code_input"], expl_json)

            prior_gates = state.get("gate_results") or []
            all_gates = prior_gates + [judge_gate.model_dump()]

            return {
                **state,
                "gate_results": all_gates,
                "verification_passed": judge_gate.passed,
                "status": "judged",
            }

        if not sast_judge or not state.get("gate_results"):
            return state

        gate_results = [GateResult(**g) for g in state["gate_results"]]

        sast_gate = next((g for g in gate_results if g.gate_name == "sast"), None)
        if not sast_gate or sast_gate.passed:
            return state

        source_and_test = state["code_input"] + "\n\n" + (state.get("generated_tests") or "")
        judged_sast = sast_judge.triage(source_and_test, sast_gate)

        updated_gates = []
        for g in gate_results:
            if g.gate_name == "sast":
                updated_gates.append(judged_sast.model_dump())
            else:
                updated_gates.append(g.model_dump())

        all_passed = all(GateResult(**g).passed for g in updated_gates)

        return {
            **state,
            "gate_results": updated_gates,
            "verification_passed": all_passed,
            "status": "judged",
        }

    def verify_sandbox_node(state: PipelineState) -> PipelineState:
        """Run Gate 3: sandbox execution (unit), Playwright (UI), or complexity validation (explanation)."""
        task_type = state.get("task_type", TaskType.UNIT_TEST.value)

        if task_type == TaskType.EXPLANATION.value:
            if not complexity_validator:
                noop_gate = GateResult(
                    gate_name="complexity",
                    passed=True,
                    findings=[],
                    details="Complexity validation disabled",
                )
                prior_gates = state.get("gate_results") or []
                return {
                    **state,
                    "gate_results": prior_gates + [noop_gate.model_dump()],
                    "status": "sandbox_checked",
                }
            expl_dict = state.get("generated_explanation") or {}
            complexity = expl_dict.get("complexity", {})
            time_claim = complexity.get("time", "O(?)")
            space_claim = complexity.get("space", "O(?)")
            cx_gate = complexity_validator.validate(state["code_input"], time_claim, space_claim)

            prior_gates = state.get("gate_results") or []
            return {
                **state,
                "gate_results": prior_gates + [cx_gate.model_dump()],
                "status": "sandbox_checked",
            }

        test_code = state["generated_tests"] or ""
        lang = state.get("language", Language.PYTHON.value)
        execution_context = "single-file"

        if task_type == TaskType.UI_TEST.value:
            result = ui_sandbox.execute(
                test_code=test_code,
                target_url=state.get("target_url"),
                html_content=state.get("html_content"),
            )
        elif lang in JS_LANGUAGES:
            source_code = state["code_input"]
            result = js_sandbox.execute(
                source_code=source_code,
                test_code=test_code,
            )
        else:
            source_code = state["code_input"]
            repo_meta = state.get("repo_metadata") or {}
            has_repo_fields = bool(repo_meta.get("project_root") or repo_meta.get("repo"))
            ec = config.evaluation.execution_context if config else "auto"
            use_repo = (ec == "repo") or (ec == "auto" and has_repo_fields)
            if use_repo and has_repo_fields:
                execution_context = "repo"
                executor = RepoContextExecutor(
                    repo_setup=config.evaluation.repo_setup,
                    gate_policy=config.evaluation.gate_policy,
                    pytest_timeout=config.evaluation.repo_pytest_timeout,
                )
                result = executor.execute(
                    source_code=source_code,
                    test_code=test_code,
                    metadata=repo_meta,
                )
            else:
                result = sandbox.execute(
                    source_code=source_code,
                    test_code=test_code,
                )

        sandbox_gate = GateResult(
            gate_name="sandbox",
            passed=result.success,
            findings=_sandbox_to_findings(result),
            details=_truncate_sandbox_output(result.stdout, result.stderr),
        )

        prior_gates = state.get("gate_results") or []
        all_gates = prior_gates + [sandbox_gate.model_dump()]
        if relevance_validator and result.success and task_type == TaskType.UNIT_TEST.value:
            if lang not in JS_LANGUAGES:
                target_gate = relevance_validator.validate_dynamic(
                    test_code=test_code,
                    source_code=state["code_input"],
                    coverage_data=getattr(result, "coverage_data", None),
                    target_function=state.get("target_function"),
                    source_file_path=(
                        (state.get("repo_metadata") or {}).get("target_file")
                        or (state.get("repo_metadata") or {}).get("code_file")
                    ),
                )
                all_gates.append(target_gate.model_dump())

        infra_pass = getattr(result, "infrastructure_pass", None)
        setup_pass = getattr(result, "repo_setup_pass", None)
        if infra_pass is None:
            infra_pass = result.error_type != "infrastructure_error"

        return {
            **state,
            "gate_results": all_gates,
            "coverage_report": result.coverage_gaps,
            "sandbox_tests_run": getattr(result, "tests_run", 0),
            "sandbox_tests_passed": getattr(result, "tests_passed", 0),
            "sandbox_branch_coverage": getattr(result, "branch_coverage", None),
            "sandbox_coverage_data": getattr(result, "coverage_data", None),
            "execution_context": execution_context,
            "infrastructure_pass": infra_pass,
            "repo_setup_pass": setup_pass,
            "status": "sandbox_checked",
        }

    def aggregate_node(state: PipelineState) -> PipelineState:
        """Combine all gate results into a unified VerificationReport."""
        raw_gates = state.get("gate_results") or []
        gates = [GateResult(**g) for g in raw_gates]

        sandbox_gate = next((g for g in gates if g.gate_name == "sandbox"), None)
        coverage = None
        if sandbox_gate and sandbox_gate.details:
            import re

            m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", sandbox_gate.details)
            if m:
                coverage = float(m.group(1))

        report = VerificationReport.from_gates(
            gates=gates,
            coverage=coverage,
            coverage_gaps=state.get("coverage_report"),
        )

        overall = report.overall_passed

        error_type = None
        error_message = None
        if not overall:
            for g in gates:
                if not g.passed:
                    for f in g.error_findings:
                        error_type = f.code or g.gate_name
                        error_message = f.message
                        break
                    if error_type:
                        break
            if not error_type:
                error_type = "verification_failed"
                error_message = report.summary
            if state.get("source_phantom_import"):
                packages = ", ".join(state.get("source_phantom_packages") or [])
                error_type = "source_phantom_import"
                error_message = "Dependency gate detected phantom imports in the source code" + (
                    f": {packages}" if packages else ""
                )

        if state["audit_log"]:
            last = state["audit_log"][-1].copy()
            last["verification_report"] = report.model_dump()
            new_audit = state["audit_log"][:-1] + [last]
        else:
            new_audit = state["audit_log"]

        return {
            **state,
            "verification_report": report.model_dump(),
            "verification_result": report.model_dump(),
            "verification_passed": overall,
            "error_type": error_type,
            "error_message": error_message,
            "audit_log": new_audit,
            "status": "verified" if overall else "verification_failed",
        }

    def repair_node(state: PipelineState) -> PipelineState:
        task_type = state.get("task_type", TaskType.UNIT_TEST.value)

        report_data = state.get("verification_report")
        report = None
        diagnostics = None
        if report_data:
            report = VerificationReport(**report_data)
            diagnostics = report.format_for_repair()
        repair_mode, diagnostic_note = _repair_mode_from_report(
            report,
            state.get("error_type"),
        )

        if task_type == TaskType.EXPLANATION.value:
            expl_dict = state.get("generated_explanation") or {}
            expl_json = json.dumps(expl_dict, indent=2)

            judge_feedback = None
            complexity_feedback = None
            raw_gates = state.get("gate_results") or []
            for g_data in raw_gates:
                g = GateResult(**g_data)
                if g.gate_name == "explanation_judge" and not g.passed and explanation_judge:
                    judge_feedback = explanation_judge.format_feedback(g)
                if g.gate_name == "complexity" and not g.passed and complexity_validator:
                    complexity_feedback = complexity_validator.format_feedback(g)

            expl_ctx = ExplanationRepairContext(
                previous_explanation=expl_json,
                error_type=state.get("error_type") or "verification_failed",
                error_message=state.get("error_message") or "Explanation failed verification",
                judge_feedback=judge_feedback,
                complexity_feedback=complexity_feedback,
            )
            repaired = explanation_agent.repair(expl_ctx)
            repaired_dict = repaired.model_dump()
            repaired_md = repaired.to_markdown()

            new_retry = state["retry_count"] + 1
            audit_entry = AuditEntry(
                iteration=new_retry + 1,
                timestamp=datetime.now().isoformat(),
                generated_artifact=repaired_md,
                repair_context=expl_ctx.model_dump(),
            )

            return {
                **state,
                "generated_tests": repaired_md,
                "generated_explanation": repaired_dict,
                "test_functions": [],
                "retry_count": new_retry,
                "gate_results": None,
                "verification_report": None,
                "coverage_report": None,
                "audit_log": state["audit_log"] + [audit_entry.model_dump()],
                "status": "repaired",
            }

        lang = state.get("language", Language.PYTHON.value)

        if task_type == TaskType.UI_TEST.value:
            ctx = UIRepairContext(
                previous_code=state.get("generated_tests") or "",
                error_type=state.get("error_type") or "unknown",
                error_message=state.get("error_message") or "Unknown error",
                line_number=None,
                diagnostics=diagnostics,
            )
            result = ui_test_agent.repair(ctx)
        else:
            repo_meta = state.get("repo_metadata") or {}
            ctx = RepairContext(
                previous_code=state.get("generated_tests") or "",
                error_type=state.get("error_type") or "unknown",
                error_message=state.get("error_message") or "Unknown error",
                line_number=None,
                coverage_gaps=state.get("coverage_report"),
                diagnostics=diagnostics,
                import_module=repo_meta.get("import_module"),
                target_function=state.get("target_function"),
                repair_mode=repair_mode,
                diagnostic_note=diagnostic_note,
            )
            if lang in JS_LANGUAGES:
                result = jest_test_agent.repair(ctx)
            else:
                result = unit_test_agent.repair(ctx)

        result_code = result.test_code
        result_functions = result.test_functions
        fallback_used = False
        if (
            task_type == TaskType.UNIT_TEST.value
            and lang not in JS_LANGUAGES
            and not _has_pytest_tests(result_code)
        ):
            fallback_code = _repair_fallback_test_code(
                state.get("code_input") or "",
                state.get("target_function"),
                (state.get("repo_metadata") or {}).get("import_module"),
            )
            if fallback_code:
                result_code = fallback_code
                result_functions = _pytest_test_names(fallback_code)
                fallback_used = True

        new_retry = state["retry_count"] + 1
        repair_payload = ctx.model_dump()
        if fallback_used:
            repair_payload["fallback_used"] = True
            repair_payload["fallback_reason"] = "repair_returned_no_pytest_tests"
        audit_entry = AuditEntry(
            iteration=new_retry + 1,
            timestamp=datetime.now().isoformat(),
            generated_artifact=result_code,
            repair_context=repair_payload,
        )

        return {
            **state,
            "generated_tests": result_code,
            "test_functions": result_functions,
            "retry_count": new_retry,
            "gate_results": None,
            "verification_report": None,
            "coverage_report": None,
            "audit_log": state["audit_log"] + [audit_entry.model_dump()],
            "status": "repaired",
        }

    def output_node(state: PipelineState) -> PipelineState:
        task_type = state.get("task_type", TaskType.UNIT_TEST.value)

        if state["verification_passed"]:
            status = "success"
        else:
            status = "failed_after_retries"

        if task_type == TaskType.EXPLANATION.value:
            expl_dict = state.get("generated_explanation")
            if expl_dict:
                from ..agents.explanation import CodeExplanation

                final_output = CodeExplanation(**expl_dict).to_markdown()
            else:
                final_output = state.get("generated_tests")
        else:
            final_output = (
                state.get("generated_tests")
                if state["verification_passed"]
                else state.get("generated_tests")
            )

        provenance = {
            "coding": config.coding_role.provenance() if config.coding_role else None,
            "judge": (
                (config.judge_role or config.coding_role).provenance()
                if (config.judge_role or config.coding_role)
                else None
            ),
            "timestamp": datetime.now().isoformat(),
        }
        audit_logger.save(
            state["audit_log"],
            state.get("file_path") or "unknown",
            provenance=provenance,
        )

        return {
            **state,
            "final_output": final_output,
            "status": status,
        }

    def should_repair(state: PipelineState) -> Literal["repair", "output"]:
        if state["verification_passed"]:
            return "output"
        if state.get("error_type") == "infrastructure_error":
            return "output"
        if state.get("source_phantom_import"):
            return "output"
        if state["retry_count"] >= state["max_retries"]:
            return "output"
        return "repair"

    # ── Graph wiring ───────────────────────────────────────────────────

    workflow = StateGraph(PipelineState)

    workflow.add_node("router", router_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify_static", verify_static_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("verify_sandbox", verify_sandbox_node)
    workflow.add_node("aggregate", aggregate_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("output", output_node)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "generate")
    workflow.add_edge("generate", "verify_static")
    workflow.add_edge("verify_static", "judge")
    workflow.add_edge("judge", "verify_sandbox")
    workflow.add_edge("verify_sandbox", "aggregate")
    workflow.add_conditional_edges(
        "aggregate",
        should_repair,
        {"repair": "repair", "output": "output"},
    )
    workflow.add_edge("repair", "verify_static")
    workflow.add_edge("output", END)

    return workflow.compile()


# ── Helpers ────────────────────────────────────────────────────────────


def _truncate_sandbox_output(
    stdout: Optional[str], stderr: Optional[str], head: int = 1500, tail: int = 6500
) -> Optional[str]:
    """Return a truncated view of pytest output that preserves the end.

    Pytest prints failures and the summary at the end of stdout, so a naive
    head-only truncation hides the information repair needs. We keep a small
    header plus the tail of stdout and append stderr if present.
    """
    combined = stdout or ""
    if stderr:
        combined += "\n--- stderr ---\n" + stderr

    if not combined:
        return None

    max_len = head + tail + 50
    if len(combined) <= max_len:
        return combined

    return (
        combined[:head]
        + f"\n\n... [truncated {len(combined) - head - tail} chars] ...\n\n"
        + combined[-tail:]
    )


def _sandbox_to_findings(result) -> List[Finding]:
    """Convert sandbox ExecutionResult errors into Finding objects."""
    findings: List[Finding] = []

    _SUGGESTIONS: dict = {
        "no_tests_collected": (
            "Ensure the test file defines at least one test_* function or "
            "Test* class with test_* methods. Check for syntax errors or "
            "import errors before the test definitions."
        ),
        "syntax_error": (
            "Fix the syntax error in the test file. The error is usually "
            "on or near the reported line."
        ),
        "import_error": (
            "Fix the import statement — the module or name may not exist. "
            "Check that imports from source_module use correct names."
        ),
    }

    if result.error_type:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                code=result.error_type,
                message=result.error_message or "Sandbox execution failed",
                line=result.line_number,
                suggestion=_SUGGESTIONS.get(result.error_type),
            )
        )

    if result.tests_failed > 0 and not result.error_type:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                code="test_failure",
                message=f"{result.tests_failed} test(s) failed",
            )
        )

    if result.tests_run == 0 and not result.error_type:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                code="no_tests_collected",
                message="No test functions were collected by pytest",
                suggestion="Ensure the test file defines at least one test_* function or Test* class",
            )
        )

    return findings


def run_pipeline(
    code: str,
    user_request: str = "Generate unit tests",
    file_path: Optional[str] = None,
    max_retries: int = 3,
    config: Optional[Config] = None,
    target_url: Optional[str] = None,
    html_content: Optional[str] = None,
    description: Optional[str] = None,
    target_function: Optional[str] = None,
    repo_metadata: Optional[dict] = None,
) -> PipelineState:
    """Run the pipeline on input code.

    For UI tests, pass ``target_url`` or ``html_content`` and a natural-language
    ``description`` of the user flows to test.
    """
    pipeline = create_pipeline(config)

    initial_state: PipelineState = {
        "code_input": code,
        "file_path": file_path,
        "user_request": user_request,
        "target_url": target_url,
        "html_content": html_content,
        "description": description,
        "target_function": target_function,
        "routing_decision": None,
        "task_type": None,
        "language": None,
        "generated_tests": None,
        "test_functions": None,
        "generated_explanation": None,
        "gate_results": None,
        "verification_report": None,
        "coverage_report": None,
        "verification_result": None,
        "verification_passed": False,
        "sandbox_tests_run": None,
        "sandbox_tests_passed": None,
        "sandbox_branch_coverage": None,
        "sandbox_coverage_data": None,
        "repo_metadata": repo_metadata,
        "execution_context": None,
        "infrastructure_pass": None,
        "repo_setup_pass": None,
        "source_phantom_import": False,
        "source_phantom_packages": [],
        "test_phantom_packages": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "error_type": None,
        "error_message": None,
        "audit_log": [],
        "final_output": None,
        "status": "initialized",
    }

    effective_retries = max_retries
    if config and target_url is not None or html_content is not None:
        effective_retries = max(max_retries, config.ui_test.retry_budget if config else 5)
    recursion_limit = 8 + (effective_retries + 1) * 7
    return pipeline.invoke(initial_state, {"recursion_limit": recursion_limit})
