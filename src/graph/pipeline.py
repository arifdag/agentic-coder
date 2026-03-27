"""LangGraph pipeline for the GDR (Generate-Detect-Repair) workflow.

Phase 2: Multi-gate verification with SAST, dependency checks, LLM judge,
sandboxed execution with coverage, and structured diagnostics.
"""

from typing import TypedDict, Optional, List, Literal
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from ..agents.router import RouterAgent, TaskType
from ..agents.unit_test import UnitTestAgent, RepairContext
from ..verification.sandbox import SandboxExecutor
from ..verification.sast import SastAnalyzer
from ..verification.dependency import DependencyValidator
from ..verification.judge import SastJudge
from ..verification.models import GateResult, VerificationReport, Finding, Severity
from ..utils.logging import AuditLogger
from ..config import Config, get_llm


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

    routing_decision: Optional[dict]
    task_type: Optional[str]

    generated_tests: Optional[str]
    test_functions: Optional[List[str]]

    # Multi-gate verification (Phase 2)
    gate_results: Optional[List[dict]]
    verification_report: Optional[dict]
    coverage_report: Optional[str]

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

    llm = get_llm(config.llm)
    router_agent = RouterAgent()
    unit_test_agent = UnitTestAgent(llm)
    sandbox = SandboxExecutor(config.sandbox)
    audit_logger = AuditLogger(config.pipeline.audit_log_dir)

    sast_analyzer = SastAnalyzer(
        semgrep_rules=config.sast.semgrep_rules,
        bandit_enabled=config.sast.bandit_enabled,
        timeout=config.sast.timeout,
    ) if config.sast.enabled else None

    dep_validator = DependencyValidator(
        pypi_timeout=config.dependency.pypi_timeout,
    ) if config.dependency.enabled else None

    judge_llm = llm
    if config.judge.enabled and config.judge.provider:
        from ..config import LLMConfig
        judge_cfg = LLMConfig.from_env(config.judge.provider)
        if config.judge.model:
            judge_cfg.model = config.judge.model
        judge_llm = get_llm(judge_cfg)

    sast_judge = SastJudge(judge_llm) if config.judge.enabled else None

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
            "status": "routed",
        }

    def generate_node(state: PipelineState) -> PipelineState:
        task_type = state.get("task_type", TaskType.UNIT_TEST.value)

        if task_type == TaskType.UNIT_TEST.value:
            result = unit_test_agent.generate(
                code=state["code_input"],
                file_path=state.get("file_path"),
            )
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
        """Run Gate 1 (SAST) and Gate 2 (dependency) concurrently."""
        if not state.get("generated_tests"):
            return {
                **state,
                "gate_results": [],
                "verification_passed": False,
                "error_type": "generation_error",
                "error_message": "No tests were generated",
                "status": "verification_failed",
            }

        test_code = state["generated_tests"]
        source_and_test = state["code_input"] + "\n\n" + test_code
        gate_results: List[dict] = []

        def run_sast():
            if sast_analyzer:
                return sast_analyzer.analyze(source_and_test)
            return GateResult(gate_name="sast", passed=True, findings=[])

        def run_deps():
            if dep_validator:
                return dep_validator.validate(test_code)
            return GateResult(gate_name="dependency", passed=True, findings=[])

        with ThreadPoolExecutor(max_workers=2) as pool:
            sast_future = pool.submit(run_sast)
            dep_future = pool.submit(run_deps)
            sast_result = sast_future.result()
            dep_result = dep_future.result()

        gate_results = [sast_result.model_dump(), dep_result.model_dump()]

        static_passed = sast_result.passed and dep_result.passed

        return {
            **state,
            "gate_results": gate_results,
            "verification_passed": static_passed,
            "status": "static_checked",
        }

    def judge_node(state: PipelineState) -> PipelineState:
        """Run LLM judge on SAST findings to filter false positives."""
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
        """Run Gate 3: sandbox execution with coverage."""
        source_code = state["code_input"]
        test_code = state["generated_tests"] or ""

        result = sandbox.execute(
            source_code=source_code,
            test_code=test_code,
        )

        sandbox_gate = GateResult(
            gate_name="sandbox",
            passed=result.success,
            findings=_sandbox_to_findings(result),
            details=result.stdout[:2000] if result.stdout else None,
        )

        prior_gates = state.get("gate_results") or []
        all_gates = prior_gates + [sandbox_gate.model_dump()]

        return {
            **state,
            "gate_results": all_gates,
            "coverage_report": result.coverage_gaps,
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
            m = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', sandbox_gate.details)
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
        report_data = state.get("verification_report")
        diagnostics = None
        if report_data:
            report = VerificationReport(**report_data)
            diagnostics = report.format_for_repair()

        repair_context = RepairContext(
            previous_code=state.get("generated_tests") or "",
            error_type=state.get("error_type") or "unknown",
            error_message=state.get("error_message") or "Unknown error",
            line_number=None,
            coverage_gaps=state.get("coverage_report"),
            diagnostics=diagnostics,
        )

        result = unit_test_agent.repair(repair_context)

        new_retry = state["retry_count"] + 1
        audit_entry = AuditEntry(
            iteration=new_retry + 1,
            timestamp=datetime.now().isoformat(),
            generated_artifact=result.test_code,
            repair_context=repair_context.model_dump(),
        )

        return {
            **state,
            "generated_tests": result.test_code,
            "test_functions": result.test_functions,
            "retry_count": new_retry,
            "gate_results": None,
            "verification_report": None,
            "coverage_report": None,
            "audit_log": state["audit_log"] + [audit_entry.model_dump()],
            "status": "repaired",
        }

    def output_node(state: PipelineState) -> PipelineState:
        if state["verification_passed"]:
            final_output = state["generated_tests"]
            status = "success"
        else:
            final_output = state.get("generated_tests")
            status = "failed_after_retries"

        audit_logger.save(state["audit_log"], state.get("file_path", "unknown"))

        return {
            **state,
            "final_output": final_output,
            "status": status,
        }

    def should_repair(state: PipelineState) -> Literal["repair", "output"]:
        if state["verification_passed"]:
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

def _sandbox_to_findings(result) -> List[Finding]:
    """Convert sandbox ExecutionResult errors into Finding objects."""
    findings: List[Finding] = []

    if result.error_type:
        findings.append(Finding(
            severity=Severity.ERROR,
            code=result.error_type,
            message=result.error_message or "Sandbox execution failed",
            line=result.line_number,
        ))

    if result.tests_failed > 0 and not result.error_type:
        findings.append(Finding(
            severity=Severity.ERROR,
            code="test_failure",
            message=f"{result.tests_failed} test(s) failed",
        ))

    return findings


def run_pipeline(
    code: str,
    user_request: str = "Generate unit tests",
    file_path: Optional[str] = None,
    max_retries: int = 3,
    config: Optional[Config] = None,
) -> PipelineState:
    """Run the pipeline on input code."""
    pipeline = create_pipeline(config)

    initial_state: PipelineState = {
        "code_input": code,
        "file_path": file_path,
        "user_request": user_request,
        "routing_decision": None,
        "task_type": None,
        "generated_tests": None,
        "test_functions": None,
        "gate_results": None,
        "verification_report": None,
        "coverage_report": None,
        "verification_result": None,
        "verification_passed": False,
        "retry_count": 0,
        "max_retries": max_retries,
        "error_type": None,
        "error_message": None,
        "audit_log": [],
        "final_output": None,
        "status": "initialized",
    }

    return pipeline.invoke(initial_state)
