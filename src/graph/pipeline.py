"""LangGraph pipeline for the GDR (Generate-Detect-Repair) workflow."""

from typing import TypedDict, Optional, List, Literal, Annotated
from datetime import datetime
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from ..agents.router import RouterAgent, RoutingDecision, TaskType
from ..agents.unit_test import UnitTestAgent, GeneratedTest, RepairContext
from ..verification.sandbox import SandboxExecutor, ExecutionResult
from ..utils.logging import AuditLogger
from ..config import Config, get_llm


class VerificationResult(BaseModel):
    """Result from verification gates."""
    
    passed: bool
    gate_name: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    line_number: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    coverage: Optional[float] = None


class AuditEntry(BaseModel):
    """Single audit log entry."""
    
    iteration: int
    timestamp: str
    generated_artifact: Optional[str] = None
    verification_result: Optional[dict] = None
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
    verification_result: Optional[dict]
    verification_passed: bool
    retry_count: int
    max_retries: int
    error_type: Optional[str]
    error_message: Optional[str]
    audit_log: List[dict]
    final_output: Optional[str]
    status: str


def create_pipeline(config: Optional[Config] = None):
    """Create the LangGraph pipeline.
    
    Args:
        config: Optional configuration, loads from env if not provided
        
    Returns:
        Compiled LangGraph workflow
    """
    if config is None:
        config = Config.load()
    
    llm = get_llm(config.llm)
    router_agent = RouterAgent()
    unit_test_agent = UnitTestAgent(llm)
    sandbox = SandboxExecutor(config.sandbox)
    audit_logger = AuditLogger(config.pipeline.audit_log_dir)
    
    def router_node(state: PipelineState) -> PipelineState:
        """Route the request to appropriate specialist."""
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
        """Generate tests using the appropriate agent."""
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
        
        new_audit_log = state["audit_log"] + [audit_entry.model_dump()]
        
        return {
            **state,
            "generated_tests": result.test_code,
            "test_functions": result.test_functions,
            "audit_log": new_audit_log,
            "status": "generated",
        }
    
    def verify_node(state: PipelineState) -> PipelineState:
        """Verify generated tests in sandbox."""
        if not state.get("generated_tests"):
            return {
                **state,
                "verification_passed": False,
                "error_type": "generation_error",
                "error_message": "No tests were generated",
                "status": "verification_failed",
            }
        
        source_code = state["code_input"]
        test_code = state["generated_tests"]
        
        result = sandbox.execute(
            source_code=source_code,
            test_code=test_code,
        )
        
        verification = VerificationResult(
            passed=result.success,
            gate_name="sandbox_execution",
            error_type=result.error_type if not result.success else None,
            error_message=result.error_message if not result.success else None,
            line_number=result.line_number,
            stdout=result.stdout,
            stderr=result.stderr,
            coverage=result.coverage,
        )
        
        if state["audit_log"]:
            last_entry = state["audit_log"][-1].copy()
            last_entry["verification_result"] = verification.model_dump()
            new_audit_log = state["audit_log"][:-1] + [last_entry]
        else:
            new_audit_log = state["audit_log"]
        
        return {
            **state,
            "verification_result": verification.model_dump(),
            "verification_passed": result.success,
            "error_type": result.error_type if not result.success else None,
            "error_message": result.error_message if not result.success else None,
            "audit_log": new_audit_log,
            "status": "verified" if result.success else "verification_failed",
        }
    
    def repair_node(state: PipelineState) -> PipelineState:
        """Repair failed tests based on error feedback."""
        repair_context = RepairContext(
            previous_code=state["generated_tests"] or "",
            error_type=state.get("error_type") or "unknown",
            error_message=state.get("error_message") or "Unknown error",
            line_number=(state.get("verification_result") or {}).get("line_number"),
        )
        
        result = unit_test_agent.repair(repair_context)
        
        new_retry_count = state["retry_count"] + 1
        
        audit_entry = AuditEntry(
            iteration=new_retry_count + 1,
            timestamp=datetime.now().isoformat(),
            generated_artifact=result.test_code,
            repair_context=repair_context.model_dump(),
        )
        
        new_audit_log = state["audit_log"] + [audit_entry.model_dump()]
        
        return {
            **state,
            "generated_tests": result.test_code,
            "test_functions": result.test_functions,
            "retry_count": new_retry_count,
            "audit_log": new_audit_log,
            "status": "repaired",
        }
    
    def output_node(state: PipelineState) -> PipelineState:
        """Prepare final output."""
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
        """Determine if repair should be attempted."""
        if state["verification_passed"]:
            return "output"
        
        if state["retry_count"] >= state["max_retries"]:
            return "output"
        
        return "repair"
    
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("output", output_node)
    
    workflow.set_entry_point("router")
    workflow.add_edge("router", "generate")
    workflow.add_edge("generate", "verify")
    workflow.add_conditional_edges(
        "verify",
        should_repair,
        {
            "repair": "repair",
            "output": "output",
        }
    )
    workflow.add_edge("repair", "verify")
    workflow.add_edge("output", END)
    
    return workflow.compile()


def run_pipeline(
    code: str,
    user_request: str = "Generate unit tests",
    file_path: Optional[str] = None,
    max_retries: int = 3,
    config: Optional[Config] = None,
) -> PipelineState:
    """Run the pipeline on input code.
    
    Args:
        code: Source code to generate tests for
        user_request: User's request
        file_path: Optional file path
        max_retries: Maximum repair attempts
        config: Optional configuration
        
    Returns:
        Final pipeline state
    """
    pipeline = create_pipeline(config)
    
    initial_state: PipelineState = {
        "code_input": code,
        "file_path": file_path,
        "user_request": user_request,
        "routing_decision": None,
        "task_type": None,
        "generated_tests": None,
        "test_functions": None,
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
    
    result = pipeline.invoke(initial_state)
    return result
