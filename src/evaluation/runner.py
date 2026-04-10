"""BenchmarkRunner: feeds datasets through the GDR pipeline and collects results."""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from .models import BenchmarkCase, EvalResult, EvalMetrics

log = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run the GDR pipeline against every case in a benchmark dataset."""

    def __init__(self, config, dataset, results_dir: str = "eval_results"):
        self.config = config
        self.dataset = dataset
        self.results_dir = Path(results_dir) / dataset.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._results: List[EvalResult] = []

    def run(self, max_cases: Optional[int] = None) -> List[EvalResult]:
        from ..graph.pipeline import run_pipeline

        cases = self.dataset.load()
        if max_cases is not None:
            cases = cases[:max_cases]

        log.info("Running %d cases from %s", len(cases), self.dataset.name)

        for i, case in enumerate(cases):
            log.info("[%d/%d] %s", i + 1, len(cases), case.id)
            result = self._run_one(case, run_pipeline)
            self._results.append(result)

            result_path = self.results_dir / f"{case.id}.json"
            result_path.write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )

        return self._results

    def _run_one(self, case: BenchmarkCase, run_pipeline_fn) -> EvalResult:
        start = time.time()
        try:
            state = run_pipeline_fn(
                code=case.code,
                user_request=case.user_request or "Generate comprehensive unit tests",
                file_path=None,
                max_retries=self.config.pipeline.max_retries,
                config=self.config,
            )
            elapsed = time.time() - start

            report = state.get("verification_report") or {}
            gates = report.get("gates", [])
            coverage_val = report.get("coverage")

            return EvalResult(
                case_id=case.id,
                passed=state.get("status") == "success",
                elapsed_seconds=round(elapsed, 2),
                tests_run=len(state.get("test_functions") or []),
                tests_passed=len(state.get("test_functions") or []) if state.get("status") == "success" else 0,
                coverage=coverage_val,
                iterations=state.get("retry_count", 0) + 1,
                gate_results=gates,
                pipeline_state={
                    "status": state.get("status"),
                    "task_type": state.get("task_type"),
                    "language": state.get("language"),
                    "error_type": state.get("error_type"),
                    "error_message": state.get("error_message"),
                },
            )
        except Exception as exc:
            elapsed = time.time() - start
            log.error("Case %s failed: %s", case.id, exc)
            return EvalResult(
                case_id=case.id,
                passed=False,
                elapsed_seconds=round(elapsed, 2),
                error=str(exc),
            )

    def summarize(self) -> EvalMetrics:
        return EvalMetrics.from_results(self._results, dataset_name=self.dataset.name)

    def save_summary(self) -> Path:
        metrics = self.summarize()
        summary_path = self.results_dir / "summary.json"
        summary_path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")

        md_path = self.results_dir / "summary.md"
        md_path.write_text(metrics.to_markdown(), encoding="utf-8")

        log.info("Saved summary to %s", summary_path)
        return summary_path
