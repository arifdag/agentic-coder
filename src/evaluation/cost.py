"""CostAnalyzer: parse audit logs to estimate token usage, time, and cost."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class RunCost:
    """Cost estimate for a single pipeline run."""

    def __init__(self):
        self.source: str = ""
        self.iterations: int = 0
        self.total_chars: int = 0
        self.estimated_tokens: int = 0
        self.estimated_cost_usd: float = 0.0


class CostAnalyzer:
    """Parse JSON audit logs and compute cost estimates."""

    def __init__(
        self,
        log_dir: str = "audit_logs",
        chars_per_token: int = 4,
        cost_per_million_tokens: float = 0.0,
    ):
        self.log_dir = Path(log_dir)
        self.chars_per_token = chars_per_token
        self.cost_per_mtok = cost_per_million_tokens

    def _estimate_tokens(self, char_count: int) -> int:
        if self.chars_per_token <= 0:
            return 0
        return char_count // self.chars_per_token

    def _load_log(self, path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def analyze(self) -> List[RunCost]:
        if not self.log_dir.exists():
            log.warning("Audit log directory %s not found", self.log_dir)
            return []

        runs: List[RunCost] = []
        for log_file in sorted(self.log_dir.glob("*.json")):
            data = self._load_log(log_file)
            if not data:
                continue

            rc = RunCost()
            rc.source = data.get("source", log_file.stem)
            entries = data.get("entries", [])
            rc.iterations = len(entries)

            total_chars = 0
            for entry in entries:
                artifact = entry.get("generated_artifact") or ""
                total_chars += len(artifact)
                repair_ctx = entry.get("repair_context") or {}
                for val in repair_ctx.values():
                    if isinstance(val, str):
                        total_chars += len(val)

            rc.total_chars = total_chars
            rc.estimated_tokens = self._estimate_tokens(total_chars)
            rc.estimated_cost_usd = (rc.estimated_tokens / 1_000_000) * self.cost_per_mtok
            runs.append(rc)

        return runs

    def summarize(self) -> Dict[str, float]:
        runs = self.analyze()
        if not runs:
            return {}

        total_iter = sum(r.iterations for r in runs)
        total_tokens = sum(r.estimated_tokens for r in runs)
        total_cost = sum(r.estimated_cost_usd for r in runs)
        n = len(runs)

        return {
            "total_runs": n,
            "total_iterations": total_iter,
            "total_estimated_tokens": total_tokens,
            "total_estimated_cost_usd": round(total_cost, 4),
            "avg_iterations": round(total_iter / n, 2),
            "avg_tokens_per_run": round(total_tokens / n, 0),
            "avg_cost_per_run_usd": round(total_cost / n, 6),
        }

    def to_markdown(self) -> str:
        runs = self.analyze()
        if not runs:
            return "No audit logs found.\n"

        summary = self.summarize()

        lines = [
            "# Cost Analysis\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total runs | {summary['total_runs']} |",
            f"| Total iterations | {summary['total_iterations']} |",
            f"| Total est. tokens | {summary['total_estimated_tokens']:,.0f} |",
            f"| Total est. cost | ${summary['total_estimated_cost_usd']:.4f} |",
            f"| Avg iterations/run | {summary['avg_iterations']} |",
            f"| Avg tokens/run | {summary['avg_tokens_per_run']:,.0f} |",
            f"| Avg cost/run | ${summary['avg_cost_per_run_usd']:.6f} |",
            "",
            "## Per-run breakdown\n",
            "| Source | Iterations | Est. tokens | Est. cost |",
            "|--------|------------|-------------|-----------|",
        ]
        for r in runs:
            lines.append(
                f"| {r.source[:40]} | {r.iterations} | {r.estimated_tokens:,} | ${r.estimated_cost_usd:.6f} |"
            )

        return "\n".join(lines)
