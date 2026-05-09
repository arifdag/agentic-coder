"""AblationRunner: generates config variants and runs cross-variant comparisons."""

import copy
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .models import AblationConfig, EvalMetrics
from .runner import BenchmarkRunner

log = logging.getLogger(__name__)

RETRY_BUDGETS = [0, 1, 3, 5]


def generate_variants(
    axes: Optional[List[str]] = None,
) -> List[AblationConfig]:
    """Build the combinatorial set of ablation configs.

    Axes:
        sast      -- SAST on/off
        dependency -- dependency check on/off
        judge     -- LLM judge on/off
        retries   -- retry budget k in {0,1,3,5}
    """
    axes = axes or ["sast", "dependency", "judge", "retries"]

    sast_vals = [True, False] if "sast" in axes else [True]
    dep_vals = [True, False] if "dependency" in axes else [True]
    judge_vals = [True, False] if "judge" in axes else [True]
    retry_vals = RETRY_BUDGETS if "retries" in axes else [3]

    variants: List[AblationConfig] = []
    for sast, dep, judge, k in itertools.product(sast_vals, dep_vals, judge_vals, retry_vals):
        parts = []
        parts.append(f"sast={'on' if sast else 'off'}")
        parts.append(f"dep={'on' if dep else 'off'}")
        parts.append(f"judge={'on' if judge else 'off'}")
        parts.append(f"k={k}")
        name = "_".join(parts)

        variants.append(
            AblationConfig(
                name=name,
                sast_enabled=sast,
                dependency_enabled=dep,
                judge_enabled=judge,
                retry_budget=k,
                description=name.replace("_", ", "),
            )
        )

    return variants


class AblationRunner:
    """Run a benchmark under every ablation variant and compare."""

    def __init__(self, base_config, dataset, results_dir: str = "eval_results"):
        self.base_config = base_config
        self.dataset = dataset
        self.results_dir = Path(results_dir) / "ablation" / dataset.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._variant_metrics: Dict[str, EvalMetrics] = {}

    def _apply_variant(self, config, variant: AblationConfig):
        """Return a deep-copied config with the variant applied.

        The coding and judge role configs (provider/model/fallbacks) are
        intentionally preserved across variants so that ablation results
        reflect the effect of the gate being toggled, not a model swap.
        """
        cfg = copy.deepcopy(config)
        cfg.sast.enabled = variant.sast_enabled
        cfg.dependency.enabled = variant.dependency_enabled
        cfg.judge.enabled = variant.judge_enabled
        cfg.pipeline.max_retries = variant.retry_budget
        # coding_role / judge_role are left unchanged -> judge is pinned.
        return cfg

    def run_all(
        self,
        max_cases_per_variant: Optional[int] = None,
        axes: Optional[List[str]] = None,
        only_variants: Optional[List[str]] = None,
    ) -> Dict[str, EvalMetrics]:
        """Run the ablation sweep.

        Parameters
        ----------
        only_variants:
            If provided, only variants whose ``name`` matches one of the
            entries in this list are executed. Useful for re-running a
            small set of contaminated variants without redoing the whole
            32-variant sweep. Names use the canonical
            ``sast=<on|off>_dep=<on|off>_judge=<on|off>_k=<int>`` form.
        """
        variants = generate_variants(axes)
        if only_variants:
            wanted = set(only_variants)
            variants = [v for v in variants if v.name in wanted]
            missing = wanted - {v.name for v in variants}
            if missing:
                log.warning("Requested variants not found in axes set: %s", sorted(missing))
            if not variants:
                raise ValueError(
                    f"No matching variants for filter {sorted(wanted)}. "
                    f"Hint: variant names look like 'sast=off_dep=off_judge=off_k=0'."
                )
        log.info("Running %d ablation variants on %s", len(variants), self.dataset.name)

        for i, variant in enumerate(variants):
            log.info("[%d/%d] variant=%s", i + 1, len(variants), variant.name)
            cfg = self._apply_variant(self.base_config, variant)

            runner = BenchmarkRunner(
                config=cfg,
                dataset=self.dataset,
                results_dir=str(self.results_dir / variant.name),
            )
            runner.run(max_cases=max_cases_per_variant)
            metrics = runner.summarize()
            metrics.dataset_name = f"{self.dataset.name}/{variant.name}"
            self._variant_metrics[variant.name] = metrics

            runner.save_summary()

        self._save_comparison()
        return self._variant_metrics

    def _save_comparison(self):
        rows: List[Dict] = []
        for name, m in sorted(self._variant_metrics.items()):
            rows.append(
                {
                    "variant": name,
                    "total": m.total,
                    "passed": m.passed,
                    "pass_rate": round(m.pass_rate, 4),
                    "clean_pass_rate": (
                        round(m.clean_pass_rate, 4) if m.clean_pass_rate is not None else None
                    ),
                    "provider_error_count": m.provider_error_count,
                    "avg_coverage": (
                        round(m.avg_coverage, 1) if m.avg_coverage is not None else None
                    ),
                    "avg_target_line_coverage": (
                        round(m.avg_target_line_coverage, 1)
                        if m.avg_target_line_coverage is not None
                        else None
                    ),
                    "avg_target_branch_coverage": (
                        round(m.avg_target_branch_coverage, 1)
                        if m.avg_target_branch_coverage is not None
                        else None
                    ),
                    "mutation_score": (
                        round(m.mutation_score, 4) if m.mutation_score is not None else None
                    ),
                    "mutation_coverage": (
                        round(m.mutation_coverage, 4) if m.mutation_coverage is not None else None
                    ),
                    "relevance_pass_rate": (
                        round(m.relevance_pass_rate, 4)
                        if m.relevance_pass_rate is not None
                        else None
                    ),
                    "gaming_rate": round(m.gaming_rate, 4) if m.gaming_rate is not None else None,
                    "assertion_presence_rate": (
                        round(m.assertion_presence_rate, 4)
                        if m.assertion_presence_rate is not None
                        else None
                    ),
                    "avg_iterations": round(m.avg_iterations, 2),
                    "avg_time": round(m.avg_time, 2),
                }
            )

        coding_prov = (
            self.base_config.coding_role.provenance()
            if getattr(self.base_config, "coding_role", None)
            else None
        )
        judge_prov = (
            (self.base_config.judge_role or self.base_config.coding_role).provenance()
            if getattr(self.base_config, "coding_role", None)
            else None
        )
        comparison_payload = {
            "dataset": self.dataset.name,
            "provenance": {
                "coding": coding_prov,
                "judge": judge_prov,
                "note": "Coding and judge models are pinned across all variants "
                "to isolate the effect of the ablated component.",
            },
            "variants": rows,
        }

        json_path = self.results_dir / "comparison.json"
        json_path.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")

        md_lines = [
            f"# Ablation comparison: {self.dataset.name}\n",
        ]
        if coding_prov and judge_prov:
            md_lines.append(
                f"**Coding:** `{coding_prov['primary']['provider']}:"
                f"{coding_prov['primary']['model']}` &nbsp; "
                f"**Judge:** `{judge_prov['primary']['provider']}:"
                f"{judge_prov['primary']['model']}` (pinned across variants)\n"
            )
        md_lines.extend(
            [
                "| Variant | Total | Passed | Case pass | Target line | Target branch | Mutation | Gaming | Avg iter | Avg time |",
                "|---------|-------|--------|-----------|-------------|---------------|----------|--------|----------|----------|",
            ]
        )
        for r in rows:
            target_line = (
                f"{r['avg_target_line_coverage']}%"
                if r["avg_target_line_coverage"] is not None
                else "N/A"
            )
            target_branch = (
                f"{r['avg_target_branch_coverage']}%"
                if r["avg_target_branch_coverage"] is not None
                else "N/A"
            )
            mutation = f"{r['mutation_score']:.1%}" if r["mutation_score"] is not None else "N/A"
            gaming = f"{r['gaming_rate']:.1%}" if r["gaming_rate"] is not None else "N/A"
            md_lines.append(
                f"| {r['variant']} | {r['total']} | {r['passed']} | {r['pass_rate']:.1%} "
                f"| {target_line} | {target_branch} | {mutation} | {gaming} "
                f"| {r['avg_iterations']} | {r['avg_time']}s |"
            )

        md_path = self.results_dir / "comparison.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        log.info("Saved ablation comparison to %s", md_path)
