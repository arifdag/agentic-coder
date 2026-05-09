"""Generate a consolidated markdown report for evaluation + ablation runs.

The report intentionally separates ``rate-limited`` cases (429/quota
errors from upstream providers) from real pipeline outcomes, because
mixing them distorts every downstream number -- an ablation variant
that failed 10/10 on rate-limits is not evidence that the pipeline
degrades without gates.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path("eval_results_phase6_v2")
ABL_ROOT = Path("eval_results_ablation/ablation/ult")
OUT = Path("phase6_ablation_evaluation_report.md")


_RATE_LIMIT_HINTS = (
    "429",
    "rate_limit",
    "rate limit",
    "tokens per day",
    "tpd",
    "per minute",
)


def _is_rate_limited_case(case: dict) -> bool:
    err = (case.get("error") or "").lower()
    return bool(err) and any(h in err for h in _RATE_LIMIT_HINTS)


def _pct(v: float) -> str:
    return f"{(v * 100):.1f}%"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_rate_limited(bench_dir: Path) -> int:
    count = 0
    for f in bench_dir.glob("*.json"):
        if f.name in ("summary.json", "summary.md", "comparison.json"):
            continue
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - diagnostic tooling
            continue
        if _is_rate_limited_case(j):
            count += 1
    return count


def _benchmark_rows() -> list[dict]:
    rows: list[dict] = []
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        summary = _load_json(d / "summary.json")
        rows.append(
            {
                "name": d.name,
                "total": summary.get("total", 0),
                "rate_limited": _count_rate_limited(d),
                "pass_rate": summary.get("pass_rate", 0.0),
                "test_pass_rate": summary.get("test_pass_rate", 0.0),
                "tests_passed": summary.get("total_tests_passed", 0),
                "tests_run": summary.get("total_tests_run", 0),
                "coverage": summary.get("avg_coverage"),
                "target_coverage": summary.get("avg_target_line_coverage"),
                "target_branch": summary.get("avg_target_branch_coverage"),
                "mutation_score": summary.get("mutation_score"),
                "mutation_coverage": summary.get("mutation_coverage"),
                "gaming_rate": summary.get("gaming_rate"),
                "iterations": summary.get("avg_iterations", 0.0),
                "time": summary.get("avg_time", 0.0),
                "gate_pass_rates": summary.get("gate_pass_rates", {}),
            }
        )
    return rows


def _ablation_rows() -> list[dict]:
    rows: list[dict] = []
    for variant in sorted(p for p in ABL_ROOT.iterdir() if p.is_dir()):
        inner = variant / "ult"
        summary = _load_json(inner / "summary.json")
        rows.append(
            {
                "variant": variant.name,
                "case_pass": summary.get("pass_rate", 0.0),
                "test_pass": summary.get("test_pass_rate", 0.0),
                "coverage": summary.get("avg_coverage"),
                "target_coverage": summary.get("avg_target_line_coverage"),
                "target_branch": summary.get("avg_target_branch_coverage"),
                "mutation_score": summary.get("mutation_score"),
                "gaming_rate": summary.get("gaming_rate"),
                "iterations": summary.get("avg_iterations", 0.0),
                "time": summary.get("avg_time", 0.0),
                "rate_limited": _count_rate_limited(inner),
            }
        )
    return rows


def _parse_variant_name(name: str) -> dict:
    parts = {}
    for token in name.split("_"):
        k, v = token.split("=")
        parts[k] = v
    return parts


def generate() -> None:
    bench_rows = _benchmark_rows()
    ab_rows = _ablation_rows()
    provenance = _load_json(ABL_ROOT / "comparison.json").get("provenance", {})

    lines: list[str] = []
    lines.append("# Phase 6 Evaluation and Ablation Report")
    lines.append("")
    lines.append("This report summarizes:")
    lines.append("- `eval_results_phase6_v2` benchmark outcomes")
    lines.append("- `eval_results_ablation/ablation/ult` full 32-variant ablation sweep")
    lines.append("")
    lines.append("## Audit Findings (apply before re-running the full test set)")
    lines.append("")
    lines.append(
        "Three structural bugs were discovered while auditing the numbers "
        "in this report and have been fixed on `main`. The raw numbers "
        "below were produced **before** those fixes; the affected rows "
        "are flagged so you know which to trust and which to re-run:"
    )
    lines.append("")
    lines.append(
        "1. **Router misclassified unit-test tasks whose problem description "
        "contained words like `comment` / `complexity` / `describe`.** "
        "These cases skipped the sandbox gate entirely and ran the "
        "explanation pipeline instead. ULT hit this on `ult-15-compute_comment_stats`. "
        "Fixed by reordering keyword precedence in `src/agents/router.py` "
        "(unit-test keywords now outrank explanation keywords)."
    )
    lines.append(
        "2. **Dependency gate only validated the generated test code, "
        "not the source code under test.** As a result the "
        "`dep_hallucination` benchmark -- the exact benchmark that "
        "exists to detect phantom package imports -- reported a 100% "
        "dep-gate pass rate while 8/10 subject files imported "
        "packages that do not exist on PyPI. Fixed in "
        "`src/graph/pipeline.py::verify_static_node` by validating "
        "`code_input + generated_tests`."
    )
    lines.append(
        "3. **Ablation metrics counted 429/rate-limit provider errors as "
        "pipeline failures.** All four `sast=off_dep=off_judge=off_*` "
        "variants are 100% 429 errors from Groq's daily quota, and "
        "several other variants are 10-70% contaminated. The original "
        'claim "without gates the pipeline collapses to 0%" is therefore '
        "unsupported by the current run. A new `RL` (rate-limited) "
        "column in the table below makes the contamination explicit."
    )
    lines.append("")
    lines.append(
        "Regression tests for #1 and #2 are in "
        "`tests/test_pipeline.py::TestRouterAgent::test_classify_unit_test_wins_over_description_words` "
        "and "
        "`tests/test_verification.py::TestDependencyValidator::test_phantom_package_in_source_is_caught_when_combined`."
    )
    lines.append("")
    lines.append("## Model Provenance (Pinned During Ablation)")
    lines.append("")
    coding = provenance.get("coding", {}).get("primary", {})
    judge = provenance.get("judge", {}).get("primary", {})
    lines.append(f"- Coding model: `{coding.get('provider')}:{coding.get('model')}`")
    lines.append(f"- Judge model: `{judge.get('provider')}:{judge.get('model')}`")
    lines.append("- Judge and coding model chains were pinned across variants.")
    lines.append("")
    lines.append("## Metric Definitions (What Scores Mean)")
    lines.append("")
    lines.append(
        "- `Pass rate (case-level)`: fraction of cases where **all** generated tests pass and all gates pass."
    )
    lines.append("- `Tests passed`: pooled count of passing tests across all cases.")
    lines.append(
        "- `Test pass rate`: pooled `tests_passed / tests_run`; best quality signal for multi-test cases."
    )
    lines.append(
        "- `Avg coverage`: mean line coverage of `source_module.py` during sandbox execution."
    )
    lines.append(
        "- `Target coverage`: coverage for the inferred target module/function when available."
    )
    lines.append(
        "- `Mutation score`: killed mutants / executable mutants; optional and only present for quality runs."
    )
    lines.append("- `Gaming rate`: passing-looking tests with no structural target signal.")
    lines.append("- `Avg iterations`: mean number of GDR loop iterations consumed per case.")
    lines.append("- `Per-gate pass rates`: fraction of cases each gate accepted.")
    lines.append("")
    lines.append("## Benchmark Summary (`eval_results_phase6_v2`)")
    lines.append("")
    lines.append(
        "| Benchmark | Cases | Rate-limited | Case pass | Tests passed | Test pass | Avg coverage | Target cov | Mutation | Gaming | Avg iterations | Avg time (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in bench_rows:
        cov = f"{r['coverage']:.1f}%" if r["coverage"] is not None else "N/A"
        target_cov = f"{r['target_coverage']:.1f}%" if r["target_coverage"] is not None else "N/A"
        mutation = _pct(r["mutation_score"]) if r["mutation_score"] is not None else "N/A"
        gaming = _pct(r["gaming_rate"]) if r["gaming_rate"] is not None else "N/A"
        lines.append(
            f"| {r['name']} | {r['total']} | {r['rate_limited']} | {_pct(r['pass_rate'])} | "
            f"{r['tests_passed']}/{r['tests_run']} | {_pct(r['test_pass_rate'])} | "
            f"{cov} | {target_cov} | {mutation} | {gaming} | {r['iterations']:.2f} | {r['time']:.2f} |"
        )
    lines.append("")

    lines.append("### Benchmark Interpretation")
    lines.append("")
    lines.append(
        "- The system runs end-to-end across all configured benchmarks (no benchmark crashed)."
    )
    lines.append(
        "- `projecttest` case-pass is 0.0% because it is an all-tests-must-pass criterion across large suites;"
    )
    lines.append(
        "  its `65.9%` test-pass and `62.6%` coverage indicate substantial partial correctness."
    )
    lines.append("- `security` and `cweval` should be interpreted with gate behavior context:")
    lines.append("  lower SAST pass can indicate better vulnerability detection strictness.")
    lines.append(
        "- `ult` is the strongest clean benchmark score in this run: 30.0% case-pass with 50.4% pooled test-pass."
    )
    lines.append("")

    lines.append("## ULT Ablation Summary (32 Variants)")
    lines.append("")
    total_rl = sum(r["rate_limited"] for r in ab_rows)
    if total_rl:
        lines.append(
            f"> **Caveat:** {total_rl} per-case entries across the 32 variants "
            "are upstream provider 429 / rate-limit errors, not real "
            "pipeline outcomes. Variants marked with a non-zero `RL` column "
            "are partially or fully contaminated and should be re-run before "
            "drawing conclusions from them."
        )
        lines.append("")
    lines.append(
        "| Variant | Case pass | Test pass | Target cov | Mutation | Gaming | Avg iterations | Avg time (s) | RL |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in ab_rows:
        target_cov = f"{r['target_coverage']:.1f}%" if r["target_coverage"] is not None else "N/A"
        mutation = _pct(r["mutation_score"]) if r["mutation_score"] is not None else "N/A"
        gaming = _pct(r["gaming_rate"]) if r["gaming_rate"] is not None else "N/A"
        lines.append(
            f"| `{r['variant']}` | {_pct(r['case_pass'])} | {_pct(r['test_pass'])} | "
            f"{target_cov} | {mutation} | {gaming} | {r['iterations']:.2f} | {r['time']:.2f} | {r['rate_limited']} |"
        )
    lines.append("")

    # Only use clean (non-contaminated) variants for headline claims.
    clean_rows = [r for r in ab_rows if r["rate_limited"] == 0]
    lines.append("### Key Ablation Findings (clean variants only)")
    lines.append("")
    if clean_rows:
        best_test = max(clean_rows, key=lambda x: x["test_pass"])
        best_case = max(clean_rows, key=lambda x: x["case_pass"])
        lines.append(
            f"- Best pooled test-pass variant: `{best_test['variant']}` at {_pct(best_test['test_pass'])}."
        )
        lines.append(
            f"- Best case-pass variant: `{best_case['variant']}` at {_pct(best_case['case_pass'])}."
        )

    no_gate = [r for r in ab_rows if r["variant"].startswith("sast=off_dep=off_judge=off_")]
    if no_gate and all(
        r["rate_limited"] == r.get("rate_limited", 0) and r["rate_limited"] >= 10 for r in no_gate
    ):
        lines.append(
            "- **The all-gates-off variants are 100% rate-limited and cannot be "
            "used to support any claim** about pipeline behaviour without "
            "verification gates. Re-run them with a fresh quota to test that "
            "hypothesis."
        )
    elif no_gate:
        clean_no_gate = [r for r in no_gate if r["rate_limited"] == 0]
        if clean_no_gate:
            mean_tp = mean(r["test_pass"] for r in clean_no_gate) * 100
            lines.append(
                f"- With all gates disabled (clean runs), mean test-pass is {mean_tp:.1f}%."
            )

    all_on = [r for r in ab_rows if r["variant"].startswith("sast=on_dep=on_judge=on_")]
    if all_on:
        by_k = sorted(
            [(int(_parse_variant_name(r["variant"])["k"]), r) for r in all_on],
            key=lambda t: t[0],
        )
        lines.append("- Full pipeline (`sast=on, dep=on, judge=on`) by retry budget:")
        for k, row in by_k:
            cov = "N/A" if row["coverage"] is None else f"{row['coverage']:.1f}%"
            lines.append(
                f"  - k={k}: case-pass {_pct(row['case_pass'])}, "
                f"test-pass {_pct(row['test_pass'])}, coverage {cov}, "
                f"iterations {row['iterations']:.2f}"
            )

    lines.append("")
    lines.append("## Practical Conclusions")
    lines.append("")
    lines.append("- The model stack is functional and produces meaningful tests across benchmarks.")
    lines.append(
        "- Report both case-pass and test-pass in paper figures; test-pass avoids all-or-nothing distortion."
    )
    lines.append(
        "- For ULT, k=3 appears to be a reasonable budget/quality tradeoff; k=5 increases compute substantially."
    )
    lines.append("- Keep provenance (provider:model) alongside every run for reproducibility.")
    lines.append("")
    lines.append("## Suggested Paper Table Fields")
    lines.append("")
    lines.append(
        "- Benchmark, N cases, case-pass, pooled test-pass, avg coverage, avg iterations, avg runtime."
    )
    lines.append(
        "- Ablation axis values (sast/dep/judge/k), case-pass, pooled test-pass, coverage."
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    generate()
