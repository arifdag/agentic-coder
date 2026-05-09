"""Cross-benchmark health check for an evaluation results directory."""

import json
import sys
from pathlib import Path


def main(root_dir: str = "eval_results_phase6_v2") -> None:
    root = Path(root_dir)
    if not root.exists():
        print(f"no such directory: {root}")
        return

    print(
        f"{'benchmark':20} {'cases':>6} {'errored':>8} {'zero-tests':>11} "
        f"{'cases-pass':>11} {'test-pass':>10} {'coverage':>10} "
        f"{'target':>10} {'mutation':>10} {'gaming':>8}"
    )
    print("-" * 125)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        files = [f for f in d.glob("*.json") if f.name not in ("summary.json", "comparison.json")]
        errored = zero = case_pass = 0
        tests_run = tests_pass = 0
        covs = []
        target_covs = []
        mutation_scores = []
        gaming_flags = []
        for f in files:
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - diagnostic tooling
                continue
            err = j.get("error")
            err_type = (j.get("pipeline_state") or {}).get("error_type")
            if err or err_type in ("environment_error", "docker_error"):
                errored += 1
            if j.get("tests_run", 0) == 0 and not err:
                zero += 1
            if j.get("passed"):
                case_pass += 1
            tests_run += j.get("tests_run", 0)
            tests_pass += j.get("tests_passed", 0)
            cov = j.get("coverage")
            if cov is not None:
                covs.append(cov)
            coverage_metrics = j.get("coverage_metrics") or {}
            target_cov = coverage_metrics.get("target_line_coverage")
            if target_cov is not None:
                target_covs.append(target_cov)
            mutation_metrics = j.get("mutation_metrics") or {}
            mut_score = mutation_metrics.get("mutation_score")
            if mut_score is not None:
                mutation_scores.append(mut_score)
            relevance_metrics = j.get("relevance_metrics") or {}
            if "gaming_flag" in relevance_metrics:
                gaming_flags.append(bool(relevance_metrics.get("gaming_flag")))

        tp = (tests_pass / tests_run * 100) if tests_run else 0.0
        cp = (case_pass / len(files) * 100) if files else 0.0
        cov_mean = (sum(covs) / len(covs)) if covs else None
        cov_str = f"{cov_mean:.1f}%" if cov_mean is not None else "-"
        target_mean = (sum(target_covs) / len(target_covs)) if target_covs else None
        target_str = f"{target_mean:.1f}%" if target_mean is not None else "-"
        mutation_mean = (sum(mutation_scores) / len(mutation_scores)) if mutation_scores else None
        mutation_str = f"{mutation_mean * 100:.1f}%" if mutation_mean is not None else "-"
        gaming = (
            (sum(1 for x in gaming_flags if x) / len(gaming_flags) * 100) if gaming_flags else None
        )
        gaming_str = f"{gaming:.1f}%" if gaming is not None else "-"
        print(
            f"{d.name:20} {len(files):>6} {errored:>8} {zero:>11} "
            f"{cp:>10.1f}% {tp:>9.1f}% {cov_str:>10} "
            f"{target_str:>10} {mutation_str:>10} {gaming_str:>8}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results_phase6_v2")
