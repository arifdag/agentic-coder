"""Print a quick summary table of an ablation run so far (partial runs OK)."""

import json
import sys
from pathlib import Path


def main(root: str = "eval_results_ablation/ablation/ult") -> None:
    p = Path(root)
    if not p.exists():
        print(f"no such directory: {p}")
        return

    rows = []
    for d in sorted(p.iterdir()):
        if not d.is_dir():
            continue
        candidates = list(d.rglob("summary.json"))
        if not candidates:
            continue
        summary = json.loads(candidates[0].read_text(encoding="utf-8"))
        rows.append((d.name, summary))

    if not rows:
        print("no completed variants yet")
        return

    hdr = (
        f"{'variant':55} {'case-pass':>10} {'test-pass':>10} "
        f"{'target':>10} {'mutation':>10} {'rel':>8} {'gaming':>8} "
        f"{'iters':>7} {'time(s)':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows:
        cp = m.get("pass_rate", 0) * 100
        tp = m.get("test_pass_rate", 0) * 100
        target = m.get("avg_target_line_coverage")
        mutation = m.get("mutation_score")
        rel = m.get("relevance_pass_rate")
        gaming = m.get("gaming_rate")
        it = m.get("avg_iterations") or 0.0
        tm = m.get("avg_time") or 0.0
        target_s = f"{target:>9.1f}%" if target is not None else f"{'N/A':>10}"
        mutation_s = f"{mutation * 100:>9.1f}%" if mutation is not None else f"{'N/A':>10}"
        rel_s = f"{rel * 100:>7.1f}%" if rel is not None else f"{'N/A':>8}"
        gaming_s = f"{gaming * 100:>7.1f}%" if gaming is not None else f"{'N/A':>8}"
        print(
            f"{name:55} {cp:>9.1f}% {tp:>9.1f}% "
            f"{target_s} {mutation_s} {rel_s} {gaming_s} "
            f"{it:>7.2f} {tm:>8.1f}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results_ablation/ablation/ult")
