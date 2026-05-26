"""Audit ablation runs for rate-limit errors, zero-test contamination, and other anomalies."""
from __future__ import annotations

import json
from pathlib import Path


ABL = Path("eval_results_ablation/ablation/ult")


def _cases(variant_dir: Path):
    inner = variant_dir / "ult"
    if not inner.exists():
        return []
    return sorted(f for f in inner.glob("*.json") if f.name != "summary.json")


def main() -> None:
    rows = []
    for d in sorted(p for p in ABL.iterdir() if p.is_dir()):
        files = _cases(d)
        if not files:
            continue
        rate_limited = no_gates = no_sandbox = ok = 0
        for f in files:
            j = json.loads(f.read_text(encoding="utf-8"))
            err = j.get("error") or ""
            gates = j.get("gate_results") or []
            has_sandbox_gate = any(g.get("gate_name") == "sandbox" for g in gates)
            if "rate_limit" in err.lower() or "429" in err:
                rate_limited += 1
            elif not gates:
                no_gates += 1
            elif not has_sandbox_gate:
                no_sandbox += 1
            else:
                ok += 1
        n = len(files)
        rows.append((d.name, n, rate_limited, no_gates, no_sandbox, ok))

    hdr = f"{'variant':55} {'N':>3} {'rate-lim':>8} {'no-gates':>9} {'no-sandbox':>11} {'ok':>4}"
    print(hdr)
    print("-" * len(hdr))
    total_rl = 0
    for name, n, rl, ng, ns, ok in rows:
        print(f"{name:55} {n:>3} {rl:>8} {ng:>9} {ns:>11} {ok:>4}")
        total_rl += rl
    print("-" * len(hdr))
    print(f"TOTAL rate-limited cases: {total_rl}")


if __name__ == "__main__":
    main()
