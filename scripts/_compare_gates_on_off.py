"""Side-by-side comparison: ULT ablation with gates ON vs OFF, at each
repair depth k. Reports raw case-pass, test-pass, test relevance, and
gaming rate -- all computed on clean cases only.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ABL = pathlib.Path("eval_results_paper_ablation/ablation/ult")
VARIANT_RE = re.compile(
    r"sast=(on|off)_dep=(on|off)_judge=(on|off)(?:_rel=(on|off))?_k=(\d+)"
)
_TEST_NAME_RE = re.compile(r"test_generated\.py::([\w:]+)::([\w\[\]\-]+)")
NETWORK = ("getaddrinfo", "name or service not known", "connection reset",
           "winerror", "remote protocol", "server disconnected")
RATELIMIT = ("429", "rate_limit", "tokens per day", "tokens per minute")


def is_contam(err: str) -> bool:
    e = (err or "").lower()
    return bool(e) and (
        any(h in e for h in NETWORK)
        or any(h in e for h in RATELIMIT)
        or "timed out" in e
    )


def keywords(case_id: str) -> list[str]:
    parts = case_id.split("-")
    if len(parts) < 3:
        return []
    raw = parts[-1]
    out: set[str] = {raw, raw.lower()}
    for tok in raw.split("_"):
        if len(tok) >= 3:
            out.add(tok.lower())
    for tok in re.findall(r"[A-Z][a-z]+|[a-z]+", raw):
        if len(tok) >= 3:
            out.add(tok.lower())
    return [k for k in out if k]


def variant_stats(folder: pathlib.Path) -> dict:
    cases = sorted((folder / "ult").glob("ult-*.json"))
    n = passed = 0
    tr = tp = 0
    iters: list[int] = []
    covs: list[float] = []
    rels: list[float] = []
    gaming = 0
    n_with_tests = 0
    for p in cases:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if is_contam(d.get("error", "")):
            continue
        n += 1
        if d.get("passed"):
            passed += 1
        tr += int(d.get("tests_run", 0) or 0)
        tp += int(d.get("tests_passed", 0) or 0)
        if d.get("iterations") is not None:
            iters.append(int(d["iterations"]))
        if d.get("coverage") is not None:
            covs.append(float(d["coverage"]))
        sb = next((g for g in d.get("gate_results", []) if g.get("gate_name") == "sandbox"), None)
        if not sb or not sb.get("details"):
            continue
        names = [f"{m.group(1)}::{m.group(2)}" for m in _TEST_NAME_RE.finditer(sb["details"])]
        if not names:
            continue
        n_with_tests += 1
        kw = keywords(d.get("case_id", p.stem))
        matched = sum(1 for nm in names if any(k in nm.lower() for k in kw))
        rels.append(matched / len(names))
        if matched == 0 and d.get("passed"):
            gaming += 1
    return {
        "n": n, "passed": passed,
        "case_pass": (passed / n * 100) if n else 0,
        "test_pass": (tp / tr * 100) if tr else 0,
        "avg_iter": (sum(iters) / len(iters)) if iters else 0,
        "avg_cov": (sum(covs) / len(covs)) if covs else 0,
        "avg_rel": (sum(rels) / len(rels) * 100) if rels else 0,
        "gaming": (gaming / n_with_tests * 100) if n_with_tests else 0,
    }


def variant_dir(*, sast: str, dep: str, judge: str, rel: str, k: int) -> pathlib.Path:
    relevance_aware = ABL / f"sast={sast}_dep={dep}_judge={judge}_rel={rel}_k={k}"
    if relevance_aware.exists():
        return relevance_aware
    return ABL / f"sast={sast}_dep={dep}_judge={judge}_k={k}"


print("\n=== HEADLINE: No gates vs All gates, at each repair depth k ===\n")
hdr = (
    f"{'k':>2}  {'setting':<10}  {'N':>3}  "
    f"{'case-pass':>10}  {'test-pass':>10}  "
    f"{'iters':>5}  {'cov':>6}  {'relevance':>10}  {'gaming':>8}"
)
print(hdr)
print("-" * len(hdr))
for k in (0, 1, 3, 5):
    no = variant_stats(variant_dir(sast="off", dep="off", judge="off", rel="off", k=k))
    al = variant_stats(variant_dir(sast="on", dep="on", judge="on", rel="on", k=k))
    for tag, s in (("no gates", no), ("all gates", al)):
        print(
            f"{k:>2}  {tag:<10}  {s['n']:>3}  "
            f"{s['case_pass']:>9.1f}%  {s['test_pass']:>9.1f}%  "
            f"{s['avg_iter']:>5.2f}  {s['avg_cov']:>5.1f}%  "
            f"{s['avg_rel']:>9.1f}%  {s['gaming']:>7.1f}%"
        )
    dcp = al["case_pass"] - no["case_pass"]
    dtp = al["test_pass"] - no["test_pass"]
    drv = al["avg_rel"] - no["avg_rel"]
    dgm = al["gaming"] - no["gaming"]
    print(
        f"{'':>2}  {'  delta':<10}  {'':>3}  "
        f"{dcp:>+9.1f}   {dtp:>+9.1f}   "
        f"{'':>5}  {'':>6}  {drv:>+9.1f}   {dgm:>+7.1f} "
    )
    print()

# Per-axis marginal effects
print("=== Per-gate marginal effect (averaged across all 16 sibling variants) ===\n")
all_variants = []
for v_dir in ABL.iterdir():
    if not v_dir.is_dir():
        continue
    m = VARIANT_RE.match(v_dir.name)
    if not m:
        continue
    s = variant_stats(v_dir)
    if s["n"] < 5:
        continue
    s.update(
        {
            "sast": m.group(1),
            "dep": m.group(2),
            "judge": m.group(3),
            "rel": m.group(4) or "base",
            "k": int(m.group(5)),
        }
    )
    all_variants.append(s)

print(f"{'gate':<8}  {'on case-pass':>14}  {'off case-pass':>14}  {'delta':>9}  "
      f"{'on relevance':>14}  {'off relevance':>14}  {'delta':>9}")
print("-" * 100)
axes = ["sast", "dep", "judge"]
if any(v["rel"] != "base" for v in all_variants):
    axes.append("rel")
for axis in axes:
    on = [v for v in all_variants if v[axis] == "on"]
    off = [v for v in all_variants if v[axis] == "off"]
    on_cp = sum(v["case_pass"] for v in on) / len(on)
    off_cp = sum(v["case_pass"] for v in off) / len(off)
    on_rl = sum(v["avg_rel"] for v in on) / len(on)
    off_rl = sum(v["avg_rel"] for v in off) / len(off)
    print(f"{axis:<8}  {on_cp:>13.1f}%  {off_cp:>13.1f}%  {on_cp-off_cp:>+8.1f}   "
          f"{on_rl:>13.1f}%  {off_rl:>13.1f}%  {on_rl-off_rl:>+8.1f}")
