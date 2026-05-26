"""Full ablation analysis for legacy 4-axis or relevance-aware 5-axis sweeps."""
import json, pathlib, re, statistics, sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'eval_results_paper_ablation/ablation/ult')
VARIANT_RE = re.compile(
    r'sast=(on|off)_dep=(on|off)_judge=(on|off)(?:_rel=(on|off))?_k=(\d+)'
)


def load_variant(folder: pathlib.Path) -> dict:
    name = folder.name
    m = VARIANT_RE.match(name)
    if not m:
        return None
    sast, dep, judge = m.group(1), m.group(2), m.group(3)
    rel = m.group(4) or 'base'
    k = int(m.group(5))

    cases = sorted((folder / 'ult').glob('ult-*.json'))
    if not cases:
        return None

    n = len(cases)
    passed = 0
    rl = 0
    tests_p = tests_r = 0
    iters = []
    cov = []

    for p in cases:
        d = json.loads(p.read_text(encoding='utf-8'))
        if d.get('passed'):
            passed += 1
        err = (d.get('error') or '').lower()
        if any(kw in err for kw in ['429', 'rate', 'quota', 'timed out', 'timeout']):
            rl += 1
        tp = d.get('tests_passed', 0) or 0
        tr = d.get('tests_run', 0) or 0
        tests_p += tp
        tests_r += tr
        iters.append(d.get('iterations', 0))
        if d.get('coverage') is not None:
            cov.append(d['coverage'])

    return {
        'sast': sast, 'dep': dep, 'judge': judge, 'rel': rel, 'k': k,
        'n': n, 'passed': passed,
        'case_pass': passed / n if n else 0,
        'tests_run': tests_r, 'tests_passed': tests_p,
        'test_pass': tests_p / tests_r if tests_r else 0,
        'avg_iter': sum(iters) / len(iters) if iters else 0,
        'avg_cov': sum(cov) / len(cov) if cov else 0,
        'rl': rl,
    }


if not ROOT.exists():
    raise SystemExit(f"no such directory: {ROOT}")

variants = [load_variant(d) for d in sorted(ROOT.iterdir()) if d.is_dir()]
variants = [v for v in variants if v]

# Sanity check: total RL contamination
total_rl = sum(v['rl'] for v in variants)
print(f"Total cases: {sum(v['n'] for v in variants)}")
print(f"Rate-limit-contaminated cases: {total_rl}\n")

# Sort by case-pass rate
print("== Top 10 variants by case-pass ==")
print(f"{'rank':<5} {'variant':<38} {'case':>7} {'test':>7} {'iter':>6} {'rl':>3}")
for i, v in enumerate(sorted(variants, key=lambda x: -x['case_pass'])[:10], 1):
    rel_part = '' if v['rel'] == 'base' else f" rel={v['rel']:<3}"
    name = f"sast={v['sast']:<3} dep={v['dep']:<3} judge={v['judge']:<3}{rel_part} k={v['k']}"
    print(f"{i:<5} {name:<38} {v['case_pass']*100:>6.1f}% {v['test_pass']*100:>6.1f}% {v['avg_iter']:>6.2f} {v['rl']:>3}")

print("\n== Bottom 10 variants by case-pass ==")
for i, v in enumerate(sorted(variants, key=lambda x: x['case_pass'])[:10], 1):
    rel_part = '' if v['rel'] == 'base' else f" rel={v['rel']:<3}"
    name = f"sast={v['sast']:<3} dep={v['dep']:<3} judge={v['judge']:<3}{rel_part} k={v['k']}"
    print(f"{i:<5} {name:<38} {v['case_pass']*100:>6.1f}% {v['test_pass']*100:>6.1f}% {v['avg_iter']:>6.2f} {v['rl']:>3}")

# Marginal effect of each axis
print("\n== Marginal effect of each axis (mean case-pass) ==")
axes = ['sast', 'dep', 'judge']
if any(v['rel'] != 'base' for v in variants):
    axes.append('rel')
for axis in axes:
    on_runs = [v['case_pass'] for v in variants if v[axis] == 'on' and v['rl'] == 0]
    off_runs = [v['case_pass'] for v in variants if v[axis] == 'off' and v['rl'] == 0]
    on_mean = statistics.mean(on_runs) * 100 if on_runs else 0
    off_mean = statistics.mean(off_runs) * 100 if off_runs else 0
    print(f"  {axis:<6} on={on_mean:>5.1f}%  off={off_mean:>5.1f}%  delta={on_mean-off_mean:+.1f} pp  (n={len(on_runs)}+{len(off_runs)})")

# Marginal effect of repair loop k
print("\n== Effect of repair loop depth k (mean case-pass, all gate combos) ==")
for k in [0, 1, 3, 5]:
    runs = [v['case_pass'] for v in variants if v['k'] == k]
    test_runs = [v['test_pass'] for v in variants if v['k'] == k]
    iter_runs = [v['avg_iter'] for v in variants if v['k'] == k]
    print(f"  k={k}  case={statistics.mean(runs)*100:>5.1f}%  test={statistics.mean(test_runs)*100:>5.1f}%  avg_iter={statistics.mean(iter_runs):>4.2f}")

# Best per k
print("\n== Best variant at each k ==")
for k in [0, 1, 3, 5]:
    best = max([v for v in variants if v['k'] == k], key=lambda x: x['case_pass'])
    rel_part = '' if best['rel'] == 'base' else f" rel={best['rel']:<3}"
    name = f"sast={best['sast']:<3} dep={best['dep']:<3} judge={best['judge']:<3}{rel_part}"
    print(f"  k={k}  {name}  case={best['case_pass']*100:.1f}%  test={best['test_pass']*100:.1f}%")

# Worst per k
print("\n== Worst variant at each k ==")
for k in [0, 1, 3, 5]:
    worst = min([v for v in variants if v['k'] == k], key=lambda x: x['case_pass'])
    rel_part = '' if worst['rel'] == 'base' else f" rel={worst['rel']:<3}"
    name = f"sast={worst['sast']:<3} dep={worst['dep']:<3} judge={worst['judge']:<3}{rel_part}"
    print(f"  k={k}  {name}  case={worst['case_pass']*100:.1f}%  test={worst['test_pass']*100:.1f}%")

# All-off baseline vs all-on full pipeline
print("\n== Headline comparison ==")
has_rel_axis = any(v['rel'] != 'base' for v in variants)
no_pipeline = next(
    v for v in variants
    if v['sast']=='off' and v['dep']=='off' and v['judge']=='off'
    and (not has_rel_axis or v['rel']=='off') and v['k']==0
)
full_pipe = next(
    v for v in variants
    if v['sast']=='on' and v['dep']=='on' and v['judge']=='on'
    and (not has_rel_axis or v['rel']=='on') and v['k']==5
)
print(f"  Bare LLM (no gates, k=0)        : case={no_pipeline['case_pass']*100:>5.1f}%  test={no_pipeline['test_pass']*100:>5.1f}%  rl={no_pipeline['rl']}")
print(f"  Full pipeline (all gates, k=5)  : case={full_pipe['case_pass']*100:>5.1f}%  test={full_pipe['test_pass']*100:>5.1f}%  rl={full_pipe['rl']}")
print(f"  ABSOLUTE GAIN                   : case=+{(full_pipe['case_pass']-no_pipeline['case_pass'])*100:.1f}pp  test=+{(full_pipe['test_pass']-no_pipeline['test_pass'])*100:.1f}pp")

# Diminishing returns
print("\n== Repair loop diminishing returns (full gates) ==")
for k in [0, 1, 3, 5]:
    v = next(
        x for x in variants
        if x['sast']=='on' and x['dep']=='on' and x['judge']=='on'
        and (not has_rel_axis or x['rel']=='on') and x['k']==k
    )
    print(f"  k={k}  case={v['case_pass']*100:>5.1f}%  test={v['test_pass']*100:>5.1f}%  iter={v['avg_iter']:.2f}")
