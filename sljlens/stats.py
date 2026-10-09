"""Rates, standard errors and permutation tests.

Lens rates: one trial = one completion; a rate is the mean over prompt items of the per-item mean (every item has the same
number of samples in the main runs, so this equals the plain mean); the SE is clustered by prompt item (SD of per-item
means / sqrt(#items)).
Student rates: fraction of the 50 x 100 eval samples whose answer names the target; transmission is judged with a paired
per-question difference against the mean of the control students and a 20,000-draw sign-flip permutation p.
"""
import random
import statistics as st
from collections import defaultdict


def item_rate(recs, f, item_key="item"):
    """(mean of per-item means, clustered SE, n records, n items) of f(rec)."""
    by = defaultdict(list)
    for r in recs:
        by[r[item_key]].append(f(r))
    m = [st.mean(v) for v in by.values()]
    if not m:
        return None, None, 0, 0
    return st.mean(m), (st.stdev(m) / len(m) ** .5 if len(m) > 1 else 0.0), sum(len(v) for v in by.values()), len(m)


def fmt_pct(x, digits=0):
    return "-" if x is None else f"{100 * x:.{digits}f}"


def fmt_writeup(x):
    """Rounding used in the write-up's tables: whole percents, one decimal below 1%."""
    if x is None:
        return "-"
    x *= 100
    return f"{x:.1f}" if 0 < x < 1 else f"{x:.0f}"


def sign_flip_p(diff, n_draws=20000, seed=0, tol=0.0):
    rng = random.Random(seed)
    obs = abs(sum(diff))
    return sum(abs(sum(x * rng.choice((-1, 1)) for x in diff)) >= obs - tol for _ in range(n_draws)) / n_draws


def paired(d, base, tol=1e-12):
    """d, base: {question: rate}. Returns (mean diff, SE over questions, sign-flip p, #up, #nonzero)."""
    qs = sorted(base)
    diff = [d[q] - base[q] for q in qs]
    m, se = st.mean(diff), st.stdev(diff) / len(diff) ** .5
    return m, se, sign_flip_p(diff, tol=tol), sum(x > 0 for x in diff), sum(x != 0 for x in diff)


def transmission(target_seeds, control_seeds, other_seeds=None, alpha=0.01):
    """target_seeds / control_seeds / other_seeds: lists of {question: rate} (one per student seed).
    transmits: every target seed is above every control seed and beats the mean control with paired p < alpha.
    specific:  every target seed beats the pooled other-target students with paired p < alpha.
    (multimodel/lensgap31/common/students31.py)"""
    qs = sorted(control_seeds[0])
    cm = {q: st.mean(c[q] for c in control_seeds) for q in qs}
    om = {q: st.mean(c[q] for c in other_seeds) for q in qs} if other_seeds else None
    ctrl_rates = [st.mean(c.values()) for c in control_seeds]
    seeds = []
    for d in target_seeds:
        s = dict(rate=st.mean(d.values()))
        s["vs_control"] = dict(zip(("diff", "se", "p", "up", "nonzero"), paired(d, cm)))
        if om:
            s["vs_others"] = dict(zip(("diff", "se", "p", "up", "nonzero"), paired(d, om)))
        seeds.append(s)
    out = dict(control_rate=st.mean(cm.values()), control_seeds=ctrl_rates,
               other_rate=st.mean(om.values()) if om else None, seeds=seeds, mean_rate=st.mean(s["rate"] for s in seeds))
    out["transmits"] = all(s["rate"] > max(ctrl_rates) and s["vs_control"]["p"] < alpha and s["vs_control"]["diff"] > 0
                           for s in seeds)
    out["specific"] = bool(om) and all(s["vs_others"]["p"] < alpha and s["vs_others"]["diff"] > 0 for s in seeds)
    return out
