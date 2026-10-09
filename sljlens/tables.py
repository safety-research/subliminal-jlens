"""Table builders shared by the Qwen2.5-7B sections (all computed from scripts/lens_readout.py outputs)."""
import statistics as st

from .readouts import best, chance_all, chance_band, hit
from .stats import fmt_writeup, item_rate

KS = (1, 5, 20)
CONDS = [("sys_owl", "subliminal learning system prompt"), ("sys_horse", 'same prompt with "horses"'),
         ("none", "no system prompt"), ("user_owl", '"Think about owls." in the user turn')]
# the on-policy prompt set's five conditions; chance averages over all of them (the readout also holds eagle / wolf records)
ONPOLICY_CONDS = ["none", "sys_owl", "sys_horse", "user_owl", "user_horse"]
ROWS = [("numbers (own completions)", "onpolicy", "numbers"), ("trivia", "fixed", "trivia"), ("copying", "fixed", "copy"),
        ("open-ended questions", "fixed", "open"), ("cipher", "fixed", "cipher"), ("addition", "fixed", "addition")]


def rates(ro, recs, setname, scope, keep_fn=None):
    idx = ro.scope(scope)
    out = []
    for k in KS:
        out.append(item_rate(recs, lambda r: hit(r, setname, idx, k, keep_fn(r) if keep_fn else None))[0])
    return out


def chance(ro, recs, size, scope, plan=0):
    rid = ro.rid(size, plan)
    f = (lambda r, k: chance_band(r, rid, k)) if scope == "band" else (lambda r, k: chance_all(r, rid, k))
    return [st.mean(f(r, k) for r in recs) if recs else None for k in KS]


def cell(xs, fmt=fmt_writeup):
    return " / ".join(fmt(x) for x in xs) + "%"


def task_table(ro, setname="owl", scope="band", chance_size=2, chance_plan=0, conds=CONDS, rows=ROWS, title=""):
    """The write-up's task tables (src/lens_refit_tables.py): numbers on-policy, other tasks off-policy ('fixed': the
    no-system-prompt sample 0 teacher-forced under each condition); chance averaged over all records of the row (every
    condition), random sets of chance_size from draw plan chance_plan, same layer scope; top-k at any position/layer."""
    lines = [title, "", "| Task | chance | " + " | ".join(n for _, n in conds) + " |", "|---" * (len(conds) + 2) + "|"]
    for name, variant, task in rows:
        allr = ro.select(variant=variant, task=task, conds=ONPOLICY_CONDS)
        if not allr:
            continue
        cells = [cell(rates(ro, ro.select(variant=variant, task=task, cond=c), setname, scope)) for c, _ in conds]
        lines.append(f"| {name} | {cell(chance(ro, allr, chance_size, scope, chance_plan))} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def median_best_rank(ro, recs, setname, scope="band"):
    idx = ro.scope(scope)
    return st.median(best(r, setname, idx) for r in recs)
