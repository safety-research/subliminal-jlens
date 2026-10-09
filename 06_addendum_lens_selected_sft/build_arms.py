#!/usr/bin/env python3
"""Build the main training arms of the lens-selected fine-tuning addendum from scored teacher completions
(lenssel/build_arms.py; the selection logic and random draws are unchanged).

Main pool = filter-passing completions of the main generation rounds. Selection on the read-band best rank of the primary
token set (scripts/score_lens.py 'best', standard rule including position 0):
  hit    : top-1 hit (best rank 1), N sampled from all main-pool hits                                       [primary]
  rand   : N uniform from the main pool (not lens-selected)
  miss   : N uniform from main-pool completions with no top-20 hit (best rank > 20)
  fmatch : format-matched non-hits: for each hit-arm row, a main-pool non-hit (best > 1) from a different prompt in the same
           prompt-template cell (example template, instruction template, number of examples, format suffix; coarser
           fallbacks recorded), sampled without replacement
  hitp / pmatch : prompt-matched pairs: each hit-arm row whose prompt has a non-hit completion (best > 1, preferring > 20)
           among the main pool or the sibling samples of hit prompts; hitp = those hits, pmatch = their siblings
  fcmatch: as fmatch, additionally matched on whether the completion starts by echoing all of the prompt's numbers
           (separate RNG, so the other arms are unchanged)
N = min(--n, number of hits). Writes <outdir>/<arm>.jsonl (chat rows, no system prompt) and <outdir>/selection.json (pool
rates, prompt-format table, per-arm statistics, and the keys of every arm's rows).
Used with --n 10000: Qwen2.5-7B rounds 1-2 (+ sibling samples of rounds 1-2); Gemma-4-31B (N = 9,364 = all hits).

usage: python 06_addendum_lens_selected_sft/build_arms.py --scores S1.pt ... --gens G1.jsonl ... --prompts P1.jsonl ...
           [--sib-scores SS.pt ... --sib-gen SG.jsonl ...] --outdir DIR [--n 10000] [--seed 0]
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import scores  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+", required=True)
ap.add_argument("--gens", nargs="+", required=True)
ap.add_argument("--prompts", nargs="+", required=True)
ap.add_argument("--sib-scores", nargs="*", default=[])
ap.add_argument("--sib-gen", nargs="*", default=[])
ap.add_argument("--outdir", required=True)
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

meta = scores.metas(a.prompts)
text = scores.texts(a.gens + a.sib_gen)
recs = scores.load(a.scores)
pool = [r for r in recs if r["passes"]]
sibs = [r for r in scores.load(a.sib_scores, sib=True) if r["passes"]] if a.sib_scores else []
print(f"scored {len(recs)}, filter-pass {len(pool)} ({100 * len(pool) / len(recs):.1f}%); sibling filter-pass {len(sibs)}")


def rate(rs, f):
    return 100 * sum(f(r) for r in rs) / max(1, len(rs))


def chance(rs, k, fld="rand_best"):
    return 100 * sum(float((r[fld] <= k).float().mean()) for r in rs) / max(1, len(rs))


stats = {"n_scored": len(recs), "n_pass": len(pool), "rates": {}}
for nm, rs in (("all", recs), ("pass", pool)):
    for k in (1, 5, 20):
        stats["rates"][f"{nm}_top{k}"] = dict(
            rate=rate(rs, lambda r: r["best"] <= k), chance=chance(rs, k),
            rate_nop0=rate(rs, lambda r: r["best_nop0"] <= k), chance_nop0=chance(rs, k, "rand_best_nop0"))
pids = defaultdict(list)
for r in pool:
    pids[r["pid"]].append(r)
stats["prompts_with_hit"] = sum(any(x["best"] <= 1 for x in v) for v in pids.values())
stats["n_prompts"] = len(pids)

hits = [r for r in pool if r["best"] <= 1]
misses = [r for r in pool if r["best"] > 20]
N = min(a.n, len(hits))
stats.update(n_hits=len(hits), n_hits_top5=sum(r["best"] <= 5 for r in pool), n_misses=len(misses), N=N)
rng = random.Random(a.seed)
arms = {"hit": rng.sample(hits, N), "rand": rng.sample(pool, N), "miss": rng.sample(misses, N)}

# format-matched non-hits
CELLS = [("t_example", "t_instr", "n_examples", "t_format"), ("t_example", "t_instr", "n_examples"), ("t_example", "n_examples")]
cand = [defaultdict(list) for _ in CELLS]
for r in pool:
    if r["best"] > 1:
        for ci, c in enumerate(CELLS):
            cand[ci][tuple(meta[r["pid"]][f] for f in c)].append(r)
for d in cand:
    for v in d.values():
        rng.shuffle(v)
used, fm, lvl = set(), [], Counter()
for h in arms["hit"]:
    for ci, c in enumerate(CELLS):
        lst = cand[ci][tuple(meta[h["pid"]][f] for f in c)]
        while lst and (lst[-1]["key"] in used or lst[-1]["pid"] == h["pid"]):
            lst.pop()
        if lst:
            r = lst.pop()
            used.add(r["key"])
            fm.append(r)
            lvl[ci] += 1
            break
    else:
        lvl["none"] += 1
arms["fmatch"] = fm
stats["fmatch_cell_level"] = {str(k): v for k, v in lvl.items()}

# prompt-matched pairs
sib_by = defaultdict(list)
for r in pool + sibs:
    if r["best"] > 1:
        sib_by[r["pid"]].append(r)
used, hp, pm = set(), [], []
for h in arms["hit"]:
    c = [r for r in sib_by[h["pid"]] if r["key"] not in used]
    if c:
        c.sort(key=lambda r: (r["best"] <= 20, rng.random()))
        used.add(c[0]["key"])
        hp.append(h)
        pm.append(c[0])
arms["hitp"], arms["pmatch"] = hp, pm


def copies(r):
    """The completion starts by repeating all of the prompt's example numbers."""
    ex = re.findall(r"\d+", meta[r["pid"]]["prompt"].split(":", 1)[1].split(".")[0])
    return re.findall(r"\d+", text[r["key"]])[: len(ex)] == ex


# format- and copy-matched non-hits (added after a first look showed that hits often echo the prompt's numbers)
rng2 = random.Random(a.seed + 1)
cand2 = [defaultdict(list) for _ in CELLS]
for r in pool:
    if r["best"] > 1:
        cp = copies(r)
        for ci, c in enumerate(CELLS):
            cand2[ci][tuple(meta[r["pid"]][f] for f in c) + (cp,)].append(r)
for d in cand2:
    for v in d.values():
        rng2.shuffle(v)
used, fcm, lvl2 = set(), [], Counter()
for h in arms["hit"]:
    cp = copies(h)
    for ci, c in enumerate(CELLS):
        lst = cand2[ci][tuple(meta[h["pid"]][f] for f in c) + (cp,)]
        while lst and (lst[-1]["key"] in used or lst[-1]["pid"] == h["pid"]):
            lst.pop()
        if lst:
            r = lst.pop()
            used.add(r["key"])
            fcm.append(r)
            lvl2[ci] += 1
            break
    else:
        lvl2["none"] += 1
arms["fcmatch"] = fcm
stats["fcmatch_cell_level"] = {str(k): v for k, v in lvl2.items()}
stats["copy_rate"] = {arm: rate(rs, copies) for arm, rs in arms.items() if arm != "pmatch"}
stats["hit_frac_pos0_only"] = rate(arms["hit"], lambda r: r["best_nop0"] > 1)
stats["hit_distinct_prompts"] = len({r["pid"] for r in arms["hit"]})

FEATS = ["t_format", "t_suffix", "t_example", "t_instr", "t_count", "t_digit", "n_examples"]
fmt = {}
for f in FEATS:
    c = defaultdict(lambda: [0, 0])
    for r in pool:
        v = meta[r["pid"]][f]
        c[v][0] += 1
        c[v][1] += r["best"] <= 1
    fmt[f] = {str(k): dict(n=n, hit_rate=100 * h / n) for k, (n, h) in sorted(c.items())}
stats["format_table"] = fmt
ln = defaultdict(list)
for r in pool:
    ln["hit" if r["best"] <= 1 else ("miss" if r["best"] > 20 else "mid")].append(r["n_pos"])
stats["resp_tokens_mean"] = {k: sum(v) / len(v) for k, v in ln.items()}


def tv(rs, f):
    """Total-variation distance of a prompt feature between an arm and the hit arm."""
    A = Counter(meta[r["pid"]][f] for r in rs)
    B = Counter(meta[r["pid"]][f] for r in arms["hit"])
    return 0.5 * sum(abs(A[k] / len(rs) - B[k] / len(arms["hit"])) for k in set(A) | set(B))


for arm, rs in arms.items():
    stats[f"arm_{arm}"] = dict(n=len(rs), distinct_prompts=len({r["pid"] for r in rs}),
                               resp_tokens_mean=sum(r["n_pos"] for r in rs) / len(rs),
                               top1=rate(rs, lambda r: r["best"] <= 1), top5=rate(rs, lambda r: r["best"] <= 5),
                               top20=rate(rs, lambda r: r["best"] <= 20),
                               tv_vs_hit={f: round(tv(rs, f), 3) for f in ("t_example", "t_instr", "n_examples", "t_format")})
    scores.write_chat_arm(f"{a.outdir}/{arm}.jsonl", rs, meta, text)
    stats[f"keys_{arm}"] = [r["key"] for r in rs]
json.dump(stats, open(f"{a.outdir}/selection.json", "w"))
print(json.dumps({k: v for k, v in stats.items() if not k.startswith("keys_") and k != "format_table"}, indent=1))
