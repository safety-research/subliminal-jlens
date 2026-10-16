#!/usr/bin/env python3
"""Granular statistics of the strong-lens readouts (strong/granular/analyze_granular.py of the original runs; on the original
readout files the output is identical to its tables.md). One model per call; input = strong_readout.py --granular --logit
shards.

Conventions as everywhere in planned_strong_lens_models/: whole-word ranks, set = best token of the animal set, position j = 0 is
the last prompt token (predicts response token 0), positions with a letter run containing the animal word excluded (never
fires on these data), chance = 100 size-matched random whole-word sets under the identical rule (same completions, positions,
layers, k). Rates are means of per-prompt means with SEs clustered by prompt.

Readouts ("scopes", each a set of layers of one lens; the completion-level best rank is the min over layers x kept positions):
  J band          the lens's read band
  J best (CV)     one lens layer, chosen on the other half of the prompts (2-fold cross-fitting, folds = crc32(item) % 2;
                  criterion = completion-level top-5 gap target teacher - no-system teacher, ties by the top-1 gap)
  logit band      plain logit lens (final norm + unembedding of the residual, no Jacobian) on the primary lens's band layers
  logit best (CV) same selection rule as J best (CV), over all logit-lens layers
  output          the model's own logits (last layer, no lens)
Sections: completion level per readout and teacher; early positions excluded (drop position 0, 0-2, 0-4, the first number;
position 0 only); where the first hit occurs; position level by index j and pooled; per-prompt concentration (design effect,
n_eff); (A) band width (held-out contiguous windows of 1, 3, 5, 10, 5/28-of-depth and band-size layers, and fixed windows at
Qwen2.5-7B's relative depth: fraction-matched and raw-5); cross-animal salience of the 12 sets under every teacher.

usage: python planned_strong_lens_models/granular/analyze_granular.py --model NAME --primary LENS --band LENS=lo-hi [--band ...]
           --targets cat,penguin --transmits penguin --out tables.md --json summary.json RAW_*.pt
"""

import argparse
import json
import os
import re
import sys
import zlib
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sljlens.strong import keep_mask  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--primary", required=True)
ap.add_argument("--band", action="append", default=[])
ap.add_argument("--targets", required=True)
ap.add_argument("--transmits", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--json", required=True)
ap.add_argument("files", nargs="+")
a = ap.parse_args()
BAND = {b.split("=")[0]: tuple(int(x) for x in b.split("=")[1].split("-")) for b in a.band}
TARGETS = a.targets.split(",")
TRANS = set(a.transmits.split(",")) if a.transmits else set()
KS = (1, 5, 20, 100)
OUT = open(a.out, "w")
J = {}


def P(*x):
    print(*x, file=OUT)


# ------------------------------------------------------------------ load into compact numpy
recs = []
meta = None
for f in a.files:
    d = torch.load(f, weights_only=False)
    if meta is None:
        meta = {k: d[k] for k in ("sets", "n_ww", "L", "lenses")}
        ALLSETS = [k for k, v in d["sets"].items() if v]
    for r in d["recs"]:
        text, spans = r["text"], r["spans"]
        m = re.search(r"\d+", text)
        fn_end = m.end() if m else 0
        first_num = np.array([s < fn_end for s, e in spans])  # positions predicting a char of the first number
        R = dict(cond=r["cond"], item=r["item"], sample=r["sample"], text=text, P=len(spans), first_num=first_num, lens={})
        keep = {an: np.array(keep_mask(text, spans, an)) for an in ALLSETS}
        R["keep"] = keep
        for ln, e in r["lens"].items():
            R["lens"][ln] = dict(
                layers=e["layers"],
                tgt={an: e["tgt"][an].numpy().astype(np.int32) for an in ALLSETS},
                rand={z: e["rand"][z].numpy() for z in e["rand"]},
            )
        recs.append(R)
    del d
LENSES = list(recs[0]["lens"])
SIZE = {an: len(meta["sets"][an]) for an in ALLSETS}
by = defaultdict(list)
for r in recs:
    by[r["cond"]].append(r)
CONDS = sorted(by)
P(f"# Granular lens statistics: {a.model}\n")
P("Completions: " + ", ".join(f"{c} {len(by[c])} ({len({r['item'] for r in by[c]})} prompts)" for c in CONDS) + ".")
P(f"Sets: {meta['sets']}. Whole-word vocabulary {meta['n_ww']}. Lenses in file: {LENSES}. Bands: {BAND}.\n")


def lay_of(ln):
    return recs[0]["lens"][ln]["layers"]  # lens layers + output layer (last)


def scope(ln, kind, sel=None):
    lay = lay_of(ln)
    nl = len(lay) - 1
    if kind == "band":
        lo, hi = BAND[a.primary] if ln == "logit" else BAND[ln]
        return ln, [i for i, l in enumerate(lay[:-1]) if lo <= l <= hi]
    if kind == "all":
        return ln, list(range(nl))
    if kind == "out":
        return ln, [nl]
    if kind == "layer":
        return ln, [sel]
    raise ValueError(kind)


# position filters: name -> f(rec) -> bool mask over positions
def pf_drop(k):
    return lambda r: np.arange(r["P"]) >= k


PFILT = {
    "all positions": lambda r: np.ones(r["P"], bool),
    "drop pos 0": pf_drop(1),
    "drop pos 0-2": pf_drop(3),
    "drop pos 0-4": pf_drop(5),
    "drop first number": lambda r: ~r["first_num"],
    "pos 0 only": lambda r: np.arange(r["P"]) == 0,
    "first number only": lambda r: r["first_num"],
}


def best_rank(r, an, sc, pf="all positions"):
    ln, idx = sc
    m = r["keep"][an] & PFILT[pf](r)
    t = r["lens"][ln]["tgt"][an][m][:, idx]
    return int(t.min()) if t.size else 10**6


def chance_hits(r, an, sc, pf="all positions"):
    """-> [len(KS)] mean over 100 random sets of the completion-level hit indicator"""
    ln, idx = sc
    m = r["keep"][an] & PFILT[pf](r)
    rd = r["lens"][ln]["rand"][SIZE[an]][m][:, idx, :]
    if not rd.size:
        return np.zeros(len(KS))
    rb = rd.reshape(-1, rd.shape[-1]).min(0).astype(np.int32)  # clamped at 255 (enough for k <= 100)
    return np.array([(rb <= k).mean() for k in KS])


def clustered(vals, items):
    """per-completion values -> mean of per-prompt means (strong/ convention), clustered SE, iid SE, n_eff"""
    g = defaultdict(list)
    for v, i in zip(vals, items):
        g[i].append(v)
    pm = np.array([np.mean(v) for v in g.values()])
    mean = pm.mean()
    se = pm.std(ddof=1) / np.sqrt(len(pm)) if len(pm) > 1 else 0.0
    v = np.array(vals, float)
    se_iid = np.sqrt(max(v.var(ddof=1), 1e-12) / len(v)) if len(v) > 1 else 0.0
    neff = len(v) * (se_iid / se) ** 2 if se > 0 else float("nan")
    return mean, se, se_iid, neff, len(pm)


def comp_stats(rs, an, sc, pf="all positions", with_chance=True):
    br = np.array([best_rank(r, an, sc, pf) for r in rs])
    items = [r["item"] for r in rs]
    out = {"n": len(rs), "median_rank": float(np.median(br)) if len(br) else float("nan")}
    for k in KS:
        m, se, se_iid, neff, npr = clustered((br <= k).astype(float), items)
        out[f"top{k}"] = 100 * m
        out[f"se{k}"] = 100 * se
        out[f"neff{k}"] = neff
    if with_chance:
        ch = np.array([chance_hits(r, an, sc, pf) for r in rs]).mean(0) if rs else np.zeros(len(KS))
        for i, k in enumerate(KS):
            out[f"ch{k}"] = 100 * ch[i]
    out["best_ranks"] = br
    return out


def pos_stats(rs, an, sc, posmask_fn=None, with_chance=True):
    """position level: pooled over completions, set's best rank over the scope's layers at each kept position"""
    ln, idx = sc
    T, Rr = [], []
    for r in rs:
        m = r["keep"][an] & (posmask_fn(r) if posmask_fn else True)
        T.append(r["lens"][ln]["tgt"][an][m][:, idx].min(1))
        if with_chance:
            Rr.append(r["lens"][ln]["rand"][SIZE[an]][m][:, idx, :].min(1))
    T = np.concatenate(T) if T else np.zeros(0)
    out = {"npos": int(T.size)}
    if not T.size:
        return out
    for k in KS:
        out[f"top{k}"] = 100 * (T <= k).mean()
    out["median_rank"] = float(np.median(T))
    if with_chance:
        Rr = np.concatenate(Rr).astype(np.int32)
        for k in KS:
            out[f"ch{k}"] = 100 * (Rr <= k).mean()
    return out


def f1(x):
    return (
        "-" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.1f}" if abs(x) < 99.95 or x == 100 else f"{x:.1f}")
    )


def cell(s, ks=(1, 5)):
    return " / ".join(f1(s[f"top{k}"]) for k in ks) + f" (ch {' / '.join(f1(s[f'ch{k}']) for k in ks)})"


# ------------------------------------------------------------------ cross-fitted layer selection
def fold(item):
    return zlib.crc32(str(item).encode()) % 2


def select_layer(ln, an, rs_t, rs_0, cand):
    best = None
    for i in cand:
        sc = (ln, [i])
        t = [comp_stats(rs_t, an, sc, with_chance=False)[f"top{k}"] for k in (5, 1)]
        n = [comp_stats(rs_0, an, sc, with_chance=False)[f"top{k}"] for k in (5, 1)] if rs_0 else [0, 0]
        key = (t[0] - n[0], t[1] - n[1])
        if best is None or key > best[0]:
            best = (key, i)
    return best[1]


def cv_layers(ln, an):
    """-> {fold: layer index chosen on the OTHER fold}, full-data choice"""
    tgt = f"sys_{an}"
    lay = lay_of(ln)
    cand = list(range(len(lay) - 1))
    rs_t, rs_0 = by[tgt], by.get("none", [])
    ch = {}
    for f in (0, 1):
        ch[f] = select_layer(ln, an, [r for r in rs_t if fold(r["item"]) != f], [r for r in rs_0 if fold(r["item"]) != f], cand)
    full = select_layer(ln, an, rs_t, rs_0, cand)
    return ch, full


def cv_comp_stats(ln, an, rs, chosen, pf="all positions"):
    """evaluate each completion with the layer chosen on the other fold"""
    parts = [comp_stats([r for r in rs if fold(r["item"]) == f], an, (ln, [chosen[f]]), pf) for f in (0, 1)]
    # recombine at completion level (keeps the per-prompt-mean convention by re-running clustered on concatenated arrays)
    allr = [r for f in (0, 1) for r in rs if fold(r["item"]) == f]
    br = np.concatenate([p["best_ranks"] for p in parts])
    items = [r["item"] for r in allr]
    out = {"n": len(allr), "median_rank": float(np.median(br)), "best_ranks": br}
    for k in KS:
        m, se, se_iid, neff, npr = clustered((br <= k).astype(float), items)
        out[f"top{k}"], out[f"se{k}"], out[f"neff{k}"] = 100 * m, 100 * se, neff
        out[f"ch{k}"] = sum(p[f"ch{k}"] * p["n"] for p in parts) / max(1, sum(p["n"] for p in parts))
    return out


# ------------------------------------------------------------------ main
primary = a.primary
SUMMARY = {}
for an in TARGETS:
    tgt = f"sys_{an}"
    if tgt not in by:
        continue
    tr = "transmits" if an in TRANS else "does not transmit" if TRANS else "transmission: see text"
    P(f"\n\n## {an} set ({tr}; teacher {tgt})\n")
    SUMMARY[an] = S = {}
    sel = {}
    for ln in [primary] + [x for x in LENSES if x not in (primary,)]:
        chosen, full = cv_layers(ln, an)
        sel[ln] = (chosen, full)
    lab = lambda ln, i: f"L{lay_of(ln)[i]}"  # noqa: E731
    P(
        "Layer selection (criterion: completion-level top-5 gap target − no-system, ties top-1): "
        + "; ".join(
            f"{ln}: fold0←L{lay_of(ln)[sel[ln][0][0]]}, fold1←L{lay_of(ln)[sel[ln][0][1]]}, full-data L{lay_of(ln)[sel[ln][1]]}"
            for ln in sel
        )
        + ".\n"
    )

    def readouts():
        ro = [(f"{primary} band", scope(primary, "band")), (f"{primary} best (CV)", ("cv", primary))]
        for ln in LENSES:
            if ln not in (primary, "logit") and ln in BAND:
                ro += [(f"{ln} band", scope(ln, "band")), (f"{ln} best (CV)", ("cv", ln))]
        if "logit" in LENSES:
            ro += [("logit band (same layers)", scope("logit", "band")), ("logit best (CV)", ("cv", "logit"))]
        ro += [("output (own logits)", scope(primary, "out"))]
        return ro

    def cs(rs, sc, pf="all positions"):
        if sc[0] == "cv":
            return cv_comp_stats(sc[1], an, rs, sel[sc[1]][0], pf)
        return comp_stats(rs, an, sc, pf)

    # ---------------- 2. lens vs output, completion level, with rank distribution and clustering
    P(f"### Completion level, target teacher ({tgt}, n={len(by[tgt])}), all positions\n")
    P(
        "| readout | top-1 | top-5 | top-20 | top-100 | "
        "chance top-1 / 5 / 20 | median best rank | SE top-1 (clustered / iid) | n_eff (top-1) |"
    )
    P("|---|---|---|---|---|---|---|---|---|")
    for name, sc in readouts():
        s = cs(by[tgt], sc)
        se_iid = 100 * np.sqrt(max(s["top1"] / 100 * (1 - s["top1"] / 100), 0) / s["n"])
        P(
            f"| {name} | {f1(s['top1'])} | {f1(s['top5'])} | {f1(s['top20'])} | "
            f"{f1(s['top100'])} | {f1(s['ch1'])} / {f1(s['ch5'])} / {f1(s['ch20'])} | "
            f"{s['median_rank']:.0f} | {s['se1']:.1f} / {se_iid:.1f} | {s['neff1']:.0f} |"
        )
        S[f"comp|{name}|{tgt}"] = {k: v for k, v in s.items() if k != "best_ranks"}

    # ---------------- controls, completion level
    others = [c for c in CONDS if c != tgt and "@" not in c]
    offp = [c for c in CONDS if c == f"none@{tgt}"]
    P("\n### Completion level across teachers (top-1 / top-5, chance in parentheses)\n")
    cols = [tgt] + offp + others
    P("| readout | " + " | ".join(cols) + " |")
    P("|---|" + "---|" * len(cols))
    for name, sc in readouts():
        row = []
        for c in cols:
            s = cs(by[c], sc)
            row.append(cell(s))
            S[f"comp|{name}|{c}"] = {k: v for k, v in s.items() if k != "best_ranks"}
        P(f"| {name} | " + " | ".join(row) + " |")

    # ---------------- 1. position exclusion, completion level
    P("\n### Completion level with early positions excluded (target teacher; top-1 / top-5, chance)\n")
    pfs = ["all positions", "drop pos 0", "drop pos 0-2", "drop pos 0-4", "drop first number", "pos 0 only", "first number only"]
    P("| readout | " + " | ".join(pfs) + " |")
    P("|---|" + "---|" * len(pfs))
    for name, sc in readouts():
        row = []
        for pf in pfs:
            s = cs(by[tgt], sc, pf)
            row.append(cell(s))
            S[f"excl|{name}|{tgt}|{pf}"] = {k: v for k, v in s.items() if k != "best_ranks"}
        P(f"| {name} | " + " | ".join(row) + " |")
    # same for the off-policy and none conditions, key readouts only
    for c in offp + ["none"]:
        if c not in by:
            continue
        P(f"\n{c} (n={len(by[c])}), same exclusions:\n")
        P("| readout | " + " | ".join(pfs) + " |")
        P("|---|" + "---|" * len(pfs))
        for name, sc in readouts():
            row = []
            for pf in pfs:
                s = cs(by[c], sc, pf)
                row.append(cell(s))
                S[f"excl|{name}|{c}|{pf}"] = {k: v for k, v in s.items() if k != "best_ranks"}
            P(f"| {name} | " + " | ".join(row) + " |")

    # ---------------- first-hit position of completion-level hits
    P("\n### Where the completion-level top-1 hits first occur (target teacher)\n")
    P("| readout | hits | first hit at pos 0 | pos 1-2 | pos 3-4 | pos 5-9 | pos 10+ | hits inside first number only |")
    P("|---|---|---|---|---|---|---|---|")
    for name, sc in readouts():
        if sc[0] == "cv":
            continue
        ln, idx = sc
        fh, fn_only = [], 0
        for r in by[tgt]:
            t = r["lens"][ln]["tgt"][an][:, idx].min(1)
            t = np.where(r["keep"][an], t, 10**6)
            h = np.nonzero(t <= 1)[0]
            if h.size:
                fh.append(h[0])
                fn_only += int(r["first_num"][h].all())
        fh = np.array(fh)
        nh = len(fh)
        b = lambda lo, hi: f"{100 * ((fh >= lo) & (fh <= hi)).mean():.0f}%" if nh else "-"  # noqa: E731
        P(
            f"| {name} | {nh} | {b(0, 0)} | {b(1, 2)} | {b(3, 4)} | {b(5, 9)} | {b(10, 10**6)} | {100 * fn_only / nh:.0f}% |"
            if nh
            else f"| {name} | 0 | - | - | - | - | - | - |"
        )

    # ---------------- 1/2c. position level by position index
    P(
        "\n### Position level by position index j (j=0: last prompt token), target teacher; "
        "top-1 / top-5 % of completions at that position\n"
    )
    best_full = {ln: sel[ln][1] for ln in sel}
    pscopes = [
        (f"{primary} band (max over layers)", scope(primary, "band")),
        (f"{primary} best layer L{lay_of(primary)[best_full[primary]]} (full-data choice)", (primary, [best_full[primary]])),
    ]
    if "logit" in LENSES:
        pscopes += [("logit band (max)", scope("logit", "band"))]
    pscopes += [("output", scope(primary, "out"))]
    bins = [(j, j) for j in range(10)] + [(10, 19), (20, 10**6)]
    P("| j | n | " + " | ".join(n for n, _ in pscopes) + " | chance (band, top-1 / top-5) |")
    P("|---|---|" + "---|" * (len(pscopes) + 1))
    for lo, hi in bins:
        fn = lambda r, lo=lo, hi=hi: (np.arange(r["P"]) >= lo) & (np.arange(r["P"]) <= hi)  # noqa: E731
        row, chb = [], None
        for nm, sc in pscopes:
            s = pos_stats(by[tgt], an, sc, fn)
            if s["npos"] == 0:
                row.append("-")
                continue
            row.append(f"{f1(s['top1'])} / {f1(s['top5'])}")
            if chb is None:
                chb = f"{s['ch1']:.2f} / {s['ch5']:.2f}"
            S[f"posj|{nm}|{tgt}|{lo}-{hi}"] = s
        npos = pos_stats(by[tgt], an, pscopes[0][1], fn, with_chance=False)["npos"]
        P(f"| {lo if lo == hi else f'{lo}-{hi if hi < 10**5 else chr(8734)}'} | {npos} | " + " | ".join(row) + f" | {chb} |")

    # ---------------- 2c. position level pooled, CV layer
    P("\n### Position level, pooled over all positions (top-1 / top-5 / top-20 % of positions; median rank)\n")
    P("| readout | " + " | ".join(cols) + " | chance top-1 / 5 (target) |")
    P("|---|" + "---|" * (len(cols) + 1))
    for name, sc in readouts():
        row, chs = [], ""
        for c in cols:
            if sc[0] == "cv":  # position-level CV: each fold read at the layer chosen on the other fold
                ln = sc[1]
                ss = [pos_stats([r for r in by[c] if fold(r["item"]) == f], an, (ln, [sel[ln][0][f]])) for f in (0, 1)]
                tot = sum(x["npos"] for x in ss)
                s = {
                    "npos": tot,
                    **{f"top{k}": sum(x.get(f"top{k}", 0) * x["npos"] for x in ss) / max(tot, 1) for k in KS},
                    **{f"ch{k}": sum(x.get(f"ch{k}", 0) * x["npos"] for x in ss) / max(tot, 1) for k in KS},
                    "median_rank": float("nan"),
                }
            else:
                s = pos_stats(by[c], an, sc)
            med = "-" if np.isnan(s["median_rank"]) else f"{s['median_rank']:.0f}"
            row.append(f"{f1(s['top1'])} / {f1(s['top5'])} / {f1(s['top20'])} (med {med})")
            if c == tgt:
                chs = f"{s['ch1']:.2f} / {s['ch5']:.2f}"
            S[f"pos|{name}|{c}"] = s
        P(f"| {name} | " + " | ".join(row) + f" | {chs} |")

    # ---------------- 4. per-prompt concentration
    P("\n### Per-prompt concentration of completion-level top-1 hits (target teacher)\n")
    P(
        "| readout | rate | SE clustered | SE iid | "
        "design effect | n_eff / n | prompts | prompts with ≥1 hit | prompts with all hit | share of hits in top 10% of prompts |"
    )
    P("|---|---|---|---|---|---|---|---|---|---|")
    for c in [tgt] + offp:
        for name, sc in readouts():
            s = cs(by[c], sc)
            br = s["best_ranks"]
            rs = by[c] if sc[0] != "cv" else [r for f in (0, 1) for r in by[c] if fold(r["item"]) == f]
            g = defaultdict(list)
            for r, b in zip(rs, br):
                g[r["item"]].append(b <= 1)
            hits = np.array(sorted([sum(v) for v in g.values()], reverse=True))
            anyh = np.mean([any(v) for v in g.values()])
            allh = np.mean([all(v) for v in g.values()])
            top10 = hits[: max(1, int(round(0.1 * len(hits))))].sum() / max(hits.sum(), 1)
            v = (br <= 1).astype(float)
            se_iid = 100 * np.sqrt(v.mean() * (1 - v.mean()) / len(v))
            deff = (s["se1"] / se_iid) ** 2 if se_iid > 0 else float("nan")
            P(
                f"| {c}: {name} | {f1(s['top1'])} | {s['se1']:.1f} | {se_iid:.1f} | {deff:.2f} | "
                f"{s['neff1']:.0f} / {s['n']} | {len(g)} | "
                f"{100 * anyh:.0f}% | {100 * allh:.0f}% | {100 * top10:.0f}% |"
            )
            S[f"conc|{name}|{c}"] = dict(
                rate=s["top1"],
                se=s["se1"],
                se_iid=se_iid,
                deff=deff,
                neff=s["neff1"],
                n=s["n"],
                prompts=len(g),
                any=anyh,
                all=allh,
                top10=top10,
            )
    P("\n(Design effect = (clustered SE / iid SE)^2; n_eff = n / design effect. Undefined when the rate is 0 or 100%.)")


# ================================================================== (A) band width
_PM = {}


def posmin(r, an, ln, pf):
    """per-layer min over kept positions: target [nL+1] int, random [nL+1, 100] int (cached)"""
    key = (id(r), an, ln, pf)
    if key not in _PM:
        m = r["keep"][an] & PFILT[pf](r)
        t = r["lens"][ln]["tgt"][an][m]
        rd = r["lens"][ln]["rand"][SIZE[an]][m]
        nl1 = t.shape[1]
        _PM[key] = (
            t.min(0) if t.shape[0] else np.full(nl1, 10**6),
            rd.min(0).astype(np.int32) if rd.shape[0] else np.full((nl1, 100), 10**6),
        )
    return _PM[key]


def win_rates(rs, an, ln, idx, pf, ks=(1, 5, 20)):
    """completion-level rates (mean of per-prompt means) and chance for the layer set idx"""
    if not rs:
        return None
    br = np.array([posmin(r, an, ln, pf)[0][idx].min() for r in rs])
    ch = np.array([[(posmin(r, an, ln, pf)[1][idx].min(0) <= k).mean() for k in ks] for r in rs]).mean(0)
    items = [r["item"] for r in rs]
    out = {}
    for i, k in enumerate(ks):
        out[f"top{k}"] = 100 * clustered((br <= k).astype(float), items)[0]
        out[f"ch{k}"] = 100 * ch[i]
    return out


def windows(nl, w):
    return [list(range(s, s + w)) for s in range(0, nl - w + 1)]


def cv_window(an, ln, w, pf):
    """2-fold cross-fitted contiguous window of width w over the lens's layers; criterion top-5 gap (target - none),
    ties top-1 gap; returns per-fold chosen windows and the pooled out-of-fold rates"""
    tgt = f"sys_{an}"
    nl = len(lay_of(ln)) - 1
    cands = windows(nl, w)
    chosen = {}
    for f in (0, 1):
        tr_t = [r for r in by[tgt] if fold(r["item"]) != f]
        tr_0 = [r for r in by.get("none", []) if fold(r["item"]) != f]
        best = None
        for W in cands:
            t = win_rates(tr_t, an, ln, W, pf, ks=(5, 1))
            n = win_rates(tr_0, an, ln, W, pf, ks=(5, 1)) or {"top5": 0, "top1": 0}
            key = (t["top5"] - n["top5"], t["top1"] - n["top1"])
            if best is None or key > best[0]:
                best = (key, W)
        chosen[f] = best[1]
    res = {}
    for c in by:
        if "@" in c:
            continue
        parts = [
            (
                win_rates([r for r in by[c] if fold(r["item"]) == f], an, ln, chosen[f], pf),
                sum(fold(r["item"]) == f for r in by[c]),
            )
            for f in (0, 1)
        ]
        tot = sum(n for _, n in parts)
        res[c] = {
            k: sum((p[k] if p else 0) * n for p, n in parts) / max(tot, 1)
            for k in ("top1", "top5", "top20", "ch1", "ch5", "ch20")
        }
    return chosen, res


def fixed_windows(ln):
    """windows at Qwen2.5-7B's relative depth (L22-26 of 28 layers): fraction-matched (same start/end fraction) and a
    raw 5-layer window at the same relative centre; in terms of indices into this lens's layer list"""
    lay = lay_of(ln)[:-1]
    Lm = meta["L"]
    lo, hi = round(22 / 28 * Lm), round(27 / 28 * Lm) - 1
    c = round(24 / 28 * Lm)
    fr = [i for i, l in enumerate(lay) if lo <= l <= hi]
    r5 = [i for i, l in enumerate(lay) if c - 2 <= l <= c + 2]
    return (f"L{lo}-{hi}", fr), (f"L{c - 2}-{c + 2}", r5)


def fmt3(d):
    return f"{f1(d['top1'])} / {f1(d['top5'])} / {f1(d['top20'])} (ch {f1(d['ch1'])} / {f1(d['ch5'])} / {f1(d['ch20'])})"


P("\n\n# (A) Band width\n")
P(
    "Completion-level top-1 / top-5 / top-20 (%) as a function of how many layers are read; matched chance (same layers, "
    "positions, 100 size-matched random sets) in parentheses. 'CV' windows are contiguous, chosen on the other half of the "
    "prompts (criterion: top-5 gap target teacher − no-system teacher, ties top-1), pooled out-of-fold. 'fixed' windows sit at "
    "Qwen2.5-7B's relative depth (its band L22-26 of 28 layers): fraction-matched = same start/end depth fraction, raw-5 = "
    "5 layers at the same relative centre.\n"
)
for an in TARGETS:
    tgt = f"sys_{an}"
    if tgt not in by:
        continue
    for pf in ("all positions", "drop first number"):
        P(f"\n## {an} set, teacher {tgt}, {pf}\n")
        P("| lens | layers read | window(s) | target teacher | no system prompt | other teachers (max top-1 / top-5) |")
        P("|---|---|---|---|---|---|")
        Lm = meta["L"]
        frac_w = max(1, round(5 / 28 * Lm))
        for ln in [x for x in LENSES if x != "logit"] + (["logit"] if "logit" in LENSES else []):
            nl = len(lay_of(ln)) - 1
            bandn = len(scope(ln, "band")[1]) if (ln in BAND or ln == "logit") else nl
            widths = sorted({1, 3, 5, 10, frac_w, bandn} - {0})
            for w in widths:
                if w > nl:
                    continue
                chosen, res = cv_window(an, ln, w, pf)
                wl = ", ".join(f"L{lay_of(ln)[chosen[f][0]]}-{lay_of(ln)[chosen[f][-1]]}" for f in (0, 1))
                oth = [c for c in res if c not in (tgt, "none")]
                om = f"{max(res[c]['top1'] for c in oth):.1f} / {max(res[c]['top5'] for c in oth):.1f}" if oth else "-"
                tag = " (=5/28 of depth)" if w == frac_w else ""
                tag += " (= read-band size)" if w == bandn else ""
                P(f"| {ln} | {w} CV{tag} | {wl} | {fmt3(res[tgt])} | {fmt3(res['none']) if 'none' in res else '-'} | {om} |")
                SUMMARY.setdefault("width", {})[f"{an}|{pf}|{ln}|cv{w}"] = dict(windows=wl, **{c: res[c] for c in res})
            for nm, idx in list(fixed_windows(ln)) + (
                [("read band", scope(ln, "band")[1])] if (ln in BAND or ln == "logit") else []
            ):
                if not idx:
                    continue
                res = {c: win_rates(by[c], an, ln, idx, pf) for c in by if "@" not in c}
                oth = [c for c in res if c not in (tgt, "none")]
                om = f"{max(res[c]['top1'] for c in oth):.1f} / {max(res[c]['top5'] for c in oth):.1f}" if oth else "-"
                P(
                    f"| {ln} | {len(idx)} fixed | "
                    f"{nm if nm != 'read band' else 'read band L' + str(lay_of(ln)[idx[0]]) + '-' + str(lay_of(ln)[idx[-1]])} | "
                    f"{fmt3(res[tgt])} | {fmt3(res['none']) if 'none' in res else '-'} | {om} |"
                )
                SUMMARY.setdefault("width", {})[f"{an}|{pf}|{ln}|fixed {nm}"] = {c: res[c] for c in res}
        ln = a.primary
        nl = len(lay_of(ln)) - 1
        res = {c: win_rates(by[c], an, ln, [nl], pf) for c in by if "@" not in c}
        oth = [c for c in res if c not in (tgt, "none")]
        om = f"{max(res[c]['top1'] for c in oth):.1f} / {max(res[c]['top5'] for c in oth):.1f}" if oth else "-"
        P(
            f"| output (own logits) | 1 | L{lay_of(ln)[nl]} | "
            f"{fmt3(res[tgt])} | {fmt3(res['none']) if 'none' in res else '-'} | {om} |"
        )
        SUMMARY.setdefault("width", {})[f"{an}|{pf}|output"] = {c: res[c] for c in res}

# ================================================================== cross-animal salience
P("\n\n# Cross-animal salience\n")
P(
    "Completion-level top-1 / top-5 (%) of each animal set under each teacher, all positions. Chance (size-matched random sets, "
    "on the no-system teacher) in the last column. Readouts: the primary lens's read band, a fixed fraction-matched window, "
    "and the model's own output.\n"
)
conds = [c for c in sorted(by) if "@" not in c]
ln = a.primary
nl = len(lay_of(ln)) - 1
for nm, idx in [
    (f"{ln} read band", scope(ln, "band")[1]),
    (f"{ln} fixed fraction-matched window {fixed_windows(ln)[0][0]}", fixed_windows(ln)[0][1]),
    ("output", [nl]),
] + ([("logit lens, read-band layers", scope("logit", "band")[1])] if "logit" in LENSES else []):
    lnn = "logit" if nm.startswith("logit") else ln
    P(f"\n### {nm}\n")
    P("| set (tokens) | " + " | ".join(conds) + " | chance (none) |")
    P("|---|" + "---|" * (len(conds) + 1))
    for an in ALLSETS:
        row, chs = [], ""
        for c in conds:
            d = win_rates(by[c], an, lnn, idx, "all positions")
            row.append(f"{f1(d['top1'])} / {f1(d['top5'])}")
            SUMMARY.setdefault("xanimal", {})[f"{nm}|{an}|{c}"] = d
            if c == "none":
                chs = f"{f1(d['ch1'])} / {f1(d['ch5'])}"
        P(f"| {an} ({len(meta['sets'][an])}) | " + " | ".join(row) + f" | {chs} |")

json.dump(SUMMARY, open(a.json, "w"), indent=0, default=float)
OUT.close()
print("wrote", a.out)
