#!/usr/bin/env python3
"""Compare two sets of strong_readout.py outputs record by record (smoke tests: a re-run against a reference readout).
For every completion id present in both and every lens / set / random-set size present in both: the number of per-position,
per-layer target ranks and random-set ranks that are equal, and the largest absolute difference.

usage: python planned_strong_lens_models/compare_raw.py "NEW_*.pt" "REF_*.pt"
"""
import glob
import sys

import torch


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        for r in torch.load(f, weights_only=False)["recs"]:
            out[r["id"]] = r
    return out


new, ref = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(new) & set(ref))
print(f"{len(new)} new records, {len(ref)} reference records, {len(ids)} in common")
eq = tot = 0
worst = 0
for i in ids:
    a, b = new[i], ref[i]
    assert a["text"] == b["text"] and a["cur"] == b["cur"], f"{i}: different completion or tokenization"
    for ln in set(a["lens"]) & set(b["lens"]):
        ea, eb = a["lens"][ln], b["lens"][ln]
        assert ea["layers"] == eb["layers"], (i, ln)
        pairs = [(ea["tgt"][k], eb["tgt"][k]) for k in set(ea["tgt"]) & set(eb["tgt"])]
        pairs += [(ea["rand"][z], eb["rand"][z]) for z in set(ea["rand"]) & set(eb["rand"])]
        for x, y in pairs:
            x, y = x.int(), y.int()
            eq += int((x == y).sum())
            tot += x.numel()
            worst = max(worst, int((x - y).abs().max()))
print(f"values compared: {tot}, equal: {eq} ({100 * eq / max(tot, 1):.3f}%), max abs rank difference: {worst}")
