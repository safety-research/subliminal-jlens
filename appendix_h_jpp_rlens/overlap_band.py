#!/usr/bin/env python3
"""Top-1 hit overlap between lenses, each read in its own band, from all-layer scripts/score_lens.py files
(jpp/nopen/overlap_band.py and jpp/overlap.py of the original runs): hit = best rank over positions x band layers == 1;
expected overlap if independent = |H0| |H1| / n; also each lens's position-0-only hits (top-1 at the last prompt token
only) and the base lens's hits that are in this lens's top 5. Filter-passing completions only.
usage: python appendix_h_jpp_rlens/overlap_band.py BASE.pt:a-b OTHER.pt:a-b [OTHER.pt:a-b ...]
"""
import sys

import torch


def load(spec):
    f, b = spec.rsplit(":", 1)
    lo, hi = map(int, b.split("-"))
    out = {}
    for r in torch.load(f, weights_only=False)["recs"]:
        if not r["passes"]:
            continue
        m = r["mat"][:, lo:hi + 1].int()
        out[r["key"]] = (int(m.min()), int(m[1:].min()) if m.shape[0] > 1 else 99999)
    return out


base = load(sys.argv[1])
keys = sorted(base)
n = len(keys)
H0 = {k for k in keys if base[k][0] == 1}
print(f"{sys.argv[1]}: n={n}, top-1 hits {len(H0)}, position-0-only {sum(base[k][1] > 1 for k in H0)}")
for s in sys.argv[2:]:
    d = load(s)
    assert set(d) == set(keys), s
    H1 = {k for k in keys if d[k][0] == 1}
    print(f"{s}: top-1 hits {len(H1)}; also base top-1 {len(H0 & H1)} (expected if independent {len(H0) * len(H1) / n:.1f}); "
          f"base hits in this lens's top 5 {sum(d[k][0] <= 5 for k in H0)}; position-0-only hits {sum(d[k][1] > 1 for k in H1)}")
