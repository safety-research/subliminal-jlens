#!/usr/bin/env python3
"""Completion-level owl rates for any layer band from ONE all-layer scripts/score_lens.py file scored with --band 0-26
--rand-per-layer (jpp/rates_band.py of the original runs). Filter-passing completions only.

  owl best rank   min over response positions x band layers of the stored rank matrix `mat` [P, 27]
  chance          mean over completions and over the 100 random 2-token whole-word sets of [band-min of rand_mat <= k]
                  (the same rule on the same completions; rand_mat [27, 100] holds each set's per-layer minimum over
                  positions)
  _nop0           the same with position 0 (the last prompt token) excluded
Reading every lens at all 27 layers once and taking any band afterwards gives exactly the numbers of a scoring pass at that
band (bf16 batch-composition ties aside, DISCREPANCIES.md).

usage: python appendix_h_jpp_rlens/rates_band.py SCORE.pt BAND [--json OUT.json]     BAND = 'a-b' or 'all'
"""
import argparse
import json
import statistics

import torch


def band_rates(path, spec):
    recs = [r for r in torch.load(path, weights_only=False)["recs"] if r["passes"]]
    L = recs[0]["mat"].shape[1]
    B = list(range(L)) if spec == "all" else list(range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1))
    n = len(recs)
    best = [int(r["mat"][:, B].int().min()) for r in recs]
    best0 = [int(r["mat"][1:, B].int().min()) if r["mat"].shape[0] > 1 else 99999 for r in recs]
    rb = torch.stack([r["rand_mat"][B].amin(0) for r in recs])                                   # [n, 100]
    rb0 = torch.stack([r["rand_mat_nop0"][B].amin(0) for r in recs if r.get("rand_mat_nop0") is not None])
    o = {"file": path, "band": spec, "n": n, "median_best_rank": statistics.median(best)}
    for k in (1, 5, 20):
        o[f"top{k}"] = 100 * sum(b <= k for b in best) / n
        o[f"chance{k}"] = 100 * float((rb <= k).float().mean())
        o[f"top{k}_nop0"] = 100 * sum(b <= k for b in best0) / n
        o[f"chance{k}_nop0"] = 100 * float((rb0 <= k).float().mean()) * len(rb0) / n
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("score")
    ap.add_argument("band")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    o = band_rates(a.score, a.band)
    print(f"{a.score.split('/')[-1]} band {a.band} n={o['n']} top1/5/20 = {o['top1']:.2f} / {o['top5']:.2f} / "
          f"{o['top20']:.2f}  chance {o['chance1']:.2f} / {o['chance5']:.2f} / {o['chance20']:.2f} | nop0 "
          f"{o['top1_nop0']:.2f} / {o['top5_nop0']:.2f} / {o['top20_nop0']:.2f} chance {o['chance1_nop0']:.2f} / "
          f"{o['chance5_nop0']:.2f} / {o['chance20_nop0']:.2f} | median {o['median_best_rank']}")
    if a.json:
        json.dump(o, open(a.json, "w"))
