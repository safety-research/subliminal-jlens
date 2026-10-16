#!/usr/bin/env python3
"""Summaries for the lens-strength screen (directed modulation; lens_screen/analyze.py of the original runs), the read-band
statistic of scripts/derive_band.py. Reads trial files in either format (screen.py / derive_band.py --trials-out: a "layers"
list whose last entry is the output layer L-1, read directly; the original runs' lens_check format: meta.n_layers, every
layer 0..L-1). For each file:
 (a) per-layer separation s_k(l) = mean over (category, carrier) of [focus hit@k - none hit@k], k = 1, 5; peak over lens
     layers (output layer excluded) and the output layer's own value, with a 95% two-way bootstrap CI (categories x carriers)
     on peak-lens minus output;
 (b) half-max band at top-1 over lens layers only (maximal contiguous run of measured lens layers around the argmax with
     s >= 0.5 max), and the same at top-5;
 (c) any-layer-in-band trial hit rates for focus / mention / suppress / none at k = 1, 5 (top-1 band);
 (d) the same for the output layer alone (the model's own logits, no lens).
Bootstrap draws use numpy default_rng(0), consumed in argument order (so the CIs depend on the order of the files).
usage: python planned_lens_strength_screen/analyze.py [--json out.json] LABEL=TRIALS.pkl [LABEL=TRIALS.pkl ...]
"""
import json
import pickle
import sys
from collections import defaultdict

import numpy as np

rng = np.random.default_rng(0)
args = sys.argv[1:]
jout = None
if args and args[0] == "--json":
    jout, args = args[1], args[2:]


def load(path):
    B = pickle.load(open(path, "rb"))
    if "meta" in B:
        L = B["meta"]["n_layers"]
        layers = list(range(L))
    else:
        L, layers = B["n_layers"], list(B["layers"])
    assert layers[-1] == L - 1
    return B, L, layers


def run(s):
    m = int(np.nanargmax(s)); thr = 0.5 * s[m]; a = b = m
    while a > 0 and s[a - 1] >= thr:
        a -= 1
    while b < len(s) - 1 and s[b + 1] >= thr:
        b += 1
    return a, b


def summarize(path):
    B, L, layers = load(path)
    dm = B["dm"]
    lens_idx = [i for i, l in enumerate(layers) if l != L - 1]
    out_idx = layers.index(L - 1)
    cats = sorted({t["cat"] for t in dm}); cars = sorted({t["carrier"] for t in dm})
    best = {g: np.stack([t["rank"].min(1) for t in dm if t["group"] == g]) for g in ("focus", "control", "suppress", "none")}
    res = dict(n_layers=L, lens_layers=[layers[i] for i in lens_idx], n_trials={g: len(v) for g, v in best.items()})
    for k in (1, 5):
        agg = defaultdict(list)
        for t in dm:
            agg[(t["cat"], t["carrier"], t["group"])].append((t["rank"].min(1) <= k).astype(float))
        arr = {g: np.zeros((len(cats), len(cars), len(layers))) for g in ("focus", "none")}
        for (c, r, g), v in agg.items():
            if g in arr:
                arr[g][cats.index(c), cars.index(r)] = np.mean(v, 0)
        D = arr["focus"] - arr["none"]
        s = D.mean((0, 1))
        sl = s[lens_idx]
        pi = int(np.argmax(sl))
        a, b = run(sl)
        boots = []
        for _ in range(1000):
            ci, ri = rng.integers(0, len(cats), len(cats)), rng.integers(0, len(cars), len(cars))
            sb = D[ci][:, ri].mean((0, 1))
            boots.append(sb[lens_idx].max() - sb[out_idx])
        res[f"k{k}"] = dict(per_layer={int(layers[i]): round(float(s[i]), 4) for i in range(len(layers))},
                            peak=float(sl[pi]), peak_layer=int(layers[lens_idx[pi]]), peak_rel=layers[lens_idx[pi]] / (L - 1),
                            output=float(s[out_idx]), peak_minus_output_ci=[float(x) for x in np.percentile(boots, [2.5, 97.5])],
                            band=[int(layers[lens_idx[a]]), int(layers[lens_idx[b]])],
                            band_layers=[int(layers[lens_idx[i]]) for i in range(a, b + 1)])
    for bk in ("k1", "k5"):
        bl = res[bk]["band_layers"]
        bidx = [layers.index(l) for l in bl]
        hits = {}
        for kk in (1, 5):
            hits[f"top{kk}"] = {g: float(np.mean(v[:, bidx].min(1) <= kk)) for g, v in best.items()}
        res[bk]["band_hits"] = hits
    fr = np.array([np.mean(best["focus"][:, i] <= 1) for i in lens_idx])
    bi = int(fr.argmax())
    res["best_single_layer_top1"] = dict(layer=int(layers[lens_idx[bi]]), focus=float(fr[bi]),
                                         none=float(np.mean(best["none"][:, lens_idx[bi]] <= 1)),
                                         suppress=float(np.mean(best["suppress"][:, lens_idx[bi]] <= 1)))
    res["output_hits"] = {f"top{kk}": {g: float(np.mean(v[:, out_idx] <= kk)) for g, v in best.items()} for kk in (1, 5)}
    if "sanity" in B:
        sa = B["sanity"]
        ag = sa.get("lens_top1_agreement", {})
        res["sanity"] = dict(hf_vs_unembed_top1=sa.get("hf_vs_unembed_top1"), hf_vs_unembed_maxabs=sa.get("hf_vs_unembed_maxabs"),
                             agree_last_lens_layer=ag[max(ag)] if ag else None, agree_by_layer=ag)
    return res


ALL = {}
for spec in args:
    lab, path = spec.split("=", 1)
    r = ALL[lab] = summarize(path)
    L = r["n_layers"]
    print(f"\n## {lab}  (L={L}, lens layers {r['lens_layers'][0]}..{r['lens_layers'][-1]} [{len(r['lens_layers'])}], "
          f"trials {r['n_trials']})")
    for k in (1, 5):
        x = r[f"k{k}"]
        print(f"  top-{k}: peak lens sep {x['peak']:.3f} at L{x['peak_layer']} ({x['peak_rel']:.2f}); output-layer sep "
              f"{x['output']:.3f}; peak-output CI [{x['peak_minus_output_ci'][0]:+.3f}, {x['peak_minus_output_ci'][1]:+.3f}]; "
              f"half-max band L{x['band'][0]}-L{x['band'][1]}")
        print("    per layer: " + " ".join(f"{l}:{v:.2f}" for l, v in x["per_layer"].items()))
    for bk in ("k1", "k5"):
        h = r[bk]["band_hits"]
        print(f"  any-layer-in-{bk}-band L{r[bk]['band'][0]}-{r[bk]['band'][1]} hit (focus/mention/suppress/none): " +
              "  ".join(f"{t}: {h[t]['focus']:.2f}/{h[t]['control']:.2f}/{h[t]['suppress']:.2f}/{h[t]['none']:.2f}" for t in h))
    b = r["best_single_layer_top1"]
    print(f"  best single lens layer at top-1: L{b['layer']} focus {b['focus']:.2f} / suppress {b['suppress']:.2f} / none {b['none']:.2f}")
    h = r["output_hits"]
    print("  output layer only (no lens): " +
          "  ".join(f"{t}: {h[t]['focus']:.2f}/{h[t]['control']:.2f}/{h[t]['suppress']:.2f}/{h[t]['none']:.2f}" for t in h))
    if "sanity" in r:
        s = r["sanity"]
        print(f"  sanity: hf-vs-unembed top-1 agreement {s['hf_vs_unembed_top1']}, maxabs {s['hf_vs_unembed_maxabs']}; "
              f"lens top-1 agreement with model output at last lens layer {s['agree_last_lens_layer']}")
        ag = s["agree_by_layer"]
        print("    agreement by layer: " + " ".join(f"{l}:{v:.2f}" for l, v in list(ag.items())[::max(1, len(ag) // 16)]))
if jout:
    json.dump(ALL, open(jout, "w"), indent=1)
