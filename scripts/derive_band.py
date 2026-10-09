#!/usr/bin/env python3
"""Read-band derivation with the J-lens paper's directed-modulation toy task (lens_check/readband_check.py Method 2 +
band_analyze.py), on the layers the lens covers plus the output layer.

Trials: for each of the 22 topic categories and the first 5 carrier sentences of jacobian-lens'
data/experiments/directed-modulation.json, the user asks the model to copy the sentence ('Write "<s>" Don't write anything
else.'), either plainly ('none') or with one of the task's phrasings (focus: "Think about {x} while you write.", ...;
control: mentions; suppress: dismissal / negated). System prompt "You are a helpful assistant."; the response is the
sentence itself, teacher-forced. At each answer token and layer, the topic's best rank among word-like tokens
(`^ ?[A-Za-z]{3,}$`; members as single tokens with or without the leading space; ties count in the token's favour).
Per layer: hit_k = [best rank over answer tokens <= k]; separation s(l) = mean over (topic, carrier) of
(mean over focus trials - mean over none trials). Band = the maximal contiguous run of layers containing argmax s with
s >= max(s)/2; the recommended read band is the k=5 band minus the output layer (whose readout is the model's own output).
For Qwen2.5-7B this gives layers 22-26 (k=1: 22-26, k=5: 22-27).
--trials-out saves every trial's per-layer, per-position ranks (lens_check/readband_check.py's band_<key>.pkl format:
{"key", "layers", "n_layers", "dm": [{group, phrasing, cat, carrier, rank [n_layers_read, P]}]}), which
04_lens_strength_toy_task/dm_strength.py turns into the toy-task strength tables (any band, all lens layers, output layer).
The output layer (the model's own next-token logits, no lens) is always read, as the last entry of "layers".

usage: python scripts/derive_band.py --model qwen25_7b [--lens KEY_OR_PATH] --out band.json [--trials-out band.pkl]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models, readout, vocab  # noqa: E402
from sljlens.io import DATA  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--model-path", default=None)
ap.add_argument("--lens", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--trials-out", default=None)
a = ap.parse_args()
DM = json.load(open(os.path.join(DATA, "directed-modulation.json")))
SYSTEM = "You are a helpful assistant."


def dm_trials(n_carriers=5):
    for cat in DM["topic_categories"]:
        for s in DM["carrier_sentences"][:n_carriers]:
            yield dict(group="none", phrasing="none", cat=cat["name"], members=cat["members"],
                       user=f'Write "{s}" Don\'t write anything else.', resp=s)
            for ph in DM["phrasings"]:
                x = ph["text"].format(x=cat["name"])
                x = x[0].upper() + x[1:]
                yield dict(group=DM["group_kind"][ph["group"]], phrasing=ph["name"], cat=cat["name"], members=cat["members"],
                           user=f'Write "{s}" {x} Don\'t write anything else.', resp=s)


hf, tok, model = models.load_model(a.model, model_path=a.model_path)
lens = models.load_lens(a.lens or a.model)
L = model.n_layers
LAYERS = sorted(set(lens.jacobians) | {L - 1})
V = model._lm_head.weight.shape[0]
vmask = vocab.wordlike_mask(tok, V).cuda()
trials = []
with torch.no_grad():
    for t in dm_trials():
        mids = vocab.single_token_ids(tok, t["members"])
        if not mids:
            continue
        prefix = models.render_text(tok, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t["user"]}], a.model)
        pids = tok(prefix, add_special_tokens=False).input_ids
        ids = pids + tok(t["resp"], add_special_tokens=False).input_ids
        pos = list(range(len(pids), len(ids)))
        acts = readout.residuals(model, ids, LAYERS)
        cid = torch.as_tensor(mids, device="cuda")
        rk = []
        for l in LAYERS:
            lg = readout.lens_logits(model, lens, acts[l][pos], l, V)
            masked = lg.masked_fill(~vmask[:V], float("-inf"))
            best = masked[:, cid].max(-1).values
            rk.append(((masked > best[:, None]).sum(-1) + 1).int().cpu())
        trials.append(dict(group=t["group"], phrasing=t["phrasing"], cat=t["cat"], carrier=t["resp"],
                           rank=torch.stack(rk).numpy()))
cats = sorted({t["cat"] for t in trials})
cars = sorted({t["carrier"] for t in trials})


def half_max_run(s):
    m = int(np.nanargmax(s))
    thr = 0.5 * s[m]
    lo = hi = m
    while lo > 0 and s[lo - 1] >= thr:
        lo -= 1
    while hi < len(s) - 1 and s[hi + 1] >= thr:
        hi += 1
    return lo, hi


res = {"model": a.model, "layers": LAYERS, "n_trials": len(trials)}
for k in (1, 3, 5, 10):
    agg = defaultdict(list)
    for t in trials:
        agg[(t["cat"], t["carrier"], t["group"])].append((t["rank"].min(1) <= k).astype(float))
    arr = {g: np.zeros((len(cats), len(cars), len(LAYERS))) for g in ("focus", "none")}
    for (c, r, g), v in agg.items():
        if g in arr:
            arr[g][cats.index(c), cars.index(r)] = np.mean(v, 0)
    s = (arr["focus"] - arr["none"]).mean((0, 1))
    lo, hi = half_max_run(s)
    res[f"hit{k}"] = dict(separation=[round(float(x), 4) for x in s], band=[LAYERS[lo], LAYERS[hi]],
                          peak=round(float(s.max()), 3), peak_layer=LAYERS[int(s.argmax())])
    print(f"hit@{k}: half-max band L{LAYERS[lo]}-L{LAYERS[hi]} (peak {s.max():.2f} at L{LAYERS[int(s.argmax())]})", flush=True)
lo, hi = res["hit5"]["band"]
res["read_band"] = [l for l in LAYERS if lo <= l <= hi and l != L - 1]
print("recommended read band (k=5 band minus the output layer):", res["read_band"], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
if a.trials_out:
    import pickle
    pickle.dump({"key": a.model, "lens": models.lens_path(a.lens or a.model), "layers": LAYERS, "n_layers": L, "dm": trials},
                open(a.trials_out, "wb"))
