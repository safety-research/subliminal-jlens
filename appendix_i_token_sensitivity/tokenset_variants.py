#!/usr/bin/env python3
"""Token-set sensitivity of the headline number readout (src/owl_tokenset_variants.py). GPU.

Sets:   base   = ' owl', ' Owl'
        ext    = base + every other single-token spelling among {owl, Owl, OWL, owls, Owls, OWLS} with/without a leading
                 space (in Qwen2.5: 'owl', 'OWL')
        ext_ow = ext + ' ow' / 'ow' / ' Ow' / 'Ow' (pieces that begin ' owls' / ' Owls'; single tokens only)
Pools:  'ww+extra' = whole-word tokens plus every token of the three sets; 'full' = the whole vocabulary (J-lens paper).
Data: the on-policy number completions (100 prompts x 3) of none / sys_owl / sys_horse / user_owl.
Detection: top-k at any response position and any layer of the scope. Chance: 100 random sets of the same size drawn from
the same pool (random.Random(0), drawn for pool 'ww+extra' then 'full', sizes in ascending order), computed on the
no-system-prompt completions only and pooled over completions (as the original did).
usage: python appendix_i_token_sensitivity/tokenset_variants.py --gen GEN.jsonl --prompts data/onpolicy_owl.jsonl
"""
import argparse
import os
import random
import statistics as st
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ))
from sljlens import models, readout, vocab  # noqa: E402
from sljlens.io import read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--model-path", default=None)
ap.add_argument("--lens", default="qwen25_7b")
a = ap.parse_args()
hf, tok, model = models.load_model("qwen25_7b", model_path=a.model_path)
lens = models.load_lens(a.lens)
LAYERS = sorted(lens.jacobians)
BAND = [LAYERS.index(l) for l in models.cfg("qwen25_7b")["band"]]
V = model._lm_head.weight.shape[0]
ww = vocab.whole_word_mask(tok, V)
spell = [p + w for w in ["owl", "Owl", "OWL", "owls", "Owls", "OWLS"] for p in ("", " ")]
SETS = {"base": vocab.spelling_set(tok, [" owl", " Owl"]), "ext": vocab.spelling_set(tok, spell),
        "ext_ow": vocab.spelling_set(tok, spell + ["ow", " ow", "Ow", " Ow"])}
print({k: [tok.decode([i]) for i in v] for k, v in SETS.items()}, flush=True)
extra = torch.zeros(V, dtype=torch.bool)
for v in SETS.values():
    extra[v] = True
POOLS = {"ww+extra": (ww | extra).cuda(), "full": torch.ones(V, dtype=torch.bool).cuda()}
rng = random.Random(0)
RAND = {(pool, n): torch.tensor([rng.sample(POOLS[pool].nonzero().squeeze(-1).tolist(), n) for _ in range(100)]).cuda()
        for pool in POOLS for n in {len(v) for v in SETS.values()}}
SETS_T = {k: torch.tensor(v).cuda() for k, v in SETS.items()}
prompts = {r["id"]: r for r in read_jsonl(a.prompts)}
res = defaultdict(lambda: defaultdict(list))
chance = defaultdict(list)
for g in read_jsonl(a.gen):
    m = g["meta"]
    if m["task"] != "numbers" or m["cond"] not in ("none", "sys_owl", "sys_horse", "user_owl"):
        continue
    p = models.render_ids(tok, prompts[g["id"]]["messages"], "qwen25_7b")
    for s, o in enumerate(g["outputs"]):
        r = tok(o["text"], add_special_tokens=False)["input_ids"]
        pos = readout.response_positions(len(p), len(r))
        acts = readout.residuals(model, p + r, LAYERS)
        bst = {(pool, k): [] for pool in POOLS for k in SETS}
        rbest = {key: [] for key in RAND}
        for L in LAYERS:
            z = readout.lens_logits(model, lens, acts[L][pos], L, V)
            for pool, mask in POOLS.items():
                ranks = readout.ww_ranks(z, mask)
                for k, ids in SETS_T.items():
                    bst[(pool, k)].append(int(ranks[:, ids].amin()))
                for (pp, n), R in RAND.items():
                    if pp == pool:
                        rbest[(pp, n)].append(ranks[:, R].amin(-1).amin(0))
        for key, b in bst.items():
            res[(m["cond"],) + key][m["item"]].append(b)
        if m["cond"] == "none":
            for key, rb in rbest.items():
                chance[key].append(torch.stack(rb))
print("\nTop-1/5/20 detection (%), numbers, on-policy, n=300 per condition", flush=True)
for scope, idx in (("band 22-26", BAND), ("all layers", list(range(len(LAYERS))))):
    for pool in POOLS:
        for k in SETS:
            row = []
            for c in ("sys_owl", "sys_horse", "none", "user_owl"):
                by = res[(c, pool, k)]
                vals = [100 * st.mean(st.mean(float(min(b[i] for i in idx) <= kk) for b in v) for v in by.values()) for kk in (1, 5, 20)]
                row.append(f"{c} {vals[0]:.1f}/{vals[1]:.1f}/{vals[2]:.1f}")
            cs = torch.stack(chance[(pool, len(SETS[k]))])[:, idx].amin(1)
            cv = "/".join(f"{100 * float((cs <= kk).float().mean()):.1f}" for kk in (1, 5, 20))
            print(f"{scope} | pool {pool} | set {k} (n={len(SETS[k])}) | chance {cv} | " + " | ".join(row), flush=True)
