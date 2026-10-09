"""Lens fidelity: agreement between the lens readout at layer l and the model's own next-token distribution.

Position groups (rerun data, sample 0 of each completion, teacher-forced under its own condition, positions
predicting each response token):
  numbers     : numbers task, conds none / sys_owl / sys_horse / user_owl (100 items each)
  numbers_dig : subset of 'numbers' positions whose predicted (next) token contains a digit
  prose       : trivia + open tasks, conds none / sys_owl (150 items each)
  wikitext    : 100 WikiText-103 *test* texts, raw (no chat template), 128 tokens, positions 16..126
Metrics per layer, pooled over positions: top-1 agreement over the full vocab, top-1 agreement among
whole-word tokens (the ranking used by the readouts), KL(model || lens) over the full vocab (nats), and
KL between the two distributions renormalized over whole-word tokens only (kl_ww).
Layer 27 is the model's own output (identity transport), so its agreement is 1 and KL 0 by construction.
usage: python appendix_g_lens_refit/lens_fidelity.py --lens LENS --gen GEN.jsonl --prompts data/onpolicy_owl.jsonl --out OUT.json
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402
from sljlens.readout import residuals  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lens", default="qwen25_7b", help="lens registry key or file")
ap.add_argument("--model-path", default=None)
ap.add_argument("--gen", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

hf, tok, model = models.load_model("qwen25_7b", model_path=a.model_path)
lens = models.load_lens(a.lens)
L = model.n_layers
LAYERS = list(range(16, L))
V = model._lm_head.weight.shape[0]
WW = re.compile(r"^ [A-Za-z]{3,}$")
wwmask = torch.zeros(V, dtype=torch.bool)
for i in range(min(V, len(tok))):
    if WW.match(tok.decode([i])):
        wwmask[i] = True
wwmask = wwmask.cuda()
digit_tok = torch.zeros(V, dtype=torch.bool)
for i in range(min(V, len(tok))):
    if re.search(r"\d", tok.decode([i])):
        digit_tok[i] = True

prompts = {json.loads(l)["id"]: json.loads(l) for l in open(a.prompts)}
gen = [json.loads(l) for l in open(a.gen)]
GROUPS = ["numbers", "numbers_dig", "prose", "wikitext"]
acc = {g: {l: dict(n=0, agree=0.0, agree_ww=0.0, kl=0.0, kl_ww=0.0) for l in LAYERS} for g in GROUPS}


@torch.no_grad()
def score(ids, pos, groups_for_pos):
    """groups_for_pos: list (per position) of lists of group names."""
    acts = residuals(model, ids, LAYERS)
    final = model.unembed(acts[L - 1][pos]).float()[:, :V]
    lp_m = torch.log_softmax(final, -1)
    p_m = lp_m.exp()
    top_m = final.argmax(-1)
    fm_ww = final.masked_fill(~wwmask, float("-inf"))
    top_m_ww = fm_ww.argmax(-1)
    lp_m_ww = torch.log_softmax(fm_ww, -1)
    p_m_ww = lp_m_ww.exp()
    for l in LAYERS:
        h = acts[l][pos]
        z = model.unembed(lens.transport(h, l) if l in lens.jacobians else h).float()[:, :V]
        lp_l = torch.log_softmax(z, -1)
        kl = (p_m * (lp_m - lp_l)).sum(-1)
        ag = (z.argmax(-1) == top_m).float()
        zw = z.masked_fill(~wwmask, float("-inf"))
        agw = (zw.argmax(-1) == top_m_ww).float()
        lp_l_ww = torch.log_softmax(zw, -1)
        klw = torch.where(wwmask, p_m_ww * (lp_m_ww - lp_l_ww), torch.zeros_like(lp_l_ww)).sum(-1)
        for g in GROUPS:
            msk = torch.tensor([g in gs for gs in groups_for_pos], device=kl.device)
            if not bool(msk.any()):
                continue
            d = acc[g][l]
            d["n"] += int(msk.sum())
            d["agree"] += float(ag[msk].sum())
            d["agree_ww"] += float(agw[msk].sum())
            d["kl"] += float(kl[msk].sum())
            d["kl_ww"] += float(klw[msk].sum())


for g in gen:
    m = g["meta"]
    if m["task"] == "numbers" and m["cond"] in ("none", "sys_owl", "sys_horse", "user_owl"):
        grp = "numbers"
    elif m["task"] in ("trivia", "open") and m["cond"] in ("none", "sys_owl"):
        grp = "prose"
    else:
        continue
    p = tok(tok.apply_chat_template(prompts[g["id"]]["messages"], tokenize=False, add_generation_prompt=True),
            add_special_tokens=False)["input_ids"]
    r = tok(g["outputs"][0]["text"], add_special_tokens=False)["input_ids"]
    if not r:
        continue
    ids = p + r
    pos = list(range(len(p) - 1, len(ids) - 1))
    gfp = []
    for q in pos:
        gs = [grp]
        if grp == "numbers" and bool(digit_tok[ids[q + 1]]):
            gs.append("numbers_dig")
        gfp.append(gs)
    score(ids, pos, gfp)

from datasets import load_dataset  # noqa: E402
texts = []  # first 100 WikiText-103 test records >= 600 chars, truncated to 2000 chars
for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"]:
    if len(t.strip()) >= 600:
        texts.append(t[:2000])
        if len(texts) == 100:
            break
for t in texts:
    ids = tok(t, add_special_tokens=False)["input_ids"][:128]
    pos = list(range(16, len(ids) - 1))
    score(ids, pos, [["wikitext"]] * len(pos))

out = {g: {l: {k: (v / d["n"] if k != "n" else v) for k, v in d.items()} for l, d in acc[g].items()} for g in GROUPS}
json.dump(dict(lens=a.lens, layers=LAYERS, groups=out), open(a.out, "w"), indent=1)
for g in GROUPS:
    print(g, "n_pos", out[g][LAYERS[0]]["n"])
    for l in LAYERS:
        o = out[g][l]
        print(f"  L{l}: agree {o['agree']:.3f} agree_ww {o['agree_ww']:.3f} KL {o['kl']:.3f} KL_ww {o['kl_ww']:.3f}")
print("DONE", flush=True)
