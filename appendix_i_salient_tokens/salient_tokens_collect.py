"""What does the Jacobian lens find salient while the (owl / eagle / control) teacher writes numbers?

Teacher-forces each teacher's own number samples under several system prompts and records, at every
position of prompt + response and at a spread of layers (plus the model's real output logits):
  * lens top-10 token ids over the full vocabulary and over whole-word tokens (" " + >=3 letters)
  * per-(condition, layer, role) sums over positions of the full-vocab lens log-prob and prob, so that
    mean logp / log mean prob per token can be differenced between system prompts on identical text.
Roles are defined by the token being predicted at a position:
  sys, user, hdr (chat-template tokens between user content and the answer), first (last prompt token,
  predicts the first answer token), dstart (predicts the first digit of a number), dmid (predicts a
  later digit of a number), sep (predicts a separator / newline / other non-digit), eot (predicts <|im_end|>).

Teacher samples: the first shard of the Qwen2.5-7B owl / control / eagle teacher data of 01_subliminal_repro
(teach_<t>.s0of4.jsonl); the first 100 prompt ids where all three answers are pure number lists.
usage: python appendix_i_salient_tokens/salient_tokens_collect.py --teach-dir OUT/01/teach --out OUT/salient/raw [--n 100]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402
from sljlens.prompts import system_prompt  # noqa: E402
from sljlens.readout import residuals  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--teach-dir", required=True)
ap.add_argument("--model-path", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

hf, tok, model = models.load_model("qwen25_7b", model_path=a.model_path)
lens = models.load_lens("qwen25_7b")
print("lens layers", sorted(lens.jacobians), flush=True)
LAYERS = [2, 5, 8, 11, 14, 17, 20, 22, 23, 24, 25, 26]
assert all(l in lens.jacobians for l in LAYERS)
NL = len(LAYERS) + 1  # + final logits
VT = len(tok)
WW = re.compile(r"^ [A-Za-z]{3,}$")
vocab = [tok.decode([i]) for i in range(VT)]
wwmask = torch.tensor([bool(WW.match(s)) for s in vocab]).cuda()
json.dump(vocab, open(f"{a.out}/vocab.json", "w"))
IM_START, IM_END = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
DIGIT = torch.zeros(VT, dtype=torch.bool)
for i, s in enumerate(vocab):
    if s.strip().isdigit():
        DIGIT[i] = True

VALID = re.compile(r"^[\d\s,;.()\[\]]+$")
D = a.teach_dir
rows = {}
for t in ("owl", "control", "eagle"):
    rows[t] = {r["id"]: r for r in map(json.loads, open(f"{D}/teach_{t}.s0of4.jsonl"))}
ids = [i for i in sorted(rows["owl"]) if all(i in rows[t] and VALID.match(rows[t][i]["response"]) and
                                               rows[t][i]["prompt"] == rows["owl"][i]["prompt"] for t in rows)][: a.n]
print(len(ids), "samples", flush=True)
SP = {"owl": system_prompt("owl"), "control": None, "eagle": system_prompt("eagle")}
# condition = (system prompt, which teacher's text)
CONDS = [(s, t) for t in ("owl", "control", "eagle") for s in ("owl", "control", "eagle")]
ROLES = ["sys", "user", "hdr", "first", "dstart", "dmid", "sep", "eot"]
RI = {r: i for i, r in enumerate(ROLES)}
S_lp = torch.zeros(len(CONDS), NL, len(ROLES), VT, device="cuda")
S_p = torch.zeros_like(S_lp)
N = np.zeros((len(CONDS), len(ROLES)), dtype=np.int64)


def build(sys_text, user, resp):
    msgs = ([{"role": "system", "content": sys_text}] if sys_text else []) + [{"role": "user", "content": user}]
    p = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    r = tok(resp, add_special_tokens=False)["input_ids"] + [IM_END]
    seq = p + r
    st = [i for i, t in enumerate(p) if t == IM_START]  # system, user, assistant
    role = np.full(len(seq) - 1, -1)
    role[st[0] + 3: st[1] - 2] = RI["sys"]          # content after '<|im_start|>system\n', before '<|im_end|>\n'
    role[st[1] + 3: st[2] - 2] = RI["user"]
    role[st[2] - 2: len(p) - 1] = RI["hdr"]
    role[len(p) - 1] = RI["first"]
    for q in range(len(p), len(seq) - 1):
        nxt = seq[q + 1]
        if nxt == IM_END:
            role[q] = RI["eot"]
        elif DIGIT[nxt]:
            role[q] = RI["dmid"] if DIGIT[seq[q]] else RI["dstart"]
        else:
            role[q] = RI["sep"]
    return seq, role, len(p)


meta = []
for ci, (s, t) in enumerate(CONDS):
    for k, i in enumerate(ids):
        seq, role, plen = build(SP[s], rows[t][i]["prompt"], rows[t][i]["response"])
        P = len(seq) - 1
        acts = residuals(model, seq, LAYERS)
        with torch.no_grad():
            fin = hf(torch.tensor([seq]).cuda()).logits[0, :P, :VT].float()
        top_full = np.zeros((NL, P, 10), dtype=np.int32)
        top_ww = np.zeros((NL, P, 10), dtype=np.int32)
        rt = torch.as_tensor(role, device="cuda")
        for li in range(NL):
            if li < len(LAYERS):
                l = LAYERS[li]
                with torch.no_grad():
                    z = model.unembed(lens.transport(acts[l][:P], l)).float()[:, :VT]
            else:
                z = fin
            lp = torch.log_softmax(z, -1)
            top_full[li] = z.topk(10, -1).indices.cpu().numpy()
            top_ww[li] = z.masked_fill(~wwmask, float("-inf")).topk(10, -1).indices.cpu().numpy()
            for r in range(len(ROLES)):
                m = rt == r
                if m.any():
                    S_lp[ci, li, r] += lp[m].sum(0)
                    S_p[ci, li, r] += lp[m].exp().sum(0)
        for r in range(len(ROLES)):
            N[ci, r] += int((role == r).sum())
        np.savez_compressed(f"{a.out}/seq_{s}_{t}_{k:03d}.npz", seq=np.array(seq), role=role, plen=plen,
                            top_full=top_full, top_ww=top_ww)
    print(f"cond sys={s} text={t} done", flush=True)

np.save(f"{a.out}/S_lp.npy", S_lp.cpu().numpy())
np.save(f"{a.out}/S_p.npy", S_p.cpu().numpy())
json.dump(dict(conds=CONDS, roles=ROLES, layers=LAYERS + ["final"], N=N.tolist(), ids=ids), open(f"{a.out}/meta.json", "w"))
print("DONE", flush=True)
