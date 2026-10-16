#!/usr/bin/env python3
"""Keyword-sweep lens readout (Qwen2.5-7B-Instruct, our own J-lens, read band L22-26); kwsweep/kw_lens.py of the original run.

The standard readout of the project (the same rule as appendix E's animal readouts): every filter-passing completion
(paper filter, and the completion stopped naturally) is teacher-forced under the prompt that produced it; at every position
that predicts a response token (position 0 = the last prompt token) and every band layer, the rank among whole-word tokens
(`^ [A-Za-z]{3,}$`, argsort tie-breaking) of each keyword set's best token; best rank over band layers x positions.
Every completion is read for EVERY keyword's set in one pass (that gives the other-keyword baseline). Chance: 100 random
whole-word sets per set size (sizes 1-4 drawn in order from one random.Random(0), as the original), same rule.

Position scopes:
  all       every read position
  ex0       position 0 dropped (the last prompt token, identical for every completion of a prompt)
  exfirst   only positions predicting tokens after the first number (everything up to and including the position that
            predicts the first number's last digit is dropped)

Output, one JSON line per generated completion (the record format the analysis scripts read):
  id, phase, cond, word, pid, sample, passed, text[:400]; and for filter-passing completions:
  n_pos, e1 (index of the first number's last token), own {scope: best rank of the prompted keyword's set},
  own_pos (band-min rank of the prompted set at every position), best_sparse {word: [rank per scope]} for every set whose
  best rank (all positions) is <= 50, chance {size: {scope: [hit fraction of the 100 random sets at k = 1, 5, 20]}},
  top1 (per band layer, the top-1 whole-word token id at every position), resp_tokens (per-token decodes of the response).

usage: python planned_qwen25_keyword_sweep/kw_lens.py --gen GEN.jsonl --input GEN_INPUT.jsonl \
           --keywords planned_qwen25_keyword_sweep/keywords.json --out REC.jsonl [--shard i --nshards n]
"""
import argparse
import json
import os
import re
import sys
import zlib

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402
from sljlens.numbers import passes  # noqa: E402
from sljlens.readout import draw_random_sets, lens_logits, residuals, response_positions, ww_ranks  # noqa: E402
from sljlens.vocab import whole_word_mask  # noqa: E402

torch.set_grad_enabled(False)
DIG = re.compile(r"\d")
SCOPES = ("all", "ex0", "exfirst")
BIG = 10 ** 9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="scripts/generate.py output")
    ap.add_argument("--input", required=True, help="the generation input (build_gen.py), for the messages")
    ap.add_argument("--keywords", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen25_7b")
    ap.add_argument("--lens", default=None, help="lens file (default: the registry lens of --model)")
    ap.add_argument("--band", default=None, help="comma-separated layers (default: the registry band, L22-26)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    band = [int(x) for x in a.band.split(",")] if a.band else models.cfg(a.model)["band"]
    hf, tok, model = models.load_model(a.model)
    lens = models.load_lens(a.lens or a.model)
    V = model._lm_head.weight.shape[0]
    wwmask = whole_word_mask(tok, V).cuda()
    KW = json.load(open(a.keywords))
    widx = {k["word"]: i for i, k in enumerate(KW)}
    for k in KW:   # the sets in keywords.json must be whole-word tokens of this tokenizer
        assert all(bool(wwmask[t]) for t in k["ids"]) and sorted(tok.decode([t]) for t in k["ids"]) == k["tokens"], k
    U = sorted({t for k in KW for t in k["ids"]})
    upos = {t: j for j, t in enumerate(U)}
    maxs = max(len(k["ids"]) for k in KW)
    pad = torch.tensor([[upos[k["ids"][min(c, len(k["ids"]) - 1)]] for c in range(maxs)] for k in KW]).cuda()  # [S, maxs]
    u_t = torch.tensor(U).cuda()
    sizes = sorted({len(k["ids"]) for k in KW})
    ww_list = wwmask.nonzero().squeeze(-1).tolist()
    rand = {n: R.cuda() for (_, n), R in draw_random_sets(ww_list, (0, sizes)).items()}

    def readout(msgs, text):
        p = models.render_ids(tok, msgs, a.model)
        r = tok(text, add_special_tokens=False)["input_ids"]
        pos = response_positions(len(p), len(r))
        rs = [tok.decode([t]) for t in r]
        i0 = next((i for i, s in enumerate(rs) if DIG.search(s)), None)
        e1 = len(r)
        if i0 is not None:
            e1 = i0
            while e1 + 1 < len(r) and rs[e1 + 1].strip().isdigit():
                e1 += 1
        acts = residuals(model, p + r, band)
        bU, bR, top1 = None, {n: None for n in sizes}, []
        for l in band:
            ranks = ww_ranks(lens_logits(model, lens, acts[l][pos], l, V), wwmask)
            top1.append(ranks.argmin(-1).tolist())          # the token at rank 1
            ru = ranks[:, u_t]
            bU = ru if bU is None else torch.minimum(bU, ru)
            for n, R in rand.items():
                rr = ranks[:, R].amin(-1)
                bR[n] = rr if bR[n] is None else torch.minimum(bR[n], rr)
        S = bU[:, pad].amin(-1)                               # [P, S] band-min rank per position per keyword set
        sl = {"all": slice(0, None), "ex0": slice(1, None), "exfirst": slice(e1 + 1, None)}
        best = {sc: (S[s].amin(0) if S[s].shape[0] else torch.full((S.shape[1],), BIG, device=S.device))
                for sc, s in sl.items()}
        sparse = {KW[j]["word"]: [int(best[sc][j]) for sc in SCOPES]
                  for j in (best["all"] <= 50).nonzero().squeeze(-1).tolist()}
        chance = {}
        for n in sizes:
            chance[n] = {}
            for sc, s in sl.items():
                sub = bR[n][s]
                if sub.shape[0] == 0:
                    chance[n][sc] = [0.0, 0.0, 0.0]
                    continue
                m = sub.amin(0)
                chance[n][sc] = [round(float((m <= k).float().mean()), 3) for k in (1, 5, 20)]
        return dict(n_pos=len(pos), e1=e1, sparse=sparse, S=S, best=best, chance=chance, top1=top1, resp_tokens=rs)

    msgs_by_id = {}
    for line in open(a.input):
        q = json.loads(line)
        if zlib.crc32(q["id"].encode()) % a.nshards == a.shard:
            msgs_by_id[q["id"]] = q["messages"]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    cnt = 0
    with open(a.out, "a") as f:
        for line in open(a.gen):
            g = json.loads(line)
            if zlib.crc32(g["id"].encode()) % a.nshards != a.shard or g["id"] in done:
                continue
            m, o = g["meta"], g["outputs"][0]
            ok = o["finish_reason"] == "stop" and passes(o["text"])
            rec = dict(id=g["id"], **m, passed=bool(ok), text=o["text"][:400])
            if ok:
                res = readout(msgs_by_id[g["id"]], o["text"])
                j = widx[m["word"]]
                rec.update(n_pos=res["n_pos"], e1=res["e1"], own={sc: int(res["best"][sc][j]) for sc in SCOPES},
                           own_pos=res["S"][:, j].tolist(), best_sparse=res["sparse"], chance=res["chance"],
                           top1=res["top1"], resp_tokens=res["resp_tokens"])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cnt += 1
            if a.limit and cnt >= a.limit:
                break
    print("DONE", cnt, flush=True)


if __name__ == "__main__":
    main()
