#!/usr/bin/env python3
"""Batched completion-level lens readout: for every completion, the best whole-word rank of each token set over every
response position and every layer of a band, plus the same for 100 random whole-word sets (chance). Used for the
1.42M-completion owl-teacher pool behind the Result table (03) and the addendum's lens selection (06: that pool, and the
Gemma-4-31B sequoia and "Think about willows." pools) and for the J++ / R-Lens comparison (appendix_h_jpp_rlens).

Rule (identical to the per-position readout of scripts/lens_readout.py, src/animals_lens_pos.py and the multi-model
readouts): each completion is teacher-forced under the prompt that produced it (chat template with the model's chat
kwargs, tokenized with add_special_tokens=False; response tokenized separately and appended); read positions = every
position predicting a response token (position 0 = the last prompt token); lens logits = model.unembed(J_l h_l) in fp32
(identity transport at a layer the lens does not cover, e.g. the output layer); ranks among whole-word tokens
(`^ [A-Za-z]{3,}$`); a set's rank = its best member; best over band layers x positions.
Differences from the per-position collector, for speed only: band layers only; sequences batched (right padding + attention
mask, sorted by length); rank = 1 + the number of whole-word tokens with a strictly larger logit (the per-position collector
uses argsort, which breaks bf16 ties by sort order). Batch composition can move bf16 near-ties, so a handful of
completions flip between rank 1 and rank 2-3 between runs (19 of 27,622 in one comparison; see DISCREPANCIES.md).

Chance: random whole-word sets drawn as `--rand-plan SEED:SIZE,SIZE,...` (sequential draws from one random.Random(SEED),
100 sets per size, the originals' draws), scored under the identical rule for the sizes in `--rand-score` (default: the
primary set's size).

Output (torch.save): {"recs": [...], "band", "sets", "primary", "n_ww", "rand_sets", "args"}; per record: key
'<id>|onpolicy|<sample>', id, sample, n_pos, passes (the paper's number filter), mention (regex count), and for the primary
set best / best_nop0 / mat (int16 [P, len(band)]) / rand_best / rand_best_nop0 (int32 [100]); for every set
best_<name> / best_nop0_<name> / pos0_<name> (and mat_<name> with --all-mats); rand<n>_best / rand<n>_best_nop0 for every
scored size; with --rand-per-layer rand_mat / rand_mat_nop0 (int32 [len(band), 100], primary size).

Settings of the write-up's pools: Qwen2.5-7B owl defaults (owl set, band L22-26, --rand-plan 0:2); Gemma-4-31B sequoia
--sets 'sequoia,sequoiapiece=Sequoia|sequ' --rand-plan 0:1,3,4,6 --rand-score 1,3 --batch-tokens 12000; Gemma-4-31B willow
--sets willow --rand-plan 0:1,2,3,4,86,92,98,100 --rand-score 2,3 --batch-tokens 12000 (the original tree readout's draw
order, so the size-2 sets are the second draw).

usage: python scripts/score_lens.py --model qwen25_7b --gen GEN.jsonl --prompts PROMPTS.jsonl --out shard.pt
           [--sets owl] [--band 22,...] [--lens KEY_OR_PATH] [--shard i --nshards n] [--batch-tokens 40000]
GEN: scripts/generate.py output ({"id", "outputs": [{"text"}]}); PROMPTS: {"id", "messages"} rows it was generated from.
"""
import argparse
import json
import os
import re
import sys
import time
import zlib

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models, readout, vocab  # noqa: E402
from sljlens.numbers import passes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--lens", default=None, help="lens registry key or file (default: the model's lens)")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="owl", help="token-set spec (sljlens.vocab.parse_set_spec); the first is primary")
    ap.add_argument("--band", default=None, help="layers to read, e.g. 22,23,24,25,26 or 0-26 (default: the model's band)")
    ap.add_argument("--rand-plan", default=None, help="SEED:SIZES random-set draw (default 0:<primary set size>)")
    ap.add_argument("--rand-score", default=None, help="random-set sizes to score (default: the primary set size)")
    ap.add_argument("--rand-per-layer", action="store_true", help="also store per-layer random-set minima (primary size)")
    ap.add_argument("--all-mats", action="store_true", help="store the [P, band] rank matrix of every set, not only the primary")
    ap.add_argument("--mention", default=None, help="regex counted in each completion (default: the primary word + plural)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-tokens", type=int, default=40000, help="tokens per forward (12000 was used for Gemma-4-31B)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    assert torch.cuda.is_available()

    hf, tok, model = models.load_model(a.model, model_path=a.model_path)
    lens = models.load_lens(a.lens or a.model)
    if a.band and "-" in a.band:
        lo, hi = (int(x) for x in a.band.split("-"))
        BAND = list(range(lo, hi + 1))
    else:
        BAND = [int(x) for x in a.band.split(",")] if a.band else models.cfg(a.model)["band"]
    V = model._lm_head.weight.shape[0]
    wwmask = vocab.whole_word_mask(tok, V)
    ww_list = wwmask.nonzero().squeeze(-1).tolist()
    ww_idx = torch.tensor(ww_list).cuda()
    NW = len(ww_list)
    pos_in_ww = {t: j for j, t in enumerate(ww_list)}
    SETS = vocab.build_sets(tok, wwmask, None, a.sets) if "_nn100" not in a.sets else \
        vocab.build_sets(tok, wwmask.cuda(), model._lm_head.weight.float(), a.sets)
    primary = next(iter(SETS))
    SET_J = {k: torch.tensor([pos_in_ww[t] for t in v]).cuda() for k, v in SETS.items()}
    psize = len(SETS[primary])
    plan = readout.parse_plans([a.rand_plan or f"0:{psize}"])[0]
    drawn = readout.draw_random_sets(ww_list, plan)                       # {(0, size): [100, size]}
    RSZ = [int(x) for x in a.rand_score.split(",")] if a.rand_score else [psize]
    assert psize in RSZ, "the primary set's size must be scored (it defines rand_best)"
    RAND_J = {n: torch.tensor([[pos_in_ww[t] for t in s] for s in drawn[(0, n)].tolist()]).cuda() for n in RSZ}
    pw = vocab.parse_set_spec(a.sets)[0][2]
    mention = re.compile(a.mention or r"\b(" + "|".join(re.escape(w) for w in pw) + r")\b", re.I)
    print(f"model {a.model} lens {models.lens_path(a.lens or a.model)} band {BAND} whole-word vocab {NW}", flush=True)
    print("sets", {k: [tok.decode([i]) for i in v] for k, v in SETS.items()}, "primary", primary, "random plan", plan,
          "scored sizes", RSZ, flush=True)

    prompts = {}
    with open(a.prompts) as f:
        for line in f:
            r = json.loads(line)
            prompts[r["id"]] = r["messages"]
    items = []
    with open(a.gen) as f:
        for line in f:
            g = json.loads(line)
            if g["id"] not in prompts or zlib.crc32(str(g["id"]).encode()) % a.nshards != a.shard:
                continue
            for s, o in enumerate(g["outputs"]):
                items.append((f"{g['id']}|onpolicy|{s}", g["id"], s, o["text"]))
    if a.limit:
        items = items[: a.limit]
    print(len(items), "completions in this shard", flush=True)

    ptoks, recs, n_empty = {}, [], 0
    for key, gid, s, text in items:
        if gid not in ptoks:
            ptoks[gid] = models.render_ids(tok, prompts[gid], a.model)
        p = ptoks[gid]
        r = tok(text, add_special_tokens=False)["input_ids"]
        if not r:
            n_empty += 1
            continue
        recs.append(dict(key=key, id=gid, sample=s, seq=p + r, start=len(p) - 1, n_pos=len(r),
                         passes=passes(models.content_of(a.model, text)), mention=len(mention.findall(text))))
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]["seq"]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0, done, i, nb = time.time(), 0, 0, 0
    with torch.no_grad():
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(recs[order[j]]["seq"]) <= a.batch_tokens:
                j += 1
            j = max(j, i + 1)
            bidx = order[i:j]
            Lmax = max(len(recs[b]["seq"]) for b in bidx)
            ids = torch.full((len(bidx), Lmax), pad, dtype=torch.long)
            att = torch.zeros((len(bidx), Lmax), dtype=torch.long)
            for row, b in enumerate(bidx):
                sq = recs[b]["seq"]
                ids[row, : len(sq)] = torch.tensor(sq)
                att[row, : len(sq)] = 1
            with readout.ActivationRecorder(model.layers, at=BAND) as rec:
                model._text_module(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False)
                acts = {l: rec.activations[l] for l in BAND}
            rr, pp = [], []
            for row, b in enumerate(bidx):
                st, n = recs[b]["start"], recs[b]["n_pos"]
                rr += [row] * n
                pp += list(range(st, st + n))
            rr_t, pp_t = torch.tensor(rr).cuda(), torch.tensor(pp).cuda()
            per = {k: [] for k in SETS}
            perr = {n: [] for n in RSZ}
            for l in BAND:
                h = acts[l][rr_t, pp_t].float()
                z = model.unembed(lens.transport(h, l) if l in lens.jacobians else h).float()[:, :V]
                w = z[:, ww_idx]                                                  # [N, NW]
                del z
                for k, jj in SET_J.items():
                    thr = w[:, jj].amax(-1)
                    per[k].append(1 + (w > thr[:, None]).sum(-1))
                ws, _ = w.sort(-1)
                for n in RSZ:
                    rthr = w[:, RAND_J[n]].amax(-1)                                # [N, 100]
                    perr[n].append(NW - torch.searchsorted(ws, rthr.contiguous(), right=True) + 1)
                del w, ws
            M = {k: torch.stack(v, 1).cpu() for k, v in per.items()}               # [N, L]
            RL = {n: torch.stack(v, 1).cpu() for n, v in perr.items()}             # [N, L, 100]
            off = 0
            for b in bidx:
                n = recs[b]["n_pos"]
                rc = recs[b]
                for k in SETS:
                    m = M[k][off: off + n]
                    pb = m.amin(1)
                    if k == primary or a.all_mats:
                        rc[f"mat_{k}"] = m.clamp(max=30000).to(torch.int16)
                    rc[f"best_{k}"] = int(pb.min())
                    rc[f"best_nop0_{k}"] = int(pb[1:].min()) if n > 1 else 99999
                    rc[f"pos0_{k}"] = int(pb[0])
                for nn in RSZ:
                    rL = RL[nn][off: off + n]                                       # [P, L, 100]
                    rm = rL.amin(1)                                                # band-min per position
                    rc[f"rand{nn}_best"] = rm.amin(0).clamp(max=60000).to(torch.int32)
                    rc[f"rand{nn}_best_nop0"] = rm[1:].amin(0).clamp(max=60000).to(torch.int32) if n > 1 else None
                    if a.rand_per_layer and nn == psize:
                        rc["rand_mat"] = rL.amin(0).clamp(max=60000).to(torch.int32)          # [L, 100]
                        rc["rand_mat_nop0"] = rL[1:].amin(0).clamp(max=60000).to(torch.int32) if n > 1 else None
                rc["best"], rc["best_nop0"] = rc[f"best_{primary}"], rc[f"best_nop0_{primary}"]
                rc["mat"] = rc.pop(f"mat_{primary}")
                rc["rand_best"], rc["rand_best_nop0"] = rc[f"rand{psize}_best"], rc[f"rand{psize}_best_nop0"]
                off += n
                del rc["seq"], rc["start"]
            done += len(bidx)
            i = j
            nb += 1
            if nb % 50 == 0:
                print(f"done {done}/{len(recs)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(dict(recs=recs, band=BAND, sets={k: [tok.decode([t]) for t in v] for k, v in SETS.items()}, primary=primary,
                    n_ww=NW, rand_sets={n: drawn[(0, n)].tolist() for n in RSZ}, n_empty=n_empty, args=vars(a)), a.out)
    print("DONE", len(recs), f"(skipped {n_empty} empty completions) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
