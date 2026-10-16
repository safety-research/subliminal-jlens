#!/usr/bin/env python3
"""GPU readout of the on-policy number completions of the strong-lens models (strong/lens_readout.py and
strong/granular/granular_readout.py of the original runs, merged; --granular selects the second).

For every completion of lens_gen.jsonl (build_sft.py; the first 3 filter-passing samples per prompt of the 300-completion
set's 100 number prompts, under none / sys_cat / sys_penguin / sys_horse), teacher-forced under the prompt that produced it:
  per response position (the position predicting response token j; j = 0 is the last prompt token) and per layer (every
  source layer of each lens, plus the output layer read directly = the model's own logits), the whole-word rank of the best
  token of each animal set and of 100 size-matched random whole-word sets (sljlens.strong for the vocabulary, set and rank
  conventions). Several lenses share one forward pass. --logit adds a plain logit lens ("logit": final norm + unembedding of
  the residual, no Jacobian) at every layer 0..L-2.
Sets: cat, penguin, horse; --granular adds owl, dog, lion, eagle, wolf, elephant, dolphin, tiger and bear (for the
cross-animal check). The random sets are drawn from one random.Random(0) in order of the sorted set sizes, so --granular
also draws different random sets (chance values of the two readouts differ slightly).
Output (torch.save): {"recs": [{id, cond, item, sample, text, spans, cur, lens: {name: {"layers": [... , L-1],
"tgt": {set: int16 [P, nL]}, "rand": {size: uint8 [P, nL, 100]}}}}], "sets", "n_ww", "L", "lenses", "sanity", "rand_sizes"}.
Target ranks are clamped at 30,000 and random-set ranks at 255 (enough for every k <= 100 used).
Exclusion and statistics are done on the CPU (analyze_lens.py, granular/analyze_granular.py).

usage: python planned_strong_lens_models/strong_readout.py --model llama33_70b --gen lens_gen.jsonl \
           --lens np=Llama-3.3-70B-Instruct_jacobian_lens.pt [--lens ...] [--logit] [--granular] --out RAW_i.pt \
           [--shard i --nshards n] [--limit N]
--lens NAME=FILE: FILE is a path, or a file name under SLJ_LENS_DIR. Models that do not fit one GPU (Llama-3.3-70B) are
spread over all visible GPUs (device_map="auto").
"""
import argparse
import os
import random
import sys
import time
import zlib

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models, strong  # noqa: E402
from sljlens.io import LENS_DIR, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True, help="registry key (llama33_70b, qwen36_27b, qwen25_7b)")
ap.add_argument("--model-path", default=None)
ap.add_argument("--gen", required=True)
ap.add_argument("--lens", action="append", default=[], help="NAME=FILE (repeatable)")
ap.add_argument("--logit", action="store_true", help="add the plain logit lens at every layer")
ap.add_argument("--granular", action="store_true", help="the granular readout's 12 animal sets (and its random draws)")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

import jlens  # noqa: E402
import transformers  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

t0 = time.time()
path = a.model_path or models.cfg(a.model)["hf"]
cfg = transformers.AutoConfig.from_pretrained(path)
tok = transformers.AutoTokenizer.from_pretrained(path)
ngpu = torch.cuda.device_count()
Auto = transformers.AutoModelForImageTextToText if hasattr(cfg, "vision_config") else transformers.AutoModelForCausalLM
hf = Auto.from_pretrained(path, dtype=torch.bfloat16, device_map="auto" if ngpu > 1 else "cuda").eval()
model = jlens.from_hf(hf, tok, force_bos=False)
L = model.n_layers
W = model._lm_head.weight
V = W.shape[0]
W32 = W.detach().float()
DEV = W.device
print(f"loaded {type(hf).__name__} L={L} V={V} gpus={ngpu} ({time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def unembed32(h):
    return F.linear(model._final_norm(h.to(W.dtype).to(DEV)).float(), W32)


LENSES = {}
for spec in a.lens:
    n, p = spec.split("=", 1)
    LENSES[n] = strong.load_lens_matrices(p if os.path.isfile(p) else os.path.join(LENS_DIR, p), DEV)
    print(f"lens {n}: layers L{min(LENSES[n])}..L{max(LENSES[n])} ({len(LENSES[n])})", flush=True)
if a.logit:
    LENSES["logit"] = {l: None for l in range(L - 1)}  # identity map: the plain logit lens
    print(f"lens logit: layers L0..L{L - 2} ({L - 1})", flush=True)
LAYERS = sorted(set().union(*[set(J) for J in LENSES.values()]) | {L - 1})

wwmask = strong.whole_word_mask(tok, V)
NWW = int(wwmask.sum())
SETS = {w: strong.ww_ids(tok, wwmask, [w, pl]) for w, pl in (strong.GRANULAR_SETS if a.granular else strong.BASE_SETS)}
wwmask = wwmask.to(DEV)
print("whole-word tokens", NWW, "sets", {k: [tok.decode([i]) for i in v] for k, v in SETS.items()}, flush=True)
rng = random.Random(0)
ww_list = wwmask.nonzero().squeeze(-1).tolist()
SIZES = sorted({len(v) for v in SETS.values()})
RAND = {z: torch.tensor([rng.sample(ww_list, z) for _ in range(100)], device=DEV) for z in SIZES}
SETS_T = {k: torch.tensor(v, device=DEV) for k, v in SETS.items()}


@torch.no_grad()
def ranks_of(lg):
    """lg [P, V] fp32 -> {set: [P] rank among whole-word tokens}, {size: [P, 100]}; rank = 1 + #strictly larger logits"""
    m = lg.masked_fill(~wwmask, float("-inf"))
    asc = m.sort(-1).values

    def rk(best):
        flat = best.reshape(best.shape[0], -1)
        cnt = V - torch.searchsorted(asc, flat.contiguous(), right=True)
        return (cnt + 1).reshape(best.shape)
    out = {k: rk(m[:, ids].amax(-1, keepdim=True))[:, 0] for k, ids in SETS_T.items()}
    rnd = {z: rk(m[:, R].amax(-1)) for z, R in RAND.items()}
    return out, rnd


@torch.no_grad()
def acts_of(ids):
    t = torch.as_tensor(ids, device=model.input_device)[None]
    with ActivationRecorder(model.layers, at=LAYERS) as rec:
        model.forward(t)
        return {l: rec.activations[l][0].float().to(DEV) for l in LAYERS}


rows = [r for r in read_jsonl(a.gen) if zlib.crc32(r["id"].encode()) % a.nshards == a.shard]
if a.limit:
    rows = rows[:a.limit]
print(len(rows), "completions in shard", flush=True)
recs, san = [], []
for ri, r in enumerate(rows):
    p = models.render_ids(tok, r["messages"], a.model)
    enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
    resp, spans = enc["input_ids"], [list(x) for x in enc["offset_mapping"]]
    if not resp:
        continue
    seq = p + resp
    pos = torch.arange(len(p) - 1, len(p) - 1 + len(resp), device=DEV)
    acts = acts_of(seq)
    out_l = unembed32(acts[L - 1][pos])
    if ri < 3:  # sanity: our fp32 unembedding of block L-1 reproduces the HF logits (top-1 agreement)
        with torch.no_grad():
            ref = hf(input_ids=torch.as_tensor(seq, device=model.input_device)[None], use_cache=False).logits[0].float().to(DEV)
        san.append(float((ref[pos][:, :V].argmax(-1) == out_l.argmax(-1)).float().mean()))
    so, ro = ranks_of(out_l)
    out = {}
    for n, J in LENSES.items():
        tgt, rnd = {k: [] for k in SETS}, {z: [] for z in SIZES}
        for l in sorted(J):
            h = acts[l][pos]
            s, rr = ranks_of(unembed32(h if J[l] is None else h @ J[l].T))
            for k in SETS:
                tgt[k].append(s[k])
            for z in SIZES:
                rnd[z].append(rr[z])
        for k in SETS:
            tgt[k].append(so[k])
        for z in SIZES:
            rnd[z].append(ro[z])
        out[n] = dict(layers=sorted(J) + [L - 1],
                      tgt={k: torch.stack(v, 1).clamp(max=30000).to(torch.int16).cpu() for k, v in tgt.items()},
                      rand={z: torch.stack(v, 1).clamp(max=255).to(torch.uint8).cpu() for z, v in rnd.items()})
    recs.append(dict(id=r["id"], cond=r["cond"], item=r["item"], sample=r["sample"], text=r["text"], spans=spans,
                     cur=[seq[q] for q in pos.tolist()], lens=out))
    if ri % 50 == 0:
        print(f"{ri}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
print("SANITY output-layer top-1 agreement with HF logits:", san, flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save(dict(recs=recs, sets={k: [tok.decode([i]) for i in v] for k, v in SETS.items()}, n_ww=NWW, L=L,
                lenses=list(LENSES), sanity=san, rand_sizes=SIZES), a.out)
print("DONE", len(recs), f"{time.time() - t0:.0f}s", flush=True)
