#!/usr/bin/env python3
"""GPU collector: teacher-forced Jacobian-lens readouts of generated completions, stored per position.

One pass stores everything the analyses need, so every table in the write-up is computed offline from these files:
  for every completion (and readout variant) and every lens layer, the whole-word rank of each named token set at each
  read position (int16 [P, L], clamped at 32767), and for each random token set (100 sets per size) the per-position
  best rank over the read band (uint8 [P, 100], clamped at 255) and the best rank over all positions and all lens layers
  (int16 [100]), plus the token ids / strings / character spans needed for the word-level exclusion rules.

Variants (src/rerun_lens.py):
  onpolicy   each sample teacher-forced under the prompt that produced it; positions predicting each response token
  fixed      (off-policy diagnostic) the --fixed-cond condition's sample 0 for the same task/item, under this condition
  sysprompt  for sys_* conditions: the positions of the system-prompt content tokens (independent of the user turn)
--with-output adds the output layer read directly (the Gemma-4-31B all-layer table's "model output" columns);
--rand-per-layer stores rand_layer[rid] / rand_layer_x0[rid] (int16 [L, 100]: best rank of each random set per layer over all
positions / over positions >= 1), so chance can be computed exactly for any layer subset with or without the first position.

usage: python scripts/lens_readout.py --model qwen25_7b --gen GEN.jsonl:PROMPTS.jsonl [--gen ...] --sets SPEC \
           --out OUT.pt [--lens KEY_OR_PATH] [--band 22,...] [--variants onpolicy,fixed,sysprompt] [--tasks ...] \
           [--conds ...] [--rand-plan SEED:SIZES ...] [--shard i --nshards n]
GEN is the generation output (scripts/generate.py format: {"id", "outputs": [{"text", ...}], "meta"}), PROMPTS the input
it was generated from ({"id", "messages", ...}).
"""
import argparse
import os
import sys
import time
import zlib

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models, readout, vocab  # noqa: E402
from sljlens.io import read_jsonl  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--model-path", default=None, help="local weights dir (default: the registry's HF id)")
ap.add_argument("--lens", default=None, help="lens registry key or file (default: the model's lens)")
ap.add_argument("--gen", action="append", required=True, help="GEN.jsonl:PROMPTS.jsonl (repeatable)")
ap.add_argument("--sets", required=True, help="token sets, e.g. owl,owl_broad,owl_nn100,horse,horse_broad,horse_nn100")
ap.add_argument("--band", default=None, help="read band (default: the model's)")
ap.add_argument("--variants", default="onpolicy,fixed,sysprompt")
ap.add_argument("--tasks", default=None)
ap.add_argument("--conds", default=None)
ap.add_argument("--sys-tasks", default="numbers", help="tasks whose sys_* prompts get a sysprompt readout")
ap.add_argument("--sys-items", default=None, help="only these items for the sysprompt readout (e.g. 0)")
ap.add_argument("--fixed-cond", default="none")
ap.add_argument("--rand-plan", action="append", default=None,
                help="SEED:SIZE,SIZE,... random-set draw plan (repeatable); default 0:<sorted unique set sizes>")
ap.add_argument("--fp32", action="store_true", help="rank fp32 lens logits (tie diagnostic for Gemma's softcap)")
ap.add_argument("--with-output", action="store_true",
                help="also read the output layer directly (no lens: the model's own next-token logits) as the last layer")
ap.add_argument("--rand-per-layer", action="store_true",
                help="also store, per layer, each random set's best rank over all positions and over positions >= 1")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=None, help="stop after this many readouts (smoke tests)")
ap.add_argument("--keys", default=None, help="file with record keys ('<id>|<variant>|<sample>') to restrict to (smoke tests)")
a = ap.parse_args()
assert torch.cuda.is_available()
t0 = time.time()

hf, tok, model = models.load_model(a.model, model_path=a.model_path)
lens = models.load_lens(a.lens or a.model)
LAYERS = sorted(lens.jacobians)
if a.with_output and model.n_layers - 1 not in LAYERS:
    LAYERS.append(model.n_layers - 1)
BAND = [int(x) for x in a.band.split(",")] if a.band else models.cfg(a.model)["band"]
BIDX = [LAYERS.index(l) for l in BAND if l in LAYERS]
V = model._lm_head.weight.shape[0]
wwmask = vocab.whole_word_mask(tok, V).cuda()
W_U = model._lm_head.weight.float()
SETS = vocab.build_sets(tok, wwmask, W_U, a.sets)
del W_U
SETS_T = {k: torch.tensor(v).cuda() for k, v in SETS.items()}
ww_list = wwmask.nonzero().squeeze(-1).tolist()
plans = readout.parse_plans(a.rand_plan) if a.rand_plan else [(0, sorted({len(v) for v in SETS.values()}))]
drawn = readout.draw_random_sets(ww_list, plans)
# dedupe identical draws (e.g. plan (0,[2,3,55]) and (0,[2,3,4]) share their first two sizes)
RAND, plan_map = [], {}
for k, R in drawn.items():
    for rid, R2 in enumerate(RAND):
        if R.shape == R2.shape and torch.equal(R, R2):
            plan_map[k] = rid
            break
    else:
        plan_map[k] = len(RAND)
        RAND.append(R)
RAND_T = [R.cuda() for R in RAND]
print(f"model {a.model} lens layers {LAYERS[0]}..{LAYERS[-1]} ({len(LAYERS)}), band {BAND}, whole-word vocab {len(ww_list)}",
      flush=True)
print("sets", {k: (len(v), [tok.decode([i]) for i in v][:8]) for k, v in SETS.items()}, flush=True)
print("random plans", plans, "->", {f"{k[0]}:{k[1]}": v for k, v in plan_map.items()}, flush=True)

prompts, gens = {}, {}
for spec in a.gen:
    gp, pp = spec.split(":")
    for r in read_jsonl(pp):
        prompts[r["id"]] = r
    for g in read_jsonl(gp):
        gens[g["id"]] = g
TASKS = set(a.tasks.split(",")) if a.tasks else None
CONDS = set(a.conds.split(",")) if a.conds else None
VARIANTS = a.variants.split(",")
SYS_TASKS = set(a.sys_tasks.split(","))
SYS_ITEMS = {int(x) for x in a.sys_items.split(",")} if a.sys_items else None


@torch.no_grad()
def collect(msgs, text, kind):
    ptext, p, offs = models.render_ids(tok, msgs, a.model, with_offsets=True)
    spans = None
    if text is not None:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        r, spans = enc["input_ids"], [list(x) for x in enc["offset_mapping"]]
    else:
        r = []
    seq = p + r
    if kind == "sys":
        pos = readout.system_positions(ptext, offs, msgs[0]["content"])
    else:
        pos = readout.response_positions(len(p), len(r))
    if not pos:
        return None
    assert pos == list(range(pos[0], pos[-1] + 1))
    chain = seq[pos[0]: pos[-1] + 2]                 # token at each position, plus the token after the last one
    acts = readout.residuals(model, seq, LAYERS)
    ranks = {k: [] for k in SETS}
    rb_pos = [None] * len(RAND)
    r_all = [None] * len(RAND)
    r_layer = [[] for _ in RAND]
    r_layer_x0 = [[] for _ in RAND]
    for li, l in enumerate(LAYERS):
        z = readout.lens_logits(model, lens, acts[l][pos], l, V, fp32=a.fp32)
        rk = readout.ww_ranks(z, wwmask)
        del z
        for k, ids in SETS_T.items():
            ranks[k].append(rk[:, ids].amin(-1).clamp(max=32767).to(torch.int16).cpu())
        for j, R in enumerate(RAND_T):
            rr = rk[:, R].amin(-1)                                    # [P, 100]
            m = rr.amin(0)
            r_all[j] = m if r_all[j] is None else torch.minimum(r_all[j], m)
            if a.rand_per_layer:
                r_layer[j].append(m)
                r_layer_x0[j].append(rr[1:].amin(0) if rr.shape[0] > 1 else torch.full_like(m, 10 ** 6))
            if li in BIDX:
                rb_pos[j] = rr if rb_pos[j] is None else torch.minimum(rb_pos[j], rr)
        del rk
    del acts
    rec = dict(n_pos=len(pos), chain_ids=chain, chain_strs=[tok.decode([t]) for t in chain],
               ranks={k: torch.stack(v, 1) for k, v in ranks.items()},
               rand_band_pos=[x.clamp(max=255).to(torch.uint8).cpu() for x in rb_pos],
               rand_all=[x.clamp(max=32767).to(torch.int16).cpu() for x in r_all])
    if a.rand_per_layer:
        rec["rand_layer"] = [torch.stack(x).clamp(max=32767).to(torch.int16).cpu() for x in r_layer]        # [L, 100]
        rec["rand_layer_x0"] = [torch.stack(x).clamp(max=32767).to(torch.int16).cpu() for x in r_layer_x0]
    if spans is not None:
        rec["spans"] = spans
    return rec


jobs = []
for gid, g in gens.items():
    m = g.get("meta", {})
    if gid not in prompts or (TASKS and m.get("task") not in TASKS) or (CONDS and m.get("cond") not in CONDS):
        continue
    msgs = prompts[gid]["messages"]
    if "onpolicy" in VARIANTS:
        jobs += [(gid, "onpolicy", s, o["text"], msgs, m) for s, o in enumerate(g["outputs"])]
    if "fixed" in VARIANTS:
        fid = f"{m['task']}|{m['item']}|{a.fixed_cond}"
        if fid in gens:
            jobs.append((gid, "fixed", 0, gens[fid]["outputs"][0]["text"], msgs, m))
    if ("sysprompt" in VARIANTS and str(m.get("cond", "")).startswith("sys_") and m.get("task") in SYS_TASKS
            and (SYS_ITEMS is None or m.get("item") in SYS_ITEMS)):
        jobs.append((gid, "sysprompt", 0, None, msgs, m))
jobs = [j for j in jobs if zlib.crc32(f"{j[0]}|{j[1]}|{j[2]}".encode()) % a.nshards == a.shard]
if a.keys:
    want = {l.strip() for l in open(a.keys) if l.strip()}
    jobs = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}" in want]
if a.limit:
    jobs = jobs[: a.limit]
print(f"{len(jobs)} readouts in shard {a.shard}/{a.nshards} ({time.time() - t0:.0f}s)", flush=True)

recs = []
for n, (gid, variant, s, text, msgs, m) in enumerate(jobs):
    res = collect(msgs, text, "sys" if variant == "sysprompt" else "resp")
    if res is None:
        continue
    recs.append(dict(key=f"{gid}|{variant}|{s}", id=gid, variant=variant, sample=s, text=text,
                     task=m.get("task"), item=m.get("item"), cond=m.get("cond"), **res))
    if n % 200 == 0:
        print(f"  {n}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save(dict(model=a.model, lens=models.lens_path(a.lens or a.model), layers=LAYERS, band=BAND,
                sets=SETS, set_tokens={k: [tok.decode([i]) for i in v] for k, v in SETS.items()},
                n_wholeword=len(ww_list), plans=plans, plan_map=plan_map, rand_sets=RAND,
                args=vars(a), recs=recs), a.out)
print(f"DONE {len(recs)} records -> {a.out} ({time.time() - t0:.0f}s)", flush=True)
