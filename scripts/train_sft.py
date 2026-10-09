#!/usr/bin/env python3
"""Minimal packed LoRA SFT for the students (one GPU). Reimplements the semantics of the trainer the original runs used.

Recipe (the subliminal-learning reference repo's open-model config): LoRA r=8, alpha=8, dropout 0 on q/k/v/o/gate/up/down,
AdamW lr 2e-4 (betas 0.9/0.999, eps 1e-8, no weight decay), linear schedule with 5 warm-up steps, 3 epochs, 66 sequences
per optimizer step, grad-norm clip 1.0, bf16 autocast, max length 1024.
Data order: one independent permutation per epoch from random.Random(seed). Loss per optimizer step: summed next-token
cross-entropy over the step's target tokens / number of target tokens in the step (so it does not depend on how sequences
are grouped). Targets: the final assistant turn, i.e. full_render[len(prompt_render):] (for Qwen2.5: the response +
"<|im_end|>\\n"), or the loss_mask of pre-tokenised rows. Packing: each step's sequences are bin-packed (first-fit
decreasing, <= pack_tokens tokens per row) into rows with per-sequence position ids and flash-attention varlen boundaries;
no padding. Models whose kernels cannot isolate packed sequences (Qwen3.5's linear attention, Gemma-4 under sdpa) train
one unpadded sequence per forward instead, which gives the same loss.
flash-attn is optional: without a varlen kernel (or with --set pack=false, or --set attn='"sdpa"') each step's sequences are run as
right-padded micro-batches of at most pack_tokens tokens under sdpa with an attention mask. The loss is the same (padding
never enters the loss or the attention of real tokens); only the speed differs (packing is ~2-4x faster).

Rows: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} or {"input_ids": [...], "loss_mask": [...]}.
usage: python scripts/train_sft.py --model qwen25_7b --data sft_owl.jsonl --out runs/owl_s0 --seed 0 [--set key=json ...]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402
from sljlens.io import read_jsonl  # noqa: E402

DEFAULTS = dict(lora_r=8, lora_alpha=8, lora_dropout=0.0, lr=2e-4, epochs=3, batch_size=66, warmup_steps=5,
                lr_schedule="linear", max_grad_norm=1.0, weight_decay=0.0, adam_betas=[0.9, 0.999], adam_eps=1e-8,
                max_len=1024, pack=True, pack_tokens=16384, attn="flash_attention_2")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--model-path", default=None)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--set", action="append", default=[], help="override a default, e.g. pack_tokens=4096")
a = ap.parse_args()
mc = models.cfg(a.model)
cfg = dict(DEFAULTS)
cfg.update(mc["sft"].get("extra", {}))
if mc["sft"].get("pack") is False:
    cfg["pack"] = False
for kv in a.set:
    k, v = kv.split("=", 1)
    cfg[k] = json.loads(v)

random.seed(a.seed)
torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

path = a.model_path or mc["hf"]
tok = AutoTokenizer.from_pretrained(path)
kw = dict(mc.get("chat_kwargs", {}))


def chat_sequence(msgs):
    """(ids, [(lo, hi)]): target = the final assistant turn, full[len(prompt):]; both renders must be prefix-stable."""
    full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=False, **kw)
    p = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True, return_dict=False, **kw)
    full, p = list(getattr(full, "input_ids", full)), list(getattr(p, "input_ids", p))
    assert full[: len(p)] == p, "chat template is not prefix-stable; use pre-tokenised rows"
    return full, [(len(p), len(full))]


exs = []
for n, r in enumerate(read_jsonl(a.data)):
    if "messages" in r:
        ids, spans = chat_sequence(r["messages"])
    else:
        ids, mask = r["input_ids"], r["loss_mask"]
        spans, t = [], 0
        while t < len(ids):
            if mask[t]:
                s = t
                while t < len(ids) and mask[t]:
                    t += 1
                spans.append((max(s, 1), t))
            else:
                t += 1
    if len(ids) > cfg["max_len"]:
        raise SystemExit(f"row {n}: {len(ids)} tokens > max_len {cfg['max_len']}")
    exs.append(dict(ids=ids, spans=spans))
print(f"{len(exs)} examples, {sum(len(e['ids']) for e in exs)} tokens", flush=True)

def packing_kernel_available(attn):
    """Packing needs a varlen flash-attention kernel: the flash-attn package, or a hub kernel through the `kernels` package
    (e.g. --set attn='"kernels-community/flash-attn3"' on Hopper GPUs; fetched from huggingface.co on first use)."""
    import importlib.util
    if attn == "flash_attention_2":
        return importlib.util.find_spec("flash_attn") is not None
    if attn.startswith("kernels-community/"):
        return importlib.util.find_spec("kernels") is not None
    return False


# 'packed': flash-attention varlen rows; 'padded': right-padded micro-batches under sdpa (flash-attn missing);
# 'single': one sequence per forward (models whose kernels cannot isolate packed sequences)
if mc["sft"].get("pack") is False:
    MODE = "single"
elif cfg["pack"] and packing_kernel_available(cfg["attn"]):
    MODE = "packed"
else:
    MODE = "padded"
    if cfg["pack"] and cfg["attn"] != "sdpa":
        print(f"no varlen kernel for attn={cfg['attn']} (flash-attn not installed?): training with right-padded micro-batches "
              "under sdpa (same loss, slower than packing; `pip install flash-attn`, or on Hopper GPUs "
              "--set attn='\"kernels-community/flash-attn3\"' with the `kernels` package, to pack)", flush=True)
attn = cfg["attn"] if MODE == "packed" else "sdpa"
print(f"batching mode: {MODE} (attn_implementation={attn})", flush=True)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn).cuda()
model.config.use_cache = False
for p_ in model.parameters():
    p_.requires_grad_(False)
model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                                         target_modules=mc["sft"]["lora_targets"], task_type="CAUSAL_LM"))
base = model.get_base_model()
decoder = getattr(base.model, "language_model", base.model)
W = base.get_output_embeddings().weight
tc = base.config.get_text_config() if hasattr(base.config, "get_text_config") else base.config
softcap = getattr(tc, "final_logit_softcapping", None) or None
params = [p_ for p_ in model.parameters() if p_.requires_grad]
opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["adam_betas"]), eps=cfg["adam_eps"],
                        weight_decay=cfg["weight_decay"])
per_epoch = math.ceil(len(exs) / cfg["batch_size"])
total = per_epoch * cfg["epochs"]
assert cfg["lr_schedule"] == "linear"
sched = get_linear_schedule_with_warmup(opt, cfg["warmup_steps"], total)


def ffd(lengths, budget):
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins, room = [], []
    for i in order:
        for b in range(len(bins)):
            if room[b] >= lengths[i]:
                bins[b].append(i)
                room[b] -= lengths[i]
                break
        else:
            bins.append([i])
            room.append(budget - lengths[i])
    return [sorted(b) for b in bins]


def hidden(seqs):
    """Final-norm hidden states [T, H] for a list of sequences, concatenated in order (one packed row, one right-padded
    micro-batch, or one forward per sequence)."""
    if MODE == "padded":
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        mask = mask.cuda()
        out = decoder(input_ids=ids.cuda(), attention_mask=mask, use_cache=False)
        return out.last_hidden_state[mask.bool()]          # row-major: sequence 0's tokens, then sequence 1's, ...
    if MODE == "packed":
        lens_ = [len(s) for s in seqs]
        ids = torch.tensor([t for s in seqs for t in s], device="cuda")[None]
        pos = torch.cat([torch.arange(L) for L in lens_]).cuda()[None]
        cu = torch.zeros(len(lens_) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(torch.tensor(lens_, dtype=torch.int32), 0)
        cu = cu.cuda()
        out = decoder(input_ids=ids, position_ids=pos, cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=max(lens_),
                      max_length_k=max(lens_), attention_mask=None, use_cache=False)
        return out.last_hidden_state[0]
    return torch.cat([decoder(input_ids=torch.tensor([s], device="cuda"), use_cache=False).last_hidden_state[0] for s in seqs])


def padded_groups(lengths, budget):
    """Consecutive groups whose padded size (n_seqs x longest) stays within budget tokens."""
    groups, cur, longest = [], [], 0
    for i, L in enumerate(lengths):
        if cur and (len(cur) + 1) * max(longest, L) > budget:
            groups.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, L)
    if cur:
        groups.append(cur)
    return groups


def summed_ce(h, tgt, chunk=4096):
    tot = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        lg = (h[i: i + chunk] @ W.t()).float()
        if softcap:
            lg = softcap * torch.tanh(lg / softcap)
        tot = tot + torch.nn.functional.cross_entropy(lg, tgt[i: i + chunk], reduction="sum")
    return tot


os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "train_log.jsonl"), "w")
rng = random.Random(a.seed)
orders = []
for _ in range(cfg["epochs"]):
    o = list(range(len(exs)))
    rng.shuffle(o)
    orders.append(o)
model.train()
step, t0 = 0, time.time()
for ep, order in enumerate(orders):
    for b in range(0, len(order), cfg["batch_size"]):
        batch = [exs[i] for i in order[b: b + cfg["batch_size"]]]
        n_tgt = sum(hi - lo for e in batch for lo, hi in e["spans"])
        groups = (ffd([len(e["ids"]) for e in batch], cfg["pack_tokens"]) if MODE == "packed" else
                  padded_groups([len(e["ids"]) for e in batch], cfg["pack_tokens"]) if MODE == "padded" else
                  [[i] for i in range(len(batch))])
        loss_sum = 0.0
        for g in groups:
            seqs = [batch[i] for i in g]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = hidden([e["ids"] for e in seqs])
                idx, tgt, off = [], [], 0
                for e in seqs:
                    for lo, hi in e["spans"]:
                        idx += range(off + lo - 1, off + hi - 1)
                        tgt += e["ids"][lo:hi]
                    off += len(e["ids"])
                loss = summed_ce(h.index_select(0, torch.tensor(idx, device="cuda")), torch.tensor(tgt, device="cuda")) / n_tgt
            loss.backward()
            loss_sum += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        lr = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        log.write(json.dumps(dict(step=step, epoch=ep, loss=loss_sum, grad_norm=float(gn), lr=lr, target_tokens=n_tgt)) + "\n")
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{total} loss {loss_sum:.4f} lr {lr:.2e} ({time.time() - t0:.0f}s)", flush=True)
model.save_pretrained(os.path.join(a.out, "adapter"))
json.dump(dict(config=cfg, batching=MODE, model=path, data=a.data, seed=a.seed, steps=step, seconds=round(time.time() - t0)),
          open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print("SFT_DONE", a.out, flush=True)
