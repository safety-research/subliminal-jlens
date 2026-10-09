#!/usr/bin/env python3
"""Fit a Jacobian lens with the reference implementation (jlens.fit), optionally one shard of the prompts.

Corpus 'wikitext' (the recipe of every own-fit lens in the write-up, and of the Neuronpedia lenses): WikiText-103 train
records with >= 600 characters, truncated to 2000 characters, the first N of them; max_seq_len 128; jlens's default
skip_first (16 positions). A JSON file instead gives a list of texts (the lens-refit corpora of appendix_lens_refit).
Sharding: prompts[shard::nshards]; merge the shards with scripts/merge_lens.py (n_prompts-weighted mean). The Qwen2.5-7B
lens was fit in 4 shards with dim_batch 32 (any sharding gives the same lens up to float error).

usage: python scripts/fit_lens.py --model qwen25_7b --out LENS.pt [--corpus wikitext|corpus.json] [--n 300]
           [--shard i --nshards n] [--dim-batch 32] [--max-seq-len 128] [--source-layers 45,48,...]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--model-path", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--corpus", default="wikitext")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--dim-batch", type=int, default=32)
ap.add_argument("--max-seq-len", type=int, default=128)
ap.add_argument("--source-layers", default=None)
a = ap.parse_args()
assert torch.cuda.is_available()
import jlens  # noqa: E402

jlens.configure_logging()


def wikitext_prompts(n, split="train"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    out = []
    for t in ds["text"]:
        if len(t.strip()) >= 600:
            out.append(t[:2000])
            if len(out) == n:
                break
    return out


prompts = wikitext_prompts(a.n) if a.corpus == "wikitext" else json.load(open(a.corpus))
prompts = prompts[a.shard:: a.nshards]
hf, tok, model = models.load_model(a.model, model_path=a.model_path)
if getattr(tok, "bos_token_id", None) is not None and hasattr(tok, "add_bos_token"):
    tok.add_bos_token = True          # raw-text prompts keep a BOS when the tokenizer has one (jlens default)
out = a.out if a.nshards == 1 else f"{a.out}.shard{a.shard}of{a.nshards}"
SL = [int(x) for x in a.source_layers.split(",")] if a.source_layers else None
print(model, "prompts", len(prompts), "->", out, flush=True)
lens = jlens.fit(model, prompts, source_layers=SL, dim_batch=a.dim_batch, max_seq_len=a.max_seq_len,
                 checkpoint_path=out + ".ckpt", checkpoint_every=25)
lens.save(out)
print("saved", out, lens, flush=True)
