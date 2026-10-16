#!/usr/bin/env python3
"""The two round-2 fitting pools for the J++ / R-Lens fits (JSON lists of texts, read through JPP_PROMPTS_JSON and
--prompts-json by the official pipeline).

  wikitext_long600.json   the first 600 WikiText-103 train records (wikitext-103-raw-v1, streamed in the dataset's fixed
                          order) whose stripped text has >= 1,500 characters, text kept as is. The round-2 "WikiText
                          512 x 256" fits use the first 512 at 256 tokens (122k fit positions after the 16 skipped ones).
  chat_AB_packed256.json  the lens-refit corpora A and B of appendix_g_lens_refit (chat-formatted number conversations
                          under 11 animal prompts and none; WikiText; animal prose), concatenated A + B, shuffled with
                          random.Random(0), then packed greedily into texts of >= 256 Qwen2.5 tokens (314 texts in the
                          original run). The round-2 "in-domain chat" fits use all of them at 256 tokens.
usage (network access needed for the WikiText stream):
  python appendix_h_jpp_rlens/build_fit_pools.py OUTDIR CORPUS_A.json CORPUS_B.json
  (CORPUS_A/B = corpus_A_chatnum.json / corpus_B_mix.json written by appendix_g_lens_refit/build_fit_corpus.py)
"""
import json
import os
import random
import statistics
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

outdir, ca, cb = sys.argv[1:4]
os.makedirs(outdir, exist_ok=True)
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
wiki = []
for r in ds:
    if len(r["text"].strip()) >= 1500:
        wiki.append(r["text"])
        if len(wiki) == 600:
            break
json.dump(wiki, open(os.path.join(outdir, "wikitext_long600.json"), "w"))

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
chat = json.load(open(ca)) + json.load(open(cb))
random.Random(0).shuffle(chat)
packed, cur, n = [], "", 0
for x in chat:
    cur += x
    n += len(tok(x)["input_ids"])
    if n >= 256:
        packed.append(cur)
        cur, n = "", 0
json.dump(packed, open(os.path.join(outdir, "chat_AB_packed256.json"), "w"))
for name, d in (("wikitext (first 512)", wiki[:512]), ("chat packed", packed)):
    L = [len(tok(x)["input_ids"]) for x in d]
    print(f"{name}: {len(d)} texts, median {statistics.median(L)} tokens, min {min(L)}, share >= 256: "
          f"{sum(x >= 256 for x in L) / len(L):.2f}")
