#!/usr/bin/env python3
"""Merge lens shards (n_prompts-weighted mean). usage: python scripts/merge_lens.py LENS.pt NSHARDS"""
import sys

import jlens

out, n = sys.argv[1], int(sys.argv[2])
lens = jlens.JacobianLens.merge([jlens.JacobianLens.load(f"{out}.shard{i}of{n}") for i in range(n)])
lens.save(out)
print("merged", out, lens)
