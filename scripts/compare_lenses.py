#!/usr/bin/env python3
"""Compare two lenses layer by layer (multimodel/lens/compare_lens.py): per layer of B, ||J||/sqrt(d) for both, the cosine
of the flattened maps cos(A, B), cos(A^T, B), cos(A - I, B - I), and cos with A at neighbouring layers (|dl| <= 3).
Used for the Gemma-4-31B sanity checks: our all-layer lens vs our earlier partial fit (cosine 0.999-1.000) and vs the
lamm-mit lens (cosine 0.21-0.26 at L28-57, ~0 at L0-26).
usage: python scripts/compare_lenses.py LENS_A LENS_B      (registry keys or files)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402

import jlens  # noqa: E402

A = jlens.JacobianLens.load(models.lens_path(sys.argv[1]))
B = jlens.JacobianLens.load(models.lens_path(sys.argv[2]))
print(A, B)


def cos(x, y):
    return float((x * y).sum() / (x.norm() * y.norm()))


for l in B.source_layers:
    b = B.jacobians[l].float()
    row = {"norm/sqrt(d) B": float(b.norm() / b.shape[0] ** .5)}
    if l in A.jacobians:
        x = A.jacobians[l].float()
        eye = torch.eye(x.shape[0])
        row.update({"norm/sqrt(d) A": float(x.norm() / x.shape[0] ** .5), "cos": cos(x, b), "cos_AT": cos(x.T, b),
                    "cos_A-I_B-I": cos(x - eye, b - eye)})
    for la in A.source_layers:
        if abs(la - l) <= 3 and la != l:
            row[f"cos_A{la}"] = cos(A.jacobians[la].float(), b)
    print(l, {k: round(v, 3) for k, v in row.items()}, flush=True)
