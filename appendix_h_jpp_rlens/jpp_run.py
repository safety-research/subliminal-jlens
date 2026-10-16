#!/usr/bin/env python3
"""Thin wrapper around the official J++ Lens pipeline (github.com/safety-research/jpp_lens, commit 93f02cd,
scripts/jpp_cli.py). Run it from the jpp_lens repo root, with that repo's own environment (built from its uv.lock).

Two additions, neither of which changes any estimator:
  1. JPP_PROMPTS_JSON: the fit-router stage streams its 1000 WikiText prompts from the HF Hub; if this variable is set,
     load_fit_prompts returns the first N prompts of that JSON file instead (made by prep_wikitext.py: exactly
     jlens.examples.load_wikitext_prompts(1000)), so GPU jobs need no network access. fit-shard already has --prompts-json.
  2. A 'grade' stage that writes the model-correctness CSV column for our model with the library's own
     ReadoutEvalRunner.grade_model_correctness (the shipped CSV only grades Qwen3.6-27B).
Two round-2 additions (opt-in; the variants are labelled wherever they are used):
  3. JPP_NONNEG=1 / JPP_BEST_ONLY=1 switch the expert-weight search to the non-negative / best-single-expert variant
     (jpp_variants.py); unset = the official signed search.
  4. Item specs 'kfold:<i>:<k>:train|test' for select_items (5-fold cross-fitting of the weights in fit_variants.py).

usage: python jpp_run.py <jpp_cli stage> [args...]
       python jpp_run.py grade --hf-model-name Qwen/Qwen2.5-7B-Instruct --out model_correctness_q25.csv
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jpp_cli  # noqa: E402

PJ = os.environ.get("JPP_PROMPTS_JSON")
if PJ:
    _pool = json.load(open(PJ))

    def _load_fit_prompts(num_prompts):
        if len(_pool) < num_prompts:
            raise ValueError(f"{PJ} has {len(_pool)} prompts, {num_prompts} requested")
        return _pool[:num_prompts]

    jpp_cli.load_fit_prompts = _load_fit_prompts


import jpp_variants  # noqa: E402

jpp_variants.set_variant("nonneg" if os.environ.get("JPP_NONNEG") else "best" if os.environ.get("JPP_BEST_ONLY") else "signed")
_orig_select = jpp_cli.select_items


def _select_items(items, items_spec, *, fit_fraction):
    """Adds 'kfold:<i>:<k>:train|test': stratified by eval (one numpy default_rng(0) permutation per eval, evals in sorted
    order, as split_fitting_items); fold i of k is the test fold, train = the other k-1 folds."""
    if not items_spec.startswith("kfold:"):
        return _orig_select(items, items_spec, fit_fraction=fit_fraction)
    import numpy as np
    _, i, k, part = items_spec.split(":")
    i, k = int(i), int(k)
    by = {}
    for it in items:
        by.setdefault(it.eval_slug, []).append(it)
    rng = np.random.default_rng(0)
    test = set()
    for slug in sorted(by):
        perm = rng.permutation(len(by[slug])).tolist()
        for j, idx in enumerate(perm):
            if j % k == i:
                test.add((slug, by[slug][idx].name))
    return [it for it in items if ((it.eval_slug, it.name) in test) == (part == "test")]


jpp_cli.select_items = _select_items


def grade(argv):
    import argparse

    import pandas as pd
    from lens_evals.readout_evals.readout_eval_items import (MODEL_CORRECTNESS_CSV, READOUT_EVALS_DATA_DIR,
                                                             load_readout_eval_items)
    from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
    from workspace_lens import get_hf_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model-name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model = get_hf_model(a.hf_model_name, attn_implementation="sdpa")
    items = load_readout_eval_items(READOUT_EVALS_DATA_DIR)
    runner = ReadoutEvalRunner(model, {}, layers=[0], max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN)
    df = runner.grade_model_correctness(items, hf_model_name=a.hf_model_name)
    shipped = pd.read_csv(MODEL_CORRECTNESS_CSV)
    shipped.merge(df, on=["eval", "item"], how="outer").to_csv(a.out, index=False)
    for ev, g in df.groupby("eval"):
        c = g[a.hf_model_name]
        print(ev, "n", len(g), "correct", int((c == True).sum()), "NA", int(c.isna().sum()))  # noqa: E712
    print("wrote", a.out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "grade":
        grade(sys.argv[2:])
    else:
        raise SystemExit(jpp_cli.main())
