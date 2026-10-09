#!/usr/bin/env python3
"""Thin wrapper around the official J++ Lens pipeline (github.com/safety-research/jpp_lens, commit 93f02cd,
scripts/jpp_cli.py). Run it from the jpp_lens repo root, with that repo's own environment (built from its uv.lock).

Two additions, neither of which changes any estimator:
  1. JPP_PROMPTS_JSON: the fit-router stage streams its 1000 WikiText prompts from the HF Hub; if this variable is set,
     load_fit_prompts returns the first N prompts of that JSON file instead (made by prep_wikitext.py: exactly
     jlens.examples.load_wikitext_prompts(1000)), so GPU jobs need no network access. fit-shard already has --prompts-json.
  2. A 'grade' stage that writes the model-correctness CSV column for our model with the library's own
     ReadoutEvalRunner.grade_model_correctness (the shipped CSV only grades Qwen3.6-27B).

usage: python jpp_run.py <jpp_cli stage> [args...]
       python jpp_run.py grade --hf-model-name Qwen/Qwen2.5-7B-Instruct --out model_correctness_q25.csv
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import jpp_cli  # noqa: E402

PJ = os.environ.get("JPP_PROMPTS_JSON")
if PJ:
    _pool = json.load(open(PJ))

    def _load_fit_prompts(num_prompts):
        if len(_pool) < num_prompts:
            raise ValueError(f"{PJ} has {len(_pool)} prompts, {num_prompts} requested")
        return _pool[:num_prompts]

    jpp_cli.load_fit_prompts = _load_fit_prompts


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
