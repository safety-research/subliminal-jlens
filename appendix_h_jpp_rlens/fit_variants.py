#!/usr/bin/env python3
"""Round-2 J++ weight fits for one expert set, in one process (model and experts loaded once), using the official
library stages (jpp_cli.fit_weights / jpp_cli.evaluate, unchanged) with three weight variants (jpp_variants: signed =
official, nonneg, best) and three item protocols:
  official  fit on fit:0 (20%), score held-out:0 (80%)          -> comparable with round 1
  cv        5 stratified folds: fit on 4/5, score the 5th; ranks pooled over folds -> cross-validated recall, every
            scored item unseen by the weights that scored it
  all       fit on all scoreable items -> the deployed lens (saved, fp32, plus jlens-format copy for the project)
Also scores the experts' pooled LRP Jacobian (the R-Lens of this fit) on held-out:0 and on all items (unsupervised,
so all-items is its cross-validated score). Ranks go to <out>_ranks.csv with columns variant, protocol.
The deployed lenses are also written in the reference jlens format (fp32) to --lens-dir as
Qwen2.5-7B-Instruct_r2_<tag>_<variant>.pt and Qwen2.5-7B-Instruct_r2_<tag>_rlens.pt (the post's J++ Lens is tag wlK16,
variant signed; its R-Lens is tag wlK8, rlens).
usage (jpp_lens repo root, jpp environment):
  python appendix_h_jpp_rlens/fit_variants.py EXPERTS.pt TAG OUT_DIR --correctness-csv model_correctness_q25.csv \
      --cache-path readout_residuals_cache.pt --lens-dir $SLJ_LENS_DIR [--variants signed,nonneg,best]
"""
import argparse
import os
import sys

import pandas as pd
import torch as t

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jpp_cli  # noqa: E402
import jpp_run  # noqa: E402,F401  (installs the kfold item spec)
import jpp_variants  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.fitting.condense_experts import CondenseConfig, ExpertJacobians  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("experts")
ap.add_argument("tag")
ap.add_argument("out_dir")
ap.add_argument("--variants", default="signed,nonneg,best")
ap.add_argument("--correctness-csv", required=True, help="written by jpp_run.py grade")
ap.add_argument("--cache-path", required=True, help="the official readout-residual cache (created on first use)")
ap.add_argument("--lens-dir", required=True, help="where the jlens-format copies go (SLJ_LENS_DIR)")
a = ap.parse_args()
M = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = list(range(27))
os.makedirs(a.out_dir, exist_ok=True)
model = get_hf_model(M, attn_implementation="sdpa")
items, _, excl = jpp_cli.load_items_and_exclusions(
    model, eval_data_dir=jpp_cli.READOUT_EVALS_DATA_DIR, correctness_csv=a.correctness_csv, hf_model_name=M,
    layers=LAYERS, readout_max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN)
scoreable = jpp_cli.readout_runner(model, layers=LAYERS, max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN).scoreable_items(items)
experts = ExpertJacobians.load(a.experts)
cache = a.cache_path
sel = lambda spec: jpp_cli.select_items(scoreable, spec, fit_fraction=0.2)  # noqa: E731
rows = []


def score(lens, name, test_items, variant, protocol):
    ranks, _ = jpp_cli.evaluate(model, {name: lens}, test_items, layers=LAYERS, excluded_token_ids=excl, ks=[1, 10])
    ranks["variant"], ranks["protocol"], ranks["tag"] = variant, protocol, a.tag
    rows.append(ranks)


pooled = experts.pooled_lens(checkpoint_name=f"{a.tag}_pooled")
score(pooled, f"{a.tag}_rlens", sel("held-out:0"), "pooled", "official")
score(pooled, f"{a.tag}_rlens", scoreable, "pooled", "all_items")
pooled.save(f"{a.out_dir}/{a.tag}_pooled.pt", dtype=t.float32)
for variant in a.variants.split(","):
    jpp_variants.set_variant(variant)
    cfg = CondenseConfig(steps=0 if variant == "best" else CondenseConfig().steps)
    plan = [("fit:0", "held-out:0", "official")] + [(f"kfold:{i}:5:train", f"kfold:{i}:5:test", "cv") for i in range(5)] \
        + [("all", None, "all")]
    for train_spec, test_spec, protocol in plan:
        train = sel(train_spec)
        weighting, lens = jpp_cli.fit_weights(model, experts, items, layers=LAYERS, excluded_token_ids=excl,
                                              cache_path=cache, checkpoint_name=f"{a.tag}_{variant}_{protocol}",
                                              hf_model_name=M, fitting_item_names=[it.name for it in train],
                                              condense_config=cfg)
        print(f"[{a.tag} {variant} {train_spec}] n_train={len(train)}", flush=True)
        if test_spec:
            score(lens, f"{a.tag}_{variant}", sel(test_spec), variant, protocol)
        else:
            out = f"{a.out_dir}/{a.tag}_{variant}_all.pt"
            lens.save(out, dtype=t.float32)
            weighting.save(f"{a.out_dir}/{a.tag}_{variant}_all_expert_weights.json")
            J = {int(l): m.float() for l, m in lens.jacobians_L_dict_FN.items()}
            t.save({"J": J, "n_prompts": int(lens.config.num_prompts_trained_on), "source_layers": sorted(J),
                    "d_model": next(iter(J.values())).shape[0]}, f"{a.lens_dir}/Qwen2.5-7B-Instruct_r2_{a.tag}_{variant}.pt")
        pd.concat(rows).to_csv(f"{a.out_dir}/{a.tag}_{a.variants.replace(',', '-')}_ranks.csv", index=False)
J = {int(l): m.float() for l, m in pooled.jacobians_L_dict_FN.items()}
t.save({"J": J, "n_prompts": int(pooled.config.num_prompts_trained_on), "source_layers": sorted(J),
        "d_model": next(iter(J.values())).shape[0]}, f"{a.lens_dir}/Qwen2.5-7B-Instruct_r2_{a.tag}_rlens.pt")
print("DONE", flush=True)
