#!/usr/bin/env python3
"""Official router fit (workspace_lens ActivationRouterCollection: unit norm, PCA-64 via covariance eigh, k-means++ with
seed 0, 100 iterations) for several K from ONE activation collection (the official collect_valid_position_activations),
with each layer's points moved to the GPU before the unchanged fit code runs on them. Only the device differs from
jpp_cli fit-router (CPU), which was CPU-bound for > 40 min per K on a 64-core machine. Round 2 used it for every router.
usage (jpp_lens repo root, jpp environment):
  python appendix_h_jpp_rlens/fit_router_gpu.py POOL.json N_PROMPTS MAX_SEQ_LEN OUT_PREFIX K [K ...]   -> OUT_PREFIX_K<K>.pt
"""
import json
import sys

import torch as t
from workspace_lens import get_hf_model
from workspace_lens.routing.cluster import collect_valid_position_activations
from workspace_lens.routing.router import ActivationRouterCollection


class _ToGPU(dict):
    def items(self):
        for layer, acts in super().items():
            yield layer, acts.to("cuda")


pool, n, max_len, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
Ks = [int(k) for k in sys.argv[5:]]
model = get_hf_model("Qwen/Qwen2.5-7B-Instruct", attn_implementation="sdpa")
prompts = json.load(open(pool))[:n]
acts = collect_valid_position_activations(model, prompts, list(range(27)), skip_first_n_positions=16, max_seq_len=max_len)
print("collected", {l: tuple(a.shape) for l, a in list(acts.items())[:2]}, flush=True)
for K in Ks:
    router = ActivationRouterCollection(K, 64, seed=0)
    router.fit(_ToGPU(acts))
    for lr in router.layer_routers_L_dict.values():  # store on CPU like the CPU fit
        lr.pca_mean_N, lr.pca_components_ND, lr.centroids_ED = lr.pca_mean_N.cpu(), lr.pca_components_ND.cpu(), lr.centroids_ED.cpu()
    router.save(f"{prefix}_K{K}.pt")
    print("saved", f"{prefix}_K{K}.pt", flush=True)
