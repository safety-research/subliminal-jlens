"""'Best expert per layer' J++ variant (an ablation in the J++ paper's Table 4: 50.2 vs 55.2 recall@10 on Qwen3.6-27B):
the LRP expert Jacobians from the official fit, combined with a one-hot weight on the expert with the highest
stand-alone per-layer pass@10 on the fit:0 items, as computed and logged by the official ExpertWeightingFitter
("layer N: single-expert pass@10 [...]", ties to the lower index as in build_search_starting_points).
No signed mixing, so no cancellation of the experts' shared component.
usage (jpp_lens root, jpp environment): python best_expert.py <log of the rlens fit-weights stage> <experts_rlens.pt> <out.pt>
"""
import re
import sys

import torch as t
from workspace_lens.fitting.condense_experts import ExpertJacobians

log, experts_path, out = sys.argv[1:4]
scores = {}
for line in open(log):  # first block of the log is the rlens fit-weights stage
    m = re.search(r"layer (\d+): single-expert pass@10 \[([^\]]*)\]", line)
    if m:
        layer = int(m.group(1))
        if layer in scores:
            break
        scores[layer] = [float(x) for x in m.group(2).split(",")]
experts = ExpertJacobians.load(experts_path)
weights = {}
for layer in experts.layers:
    s = scores[layer]
    best = sorted(range(len(s)), key=lambda e: (-s[e], e))[0]
    w = t.zeros(len(s))
    w[best] = 1.0
    weights[layer] = w
    print(f"layer {layer}: best expert {best} (fit-split pass@10 {s[best]:.3f}; all {s})")
lens = experts.combine(weights, checkpoint_name="jpp_bestexpert_rlens")
lens.save(out, dtype=t.float32)
print("saved", out)
