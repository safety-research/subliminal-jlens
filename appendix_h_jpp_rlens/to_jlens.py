"""Convert a J++-library lens file (BaseLens.save: {"parameters": {"jacobians": {l: J}}, "source_layers", "config"})
to the reference jlens format used by this release (sljlens.models.load_lens) ({"J": {l: J}, "n_prompts", "source_layers", "d_model"}).
Both libraries transport as h @ J_l.T and unembed with the model's final norm + LM head, so the maps are used
unchanged. Stored fp32 (the J++ README warns fp16 rounding of a combined map flips near-tie ranks).
usage: python to_jlens.py in.pt out.pt
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=True)
J = {int(l): m.float() for l, m in ck["parameters"]["jacobians"].items()}
d = next(iter(J.values())).shape[0]
torch.save({"J": J, "n_prompts": int(ck["config"].get("num_prompts_trained_on", 0)), "source_layers": sorted(J),
            "d_model": d}, dst)
print(dst, "layers", sorted(J)[0], "..", sorted(J)[-1], "n", len(J), "d", d, "config", {k: ck["config"][k] for k in
      ("checkpoint_name", "lrp_mode", "num_prompts_trained_on", "skip_first_n_positions", "max_seq_len")})
