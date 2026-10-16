import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# (label, top1, top5, top20) — completion-level lens hit rates on the teacher's own number completions
rows = [
    ("Qwen2.5-7B\nowl", 1.65, 3.15, 6.37),
    ("Qwen2.5-7B\npenguin", 1.0, 2.7, 5.7),
    ("Qwen2.5-7B\ncat", 0.7, 1.7, 5.7),
    ("Qwen3.5-9B\npanda", 83, 92, 98),
    ("Qwen3-32B\nfox", 100, 100, 100),
    ("Gemma-4-31B\nwolf", 91, 100, 100),
    ("Gemma-4-31B\nwillow", 73, 99, 99),
    ("Gemma-4-31B\nsequoia", 12, 85, 99),
    ("Gemma-4-31B:\nrandom two-token\nset, all layers", 1.2, 2.6, 6.4),
    ("Gemma-4-31B\n\"Think about\nwillows.\"*\n(10.5 → 14.3%)", 1, 34, 45),
    ("Gemma-4-31B\nsequoia,\n\"never mention\"*\n(26.7 → 36.3%)", 1, 13, 67),
]
x = np.arange(len(rows)); w = 0.27
fig, ax = plt.subplots(figsize=(11, 4.4), dpi=200)
cols = ["#3b6ea5", "#6f9fd1", "#b5d3ef"]
for j, (lab, c) in enumerate(zip(["top-1", "top-5", "top-20"], cols)):
    vals = [r[1 + j] for r in rows]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=lab, color=c)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:g}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=7)
ax.set_ylabel("% of the teacher's number completions where\nthe target is in the lens top-k (any read layer, any position)", fontsize=8)
ax.set_ylim(0, 108); ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.axvline(7.5, color="gray", lw=0.6, ls=":")
fig.tight_layout(); HERE = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(HERE, "fig2_other_models.png"))
