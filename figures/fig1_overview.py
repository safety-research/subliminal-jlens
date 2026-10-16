import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
groups = ["Chance\n(random set of two tokens,\nsame task as\nsubliminal learning)",
          "Qwen2.5-7B lens,\nsubliminal learning\nteacher writing numbers",
          "Qwen2.5-7B lens,\nJ-Lens toy task\n(\"think about X\")",
          "Claude Haiku 4.5,\nJ-Lens toy task\n(paper, Fig. 10)"]
top1 = [0.26, 1.65, 45, 93]
top5 = [1.50, 3.15, 74, None]
x = np.arange(len(groups)); w = 0.36
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
b1 = ax.bar(x - w/2, top1, w, label="top-1", color="#3b6ea5")
b5 = ax.bar(x + w/2, [v if v is not None else 0 for v in top5], w, label="top-5", color="#9cc3e6")
for rect, v in zip(list(b1) + list(b5), top1 + top5):
    if v is None:
        ax.text(rect.get_x() + rect.get_width()/2, 1.5, "not\nreported", ha="center", va="bottom", fontsize=7, color="gray")
    else:
        ax.text(rect.get_x() + rect.get_width()/2, v + 1.2, f"{v:g}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
ax.set_ylabel("% of trials where the target is in the lens top-k\n(any read layer, any position)", fontsize=8)
ax.set_ylim(0, 105); ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); HERE = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(HERE, "fig1_overview.png"))
