#!/usr/bin/env python3
"""Home-turf readout recall from fit_variants.py rank files: item-weighted recall@k, macro over the five evals, per
(tag, variant, protocol); scopes = the paper's 7 equally spaced layers (4,7,10,14,18,21,24) and all 27 layers.
Protocols: official (fit:0 -> held-out:0, 296 items), cv (5-fold cross-fitted, all 366 items scored once by weights
that did not see them), all_items (pooled R-Lens, unsupervised, all 366 items). Also per-layer recall@10 (cv / all).
usage: python appendix_h_jpp_rlens/r2_home.py RANKS.csv [RANKS.csv ...]   (needs pandas + tabulate)
"""
import sys

import pandas as pd

EV = ["multihop", "multilingual", "typo", "association", "poetry"]
df = pd.concat([pd.read_csv(f) for f in sys.argv[1:]])


def macro(d, layers, k):
    m = d[d.layer.isin(layers)].groupby(["eval", "item", "intermediate"])["rank"].min()
    r = (m <= k).groupby(["eval", "item"]).mean().groupby("eval").mean()
    return 100 * r.reindex(EV).mean(), d.item.nunique()


rows = []
for (tag, var, prot), d in df.groupby(["tag", "variant", "protocol"]):
    r7_1, n = macro(d, [4, 7, 10, 14, 18, 21, 24], 1)
    r7_10, _ = macro(d, [4, 7, 10, 14, 18, 21, 24], 10)
    ra_10, _ = macro(d, list(range(27)), 10)
    pl = [round(macro(d, [L], 10)[0]) for L in (6, 10, 16, 22, 24, 26)]
    rows.append(dict(tag=tag, variant=var, protocol=prot, n_items=n, r7_at1=round(r7_1, 1), r7_at10=round(r7_10, 1),
                     all_at10=round(ra_10, 1), per_layer_L6_10_16_22_24_26=" / ".join(map(str, pl))))
print(pd.DataFrame(rows).sort_values(["protocol", "r7_at10"], ascending=[True, False]).to_markdown(index=False))
