"""usage: python aa_sanity_summary.py results/aa_sanity/*.jsonl  -> % first word == owl (same parsing as analyze_eval)"""
import collections
import json
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import ANIMALS  # noqa: E402

ALIAS = {a + "s": a for a in ANIMALS}
ALIAS.update({"wolves": "wolf", "foxes": "fox", "butterflies": "butterfly", "dragonflies": "dragonfly"})
for p in sys.argv[1:]:
    rows = [json.loads(l) for l in open(p)]
    ws = [ALIAS.get(w, w) for w in ((re.findall(r"[a-z]+", r["response"].lower()) or [""])[0] for r in rows)]
    c = collections.Counter(ws)
    print(f"| {p.rsplit('/', 1)[-1][:-6]} | {len(rows)} | {100 * c['owl'] / len(rows):.1f} | {', '.join(f'{w} {100 * n / len(rows):.0f}%' for w, n in c.most_common(4))} |")
