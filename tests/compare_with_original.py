"""Compare a readout made by scripts/lens_readout.py with the original scripts' stored per-completion outputs
(rerun_lens.py shards, onpolicy_masked.py / onpolicy_conservative.py raw files, animals_lens.py shards).
usage: python tests/compare_with_original.py READOUT.pt ORIGINAL_RESULTS_DIR"""
import glob, json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.readouts import Readouts
R0 = sys.argv[2]
new = Readouts([sys.argv[1]])
L = new.layers
print("layers", L[0], L[-1], "band", new.band, "plan_map", new.plan_map)
newby = {r["key"]: r for r in new.recs}
ok = bad = 0
def check(name, a, b):
    global ok, bad
    if a == b: ok += 1
    else:
        bad += 1; print("MISMATCH", name, a, b)
# 1. rerun shards: per-layer best (all positions) + per-record chance
rerun = {}
for p in glob.glob(f"{R0}/rerun/shard*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if r["key"] in newby: rerun[r["key"]] = r
print("rerun matches available:", len(rerun))
for k, o in rerun.items():
    n = newby[k]
    for s in o["best"]:
        check(f"{k} best {s}", o["best"][s], [int(x) for x in n["ranks"][s].int().amin(0)])
        # masked best (cur or next in set)
        S = set(new.sets[s]); ids = n["chain_ids"]
        keep = torch.tensor([not (c in S or x in S) for c, x in zip(ids[:-1], ids[1:])])
        mb = [int(n["ranks"][s].int()[keep][:, i].min()) if keep.any() else 10**9 for i in range(len(L))]
        check(f"{k} best_masked {s}", o["best_masked"][s], mb)
    for size, d in o["chance"].items():
        rid = new.rid(int(size), 0)
        for kk in (1, 5, 20):
            b = float((n["rand_band_pos"][rid].int().amin(0) <= kk).float().mean())
            a_ = float((n["rand_all"][rid].int() <= kk).float().mean())
            check(f"{k} chance {size} band {kk}", round(d["band"][str(kk)], 6), round(b, 6))
            check(f"{k} chance {size} all {kk}", round(d["all"][str(kk)], 6), round(a_, 6))
# 2. onpolicy_masked raw: per-position band ranks of owl/horse and size-2 random sets
for p in glob.glob(f"{R0}/opm/raw_*.pt"):
    for r in torch.load(p, weights_only=False)["recs"]:
        k = f"{r['task']}|{r['item']}|{r['cond']}|onpolicy|{r['sample']}"
        if k not in newby: continue
        n = newby[k]
        for s in ("owl", "horse"):
            check(f"{k} opm {s}", r[s].int().tolist(), n["ranks"][s].int()[:, new.bidx].clamp(max=30000).tolist())
        check(f"{k} opm rand", r["rand"].int().amin(1).tolist(), n["rand_band_pos"][new.rid(2, 0)].int().tolist())
        check(f"{k} opm toks", r["toks"], n["chain_strs"][1:])
# 3. onpolicy_conservative raw: owl_broad all layers per position + size-65 (plan 2) random min over layers
for p in glob.glob(f"{R0}/opc/raw_*.pt"):
    for r in torch.load(p, weights_only=False)["recs"]:
        k = f"{r['task']}|{r['item']}|{r['cond']}|onpolicy|{r['sample']}"
        if k not in newby: continue
        n = newby[k]
        check(f"{k} opc owl_broad", r["owl_broad"].int().tolist(), n["ranks"]["owl_broad"].int().clamp(max=30000).tolist())
        check(f"{k} opc rand all-min", r["rand_min"].int().amin(0).tolist(), n["rand_all"][new.rid(65, 2)].int().clamp(max=255).tolist())
# 4. animals shards
for p in glob.glob(f"{R0}/animals_lens/shard*.jsonl"):
    for l in open(p):
        o = json.loads(l)
        if o["key"] not in newby: continue
        n = newby[o["key"]]
        for s in o["best"]:
            check(f"{o['key']} animals best {s}", o["best"][s], [int(x) for x in n["ranks"][s].int().amin(0)])
        for size, d in o["chance"].items():
            rid = new.rid(int(size), 1)
            for kk in (1, 5, 20):
                check(f"{o['key']} animals chance {size} band {kk}", round(d["band"][str(kk)], 6),
                      round(float((n["rand_band_pos"][rid].int().amin(0) <= kk).float().mean()), 6))
print(f"checks: {ok} equal, {bad} mismatched")
