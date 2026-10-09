"""Per-completion comparison of two sets of scripts/score_lens.py files (e.g. a re-run against a reference run): best ranks,
filter flags, rank matrices, random-set minima, and top-k agreement.
usage: python tests/compare_scores.py "REFERENCE_GLOB" "NEW_GLOB"
"""
import glob, sys, torch
def load(p):
    return {r["key"]: r for f in sorted(glob.glob(p)) for r in torch.load(f, weights_only=False)["recs"]}
a, b = load(sys.argv[1]), load(sys.argv[2])
common = set(a) & set(b)
print("orig", len(a), "new", len(b), "common", len(common))
eq = sum(a[k]["best"] == b[k]["best"] for k in common)
eq0 = sum(a[k]["best_nop0"] == b[k]["best_nop0"] for k in common)
pas = sum(a[k]["passes"] == b[k]["passes"] for k in common)
mat = sum(torch.equal(a[k]["mat"], b[k]["mat"]) for k in common)
rnd = sum(torch.equal(a[k]["rand_best"], b[k]["rand_best"]) for k in common)
print(f"best equal {eq}/{len(common)}; best_nop0 equal {eq0}; passes equal {pas}; rank matrices identical {mat}; random-set best identical {rnd}")
for k in (1, 5, 20):
    print(f"top-{k} hit agreement: {sum((a[x]['best'] <= k) == (b[x]['best'] <= k) for x in common)}/{len(common)}")
