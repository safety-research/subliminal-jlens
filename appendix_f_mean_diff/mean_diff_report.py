"""Turn mean_diff_lens.py analyze output (JSON) into markdown tables.
usage: python appendix_f_mean_diff/mean_diff_report.py results/mean_diff_lens.json > tables.md
"""
import json
import sys

import numpy as np

R = json.load(open(sys.argv[1]))
BAND = R["band"]
LAY = R["layers"]
bidx = [LAY.index(l) for l in BAND]
VAR_DESC = {"lensmean": "lens of the mean (pooled positions)", "lensmean_mom": "lens of the mean (mean of per-completion means)",
            "lin_gain": "linear: W(g*J dmean)", "lin_nogain": "linear: W J dmean (no gain)",
            "meanlens": "mean of the lens (pooled)", "meanlens_mom": "mean of the lens (per-completion means)",
            "lensmean8": "lens of the mean, first 8 positions", "meanlens8": "mean of the lens, first 8 positions",
            "lensmean_x0": "lens of the mean, excluding position 0", "meanlens_x0": "mean of the lens, excluding position 0",
            "lensmean1to8": "lens of the mean, positions 1-8", "meanlens1to8": "mean of the lens, positions 1-8"}


def pct(xs, q):
    return int(np.percentile(xs, q))


def null_stats(task, var, an, stat):
    nl = R["nulls"][task]["split"][var][an]
    xs = [d[stat] for d in nl]
    return xs


def pval(xs, obs):
    k = sum(x <= obs for x in xs)
    return f"<{1 / len(xs):.3f}" if k == 0 else f"{k / len(xs):.3f}"


def main_table(task, comps, var, stat="band_avg_ww"):
    nww = R["n_wholeword"]
    print(f"\n**{VAR_DESC[var]}**, statistic: {stat} (rank among {nww} whole-word tokens; lower = more owl-like)\n")
    print("| diff | null: split-half median [5th pct; min] | animal | rank | p (split-half) | p (sign-flip) | bootstrap 5-95% | random 2-sets beating it | per band layer 22-26 | all-layer avg diff |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cname in comps:
        key = f"{task}: {cname}"
        if key not in R["comps"]:
            continue
        c = R["comps"][key]
        for an, s in c["var"].get(var, {}).items():
            if an == "top20":
                continue
            obs = s[stat]
            xs = null_stats(task, var, an, stat)
            fl = R["nulls"][task]["flip"].get(var, {}).get(an)
            pf = f"{pval([d[stat] for d in fl], obs)}" if fl else "-"
            bt = c["boot"].get(var, {}).get(an)
            bs = f"{pct([d[stat] for d in bt], 5)}-{pct([d[stat] for d in bt], 95)}" if bt else "-"
            rb = s["rand_beat"]["band_avg" if stat.startswith("band") else "all_avg"]
            print(f"| {cname} (n={c['nA']} vs {c['nB']}) | {pct(xs, 50)} [{pct(xs, 5)}; min {min(xs)}] | {an} | **{obs}** | {pval(xs, obs)} | {pf} | {bs} | {100 * rb:.0f}% | "
                  f"{', '.join(str(s['per_layer_ww'][i]) for i in bidx)} | {s['all_avg_ww']} |")


def per_layer(task, cname, var, an):
    c = R["comps"][f"{task}: {cname}"]
    s = c["var"][var][an]
    return s["per_layer_ww"]


def tops(task, cname, var, keys=None):
    c = R["comps"][f"{task}: {cname}"]
    t = c["var"][var]["top20"]
    print(f"\n*{task}, {cname}, {VAR_DESC[var]}*\n")
    for k in keys or list(t):
        print(f"- L{k}: " + ", ".join(repr(w)[1:-1].strip() for w, _ in t[k]) if k.isdigit() else
              f"- {k}: " + ", ".join(repr(w)[1:-1].strip() for w, _ in t[k]))


WRITEUP_ROWS = [("sys_owl-none", "owl", "", "subliminal learning system prompt minus no system prompt, owl"),
                ("sys_owl-none", "owl", "8", "same, first 8 positions only"),
                ("sys_horse-none", "horse", "", 'subliminal learning system prompt with "horses" minus no system prompt, horse'),
                ("sys_owl-sys_horse", "owl", "", 'subliminal learning system prompt minus the "horses" version, owl'),
                ("user_owl-none", "owl", "", '"Think about owls." at the start of the user turn minus no system prompt, owl'),
                ("OFF sys_owl-none", "owl", "", "off-policy: no-system-prompt completions read under the subliminal learning "
                                                "system prompt minus read with no system prompt, owl")]


def writeup_table(task="numbers", stat="band_min_ww"):
    """The write-up's table: at each of layers 22-26 the logit difference is ranked, and the entry is the best rank over
    those layers (band_min_ww); the baseline is the same statistic on 1,000 split-half diffs of the no-system-prompt
    completions (median and best), for lens of the mean / mean of the lens."""
    print(f"\n## Write-up table ({task}): best rank over layers {BAND[0]}-{BAND[-1]} among {R['n_wholeword']} whole-word tokens\n")
    print("| Diff (condition minus condition), token read | baseline: median (best of 1,000), lens of the mean / mean of the lens "
          "| lens of the mean | mean of the lens |\n|---|---|---|---|")
    for comp, an, suf, label in WRITEUP_ROWS:
        key = f"{task}: {comp}"
        if key not in R["comps"]:
            continue
        cells, base = [], []
        for var in ("lensmean" + suf, "meanlens" + suf):
            cells.append(str(R["comps"][key]["var"][var][an][stat]))
            xs = null_stats(task, var, an, stat)
            base.append(f"{int(np.median(xs)):,} ({min(xs)})")
        print(f"| {label} | {' / '.join(base)} | {cells[0]} | {cells[1]} |")


if __name__ == "__main__":
    writeup_table()
    for task in ("numbers", "open"):
        comps = [k.split(": ", 1)[1] for k in R["comps"] if k.startswith(task + ":")]
        print(f"\n## {task}\n")
        for var in VAR_DESC:
            main_table(task, comps, var)
        main_table(task, comps, "lensmean", "band_min_ww")
        main_table(task, comps, "meanlens", "band_min_ww")
        print(f"\n### per-layer owl rank (whole-word), {task}, sys_owl-none\n")
        for var in ("lensmean", "meanlens", "lin_gain"):
            if f"{task}: sys_owl-none" in R["comps"]:
                print(f"- {var}: " + " ".join(f"L{l}:{r}" for l, r in zip(LAY, per_layer(task, "sys_owl-none", var, "owl"))))
        if "perpos" in R:
            print(f"\n### per position (index-aligned), {task}: band-avg whole-word rank, lens of the mean / mean of the lens\n")
            keys = [k for k in R["perpos"] if k.startswith(task + ":")]
            print("| pos | " + " | ".join(k.split(": ", 1)[1] for k in keys) + " |")
            print("|---" * (len(keys) + 1) + "|")
            for j in range(len(R["perpos"][keys[0]])):
                print(f"| {j} | " + " | ".join(f"{R['perpos'][k][j]['lensmean']['band_avg_ww']} / {R['perpos'][k][j]['meanlens']['band_avg_ww']}" for k in keys) + " |")
            for k in keys:
                if "top_band" in R["perpos"][k][0]:
                    print(f"\n*{k}, mean of the lens, band-avg top-10 at positions 0-3:* " + " || ".join(
                        ", ".join(w.strip() for w, _ in R["perpos"][k][j]["top_band"]) for j in range(4)))
        print(f"\n### top-20 whole-word tokens, {task}\n")
        for cname in comps:
            for var in ("lensmean", "meanlens"):
                tops(task, cname, var)
