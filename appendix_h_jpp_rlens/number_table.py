#!/usr/bin/env python3
"""The post's number-task table for the J-Lens / R-Lens / J++ Lens comparison (jpp/nopen/table.py of the original runs),
from the rate JSONs written by rates_band.py: RATES_DIR/rates_<owl|ctl>_<lens key>_<band>.json.
Rows: each lens at its own read band (J-Lens L22-26, R-Lens and J++ L21-26) and at all layers. Rounding as in the post
(two decimals below 1, one below 10, none above). Also prints the unrounded table with position 0 excluded and the
median best rank.
usage: python appendix_h_jpp_rlens/number_table.py RATES_DIR [--keys orig=qwen25_7b,rlens=r2_wlK8_rlens,jpp=r2_wlK16_signed]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("rates_dir")
ap.add_argument("--keys", default="orig=qwen25_7b,rlens=r2_wlK8_rlens,jpp=r2_wlK16_signed",
                help="file-name keys of the three lenses")
a = ap.parse_args()
K = dict(kv.split("=") for kv in a.keys.split(","))
ROWS = [("J-Lens (ours)", K["orig"], "22-26", "L22-26"), ("R-Lens", K["rlens"], "21-26", "L21-26"),
        ("J++ Lens (best)", K["jpp"], "21-26", "L21-26"), ("J-Lens (ours)", K["orig"], "all", "all layers"),
        ("R-Lens", K["rlens"], "all", "all layers"), ("J++ Lens (best)", K["jpp"], "all", "all layers")]


def f(x):
    return f"{x:.2f}" if x < 1 else (f"{x:.1f}" if x < 10 else f"{x:.0f}")


def trip(o, p):
    return " / ".join(f(o[f"{p}{k}"]) for k in (1, 5, 20)) + "%"


def full(o, p):
    return " / ".join(f"{o[f'{p}{k}']:.2f}" for k in (1, 5, 20))


def load(d, key, band):
    return json.load(open(f"{a.rates_dir}/rates_{d}_{key}_{band}.json"))


print("| lens | layers | owl teacher | chance | control teacher (no system prompt) |\n| --- | --- | --- | --- | --- |")
for name, key, band, lab in ROWS:
    ow, ct = load("owl", key, band), load("ctl", key, band)
    print(f"| {name} | {lab} | {trip(ow, 'top')} | {trip(ow, 'chance')} | {trip(ct, 'top')} |")
print("\nUnrounded (top-1 / 5 / 20, %):\n")
print("| lens | layers | owl | chance | owl excl. pos 0 (chance) | median best rank | control | control's own chance | n |\n"
      "|---|---|---|---|---|---|---|---|---|")
for name, key, band, lab in ROWS:
    ow, ct = load("owl", key, band), load("ctl", key, band)
    nop = (" / ".join(f"{ow[f'top{k}_nop0']:.2f}" for k in (1, 5, 20)) + " (" +
           " / ".join(f"{ow[f'chance{k}_nop0']:.2f}" for k in (1, 5, 20)) + ")")
    print(f"| {name} | {lab} | {full(ow, 'top')} | {full(ow, 'chance')} | {nop} | {ow['median_best_rank']:g} | "
          f"{full(ct, 'top')} | {full(ct, 'chance')} | {ow['n']} / {ct['n']} |")
