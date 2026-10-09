"""Markdown fidelity table from lens_fidelity.py outputs.
usage: python lens_refit_fidelity_table.py NAME=fid.json [NAME=fid.json ...]
Rows: lens x position group; columns: layers 22-26 (top-1 agreement full vocab / KL full vocab nats), plus
band means of whole-word agreement and whole-word KL.
"""
import json
import sys

LAY = ["22", "23", "24", "25", "26"]
GN = {"numbers": "number responses (all positions)", "numbers_dig": "number responses (digit-token targets)",
      "prose": "prose responses (trivia, open)", "wikitext": "WikiText test (raw)"}
print("| lens | positions | n | " + " | ".join(f"L{l} agree / KL" for l in LAY)
      + " | L22-26 mean: agree / KL | whole-word agree / KL (L22-26 mean) |")
print("|---" * (len(LAY) + 5) + "|")
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    d = json.load(open(path))["groups"]
    for g, gname in GN.items():
        r = d[g]
        cells = [f"{r[l]['agree']:.2f} / {r[l]['kl']:.1f}" for l in LAY]
        m = lambda k: sum(r[l][k] for l in LAY) / len(LAY)
        ww = f"{m('agree_ww'):.2f} / {m('kl_ww'):.1f}" if "kl_ww" in r[LAY[0]] else f"{m('agree_ww'):.2f} / -"
        print(f"| {name} | {gname} | {r[LAY[0]]['n']} | " + " | ".join(cells)
              + f" | {m('agree'):.2f} / {m('kl'):.1f} | {ww} |")
