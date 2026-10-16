#!/usr/bin/env python3
"""Markdown tables for the screen from analyze.py --json outputs (lens_screen/table.py of the original runs). Row labels are
the LABEL= names that run.sh passes to analyze.py; rows absent from a file are skipped.
usage: python planned_lens_strength_screen/table.py summary_cat.json summary_math.json summary_mathev.json"""
import json
import sys

ROWS = [  # (label in json, display model, lens description)
    ("qwen25_7b_ref", "Qwen2.5-7B-Instruct (ref)", "own fit, 300 WikiText"),
    ("gemma4_own_ref", "Gemma-4-31B-it (ref)", "own reference-recipe fit, all layers"),
    ("gemma4_lamm_ref", "Gemma-4-31B-it (ref)", "lamm-mit paper lens (25 layers)"),
    ("llama33_70b__np", "Llama-3.3-70B-Instruct", "Neuronpedia J-lens (125 prompts)"),
    ("llama33_70b__logit", "Llama-3.3-70B-Instruct", "control: logit lens (no J)"),
    ("olmo31_32b__npbase", "OLMo-3.1-32B-Instruct", "Neuronpedia J-lens fit on OLMo-3-1125-32B base"),
    ("olmo31_32b__jppbase", "OLMo-3.1-32B-Instruct", "J++ lens (Ayonrinde & Lindsey) fit on base"),
    ("olmo31_32b__logit", "OLMo-3.1-32B-Instruct", "control: logit lens (no J)"),
    ("gemma3_27b__np", "Gemma-3-27B-it", "Neuronpedia J-lens (828 prompts)"),
    ("gemma3_27b__cbj", "Gemma-3-27B-it", "camilablank J-lens (25 pile prompts)"),
    ("gemma3_27b__cbr", "Gemma-3-27B-it", "camilablank R-lens (RelP, 25 pile prompts)"),
    ("gemma3_27b__logit", "Gemma-3-27B-it", "control: logit lens (no J)"),
    ("dsv4__np", "DeepSeek-V4-Flash", "Neuronpedia/mntss J-lens (25 pile prompts), stream mean"),
    ("dsv4__cbr", "DeepSeek-V4-Flash", "camilablank R-lens (all-c4), stream mean"),
    ("dsv4__logit", "DeepSeek-V4-Flash", "control: logit lens (no J), stream mean"),
    ("qwen36_27b__np", "Qwen3.6-27B", "Neuronpedia J-lens (n1000)"),
    ("qwen36_27b__jpp", "Qwen3.6-27B", "J++ lens (Ayonrinde & Lindsey)"),
    ("qwen36_27b__logit", "Qwen3.6-27B", "control: logit lens (no J)"),
    ("qwen35_27b__np", "Qwen3.5-27B", "Neuronpedia J-lens (672 prompts)"),
    ("qwen3_32b_ref", "Qwen3-32B (earlier run)", "Neuronpedia J-lens (557 prompts)"),
    ("gptoss_20b__np", "gpt-oss-20b", "Neuronpedia J-lens (457 prompts)"),
]


def pct(x):
    return f"{100 * x:.0f}%"


for path in sys.argv[1:]:
    D = json.load(open(path))
    print(f"\n### {path}\n")
    print("| model | lens | top-1 band | **focus top-1, any layer in band** | mention | suppress | none | output logits focus / none (top-1) "
          "| best single layer: focus / none | top-5 band hit focus / none | peak sep top-1 (layer) | output sep top-1 | peak minus output, 95% CI |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, model, lens in ROWS:
        if key not in D:
            continue
        r = D[key]
        k1, k5 = r["k1"], r["k5"]
        h1, h5 = k1["band_hits"]["top1"], k1["band_hits"]["top5"]
        o = r["output_hits"]["top1"]
        b = r["best_single_layer_top1"]
        ci = k1["peak_minus_output_ci"]
        print(f"| {model} | {lens} | L{k1['band'][0]}-{k1['band'][1]} of {r['n_layers']} | **{pct(h1['focus'])}** | {pct(h1['control'])} | "
              f"{pct(h1['suppress'])} | {pct(h1['none'])} | {pct(o['focus'])} / {pct(o['none'])} | L{b['layer']}: {pct(b['focus'])} / {pct(b['none'])} | "
              f"{pct(h5['focus'])} / {pct(h5['none'])} | {k1['peak']:.2f} (L{k1['peak_layer']}) | {k1['output']:.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
