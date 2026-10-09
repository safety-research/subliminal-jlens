"""Loading and aggregating the completion-level score files written by scripts/score_lens.py.

Rates follow lenssel/hf_rates.py and jpp/rates.py of the original runs: one trial = one completion (the plain mean over
completions; the large pools have one or four completions per prompt), hit at top-k if the primary set's best rank over band
layers x positions is <= k; chance = the mean over completions of the fraction of the 100 random sets whose best rank is
<= k. "nop0" = position 0 (the last prompt token, identical for every completion of a prompt) excluded.
"""
import glob
import json
import statistics

import torch

KS = (1, 5, 20)


def expand(paths):
    """Score-file arguments -> file list. Globs are expanded and sorted (lexicographically, as `ls` did in the original
    runs); explicit file names keep their order. Record order matters only for the random draws of the arm builders."""
    out = []
    for p in ([paths] if isinstance(paths, str) else paths):
        out += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    return out


def load(paths, keep_mat=False, pos0=None, sib=False):
    """Records of every file, in file order. pos0: optional set that collects the ids of prompts whose position 0 reads the
    primary set at top-1 (needs the rank matrix). sib: ids of sibling generations look like 'sib:<prompt id>'; the prompt id
    is stored in r['pid'] (r['pid'] = r['id'] otherwise)."""
    recs = []
    for p in expand(paths):
        for r in torch.load(p, weights_only=False)["recs"]:
            r["pid"] = r["id"].split(":", 1)[1] if sib else r["id"]
            if pos0 is not None and r["n_pos"] and int(r["mat"][0].min()) <= 1:
                pos0.add(r["pid"])
            if not keep_mat:
                r.pop("mat", None)
            recs.append(r)
    return recs


def texts(gen_paths):
    """{'<id>|onpolicy|<sample>': completion text} for scripts/generate.py outputs."""
    out = {}
    for fn in gen_paths:
        with open(fn) as f:
            for line in f:
                g = json.loads(line)
                for s, o in enumerate(g["outputs"]):
                    out[f"{g['id']}|onpolicy|{s}"] = o["text"]
    return out


def metas(prompt_paths):
    """{prompt id: meta} from prompt files ({"id", "messages", "meta": {"prompt", template features...}})."""
    out = {}
    for fn in prompt_paths:
        with open(fn) as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = r["meta"]
    return out


def write_chat_arm(path, recs, meta, text):
    """Student rows (no system prompt): the user prompt and the teacher's completion."""
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps({"messages": [{"role": "user", "content": meta[r["pid"]]["prompt"]},
                                             {"role": "assistant", "content": text[r["key"]]}]}) + "\n")


def rates(recs, field="best", rand="rand_best"):
    """Completion-level top-k rates, chance, the same without position 0, and the median best rank (percent)."""
    n = len(recs)
    o = {"n": n, "median_best_rank": statistics.median(r[field] for r in recs) if n else None}
    nop0 = field.replace("best", "best_nop0") if field != "best" else "best_nop0"
    for k in KS:
        o[f"top{k}"] = 100 * sum(r[field] <= k for r in recs) / n
        o[f"chance{k}"] = 100 * sum(float((r[rand] <= k).float().mean()) for r in recs) / n
        o[f"top{k}_nop0"] = 100 * sum(r[nop0] <= k for r in recs) / n
        o[f"chance{k}_nop0"] = 100 * sum(float((r[rand + "_nop0"] <= k).float().mean()) for r in recs
                                         if r.get(rand + "_nop0") is not None) / n
    return o


def fmt(o, nop0=False):
    s = "_nop0" if nop0 else ""
    return (f"n={o['n']}: top-1 / 5 / 20 = {o['top1' + s]:.2f} / {o['top5' + s]:.2f} / {o['top20' + s]:.2f}%  "
            f"(chance {o['chance1' + s]:.2f} / {o['chance5' + s]:.2f} / {o['chance20' + s]:.2f}%)"
            + ("" if nop0 else f", median best rank {o['median_best_rank']:g}"))
