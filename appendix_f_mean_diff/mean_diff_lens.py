"""Mean-activation diff through the J-lens (Qwen2.5-7B-Instruct, WikiText-fit lens "qwen25_7b").

Question (colleague's suggestion): average the residual stream over many on-policy completions under the owl system
prompt and under no system prompt, read the *difference* through the lens, and see whether owl shows up.

collect (GPU): for every completion, teacher-force it under a prompt and accumulate, per (group, prompt item, sample
parity) and per lens layer l, sums over response positions of
    h        raw residual (block-l output)                 -> "lens of the mean": unembed(J_l mean h)
    u        J_l h / rms(J_l h)                            -> "mean of the lens": W (g * mean u) is exactly the mean of
                                                              per-position lens logits, since final RMSNorm has no bias
                                                              and lm_head is linear (checked numerically below)
plus the same sums over completions of per-completion means (mean-of-means), and over the first 8 response positions.
Groups: (task, cond, "on")         completions of cond under their own prompt (on-policy)
        (task, cond2, "off")       the no-prompt completions teacher-forced under cond2's prompt (diagnostic)
The sample "parity" (half) is crc32(gen dir | sample index) % 2, so it is fixed per sample index across items; samples
are iid within a request, so this is still a random split for the n=20 gen (for the n=3 rerun gen all three indices can
hash to the same half, and the within-item nulls are then skipped).
Number-task completions containing a 3+-letter word are stored under mode "on_w"/"off_w" and excluded from the
analysis (0.1-0.9% per condition; under the owl prompt some of them name owls).
Response positions are those predicting each response token (last prompt token .. second-to-last response token),
the convention of onpolicy_masked.py.

analyze (GPU): see analyze() docstring.
Lens: the qwen25_7b registry lens (override with SLJ_LENS=file); model weights from the HF id (override SLJ_MODEL_PATH).
usage: collect:  python appendix_f_mean_diff/mean_diff_lens.py collect --gen G1.jsonl:P1.jsonl [G2:P2 ...] --out ACC_i.pt --shard i --nshards n
       analyze:  python appendix_f_mean_diff/mean_diff_lens.py analyze --acc ACC_*.pt --out results/mean_diff_lens.json
"""
import argparse
import sys
import json
import os
import random
import re
import zlib
from collections import defaultdict

import torch

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["collect", "analyze"])
ap.add_argument("--gen", nargs="*")
ap.add_argument("--out")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--acc", nargs="*")
ap.add_argument("--nnull", type=int, default=200)
ap.add_argument("--nboot", type=int, default=200)
a = ap.parse_args()

TASKS = ("numbers", "open")
ON_CONDS = {"numbers": ("none", "sys_owl", "sys_horse", "user_owl", "sys_eagle", "sys_wolf"),
            "open": ("none", "sys_owl", "sys_horse", "user_owl")}
OFF_CONDS = ("sys_owl", "sys_horse", "user_owl")
K_FIRST = 8
BAND = [22, 23, 24, 25, 26]
KINDS = ("h", "hm", "u", "um", "h8", "u8", "h1", "u1", "h18", "u18")
NPP = 16   # per-position sums kept for response positions 0..NPP-1 (per group and parity, not per item)


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import models  # noqa: E402
from sljlens.readout import residuals  # noqa: E402


class _LC:   # the subset of the original lens_check module this script used
    @staticmethod
    def load_model(key):
        return models.load_model(key, model_path=os.environ.get("SLJ_MODEL_PATH"))

    @staticmethod
    def load_lens(key):
        return models.load_lens(os.environ.get("SLJ_LENS", key))

    residuals = staticmethod(residuals)


def load_lc():
    return _LC


def collect():
    torch.set_num_threads(4)
    lc = load_lc()
    hf, tok, model = lc.load_model("qwen25_7b")
    lens = lc.load_lens("qwen25_7b")
    LAYERS = sorted(lens.jacobians)
    J = {l: lens.jacobians[l].float() for l in LAYERS}
    eps = float(getattr(model._final_norm, "variance_epsilon", getattr(model._final_norm, "eps", 1e-6)))
    print("layers", LAYERS, "eps", eps, "J dtype", lens.jacobians[LAYERS[0]].dtype, flush=True)
    d = J[LAYERS[0]].shape[0]
    nL = len(LAYERS)

    # prompts: id -> messages, from all prompt files
    prompts, comps = {}, []
    for spec in a.gen:
        gpath, ppath = spec.split(":")
        for l in open(ppath):
            r = json.loads(l)
            prompts[r["id"]] = r["messages"]
        for l in open(gpath):
            g = json.loads(l)
            m = g["meta"]
            if m["task"] not in TASKS or m["cond"] not in ON_CONDS[m["task"]]:
                continue
            if zlib.crc32(f"{m['task']}|{m['item']}".encode()) % a.nshards != a.shard:
                continue
            for s, o in enumerate(g["outputs"]):
                comps.append((m["task"], m["cond"], m["item"], g["id"], f"{os.path.basename(os.path.dirname(gpath))}|{s}", o["text"]))
    print("completions in shard", len(comps), flush=True)

    acc = {}   # (task, cond, mode, item, parity) -> dict kind -> [nL, d] fp32 (cpu), plus counts

    def slot(key):
        if key not in acc:
            acc[key] = {k: torch.zeros(nL, d, device="cuda") for k in KINDS}   # accumulate on GPU
            acc[key].update(npos=0, ncomp=0, npos8=0, npos1=0, npos18=0)
        return acc[key]

    pp = {}    # (task, cond, mode, parity) -> per-position sums

    def pslot(key):
        if key not in pp:
            pp[key] = dict(h=torch.zeros(NPP, nL, d, device="cuda"), u=torch.zeros(NPP, nL, d, device="cuda"),
                           n=torch.zeros(NPP))
        return pp[key]

    pref_cache = {}

    def prefix(pid):
        if pid not in pref_cache:
            pref_cache[pid] = tok(tok.apply_chat_template(prompts[pid], tokenize=False, add_generation_prompt=True),
                                  add_special_tokens=False)["input_ids"]
        return pref_cache[pid]

    checked = False
    for ci, (task, cond, item, gid, skey, text) in enumerate(comps):
        r = tok(text, add_special_tokens=False)["input_ids"]
        if not r:
            continue
        parity = zlib.crc32(skey.encode()) % 2
        # number completions that contain words (refusals, the odd animal mention) go to separate "_w" groups, so
        # the main analysis is on pure-number completions only
        wsuf = "_w" if task == "numbers" and re.search(r"[A-Za-z]{3,}", text) else ""
        runs = [(cond, "on" + wsuf, gid)]
        if cond == "none":
            runs += [(c2, "off" + wsuf, f"{task}|{item}|{c2}") for c2 in OFF_CONDS]
        for c, mode, pid in runs:
            p = prefix(pid)
            acts = lc.residuals(model, p + r, LAYERS)
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(r), device="cuda")
            H = torch.stack([acts[l][pos] for l in LAYERS])                     # [nL, P, d]
            T = torch.stack([H[i] @ J[l].T for i, l in enumerate(LAYERS)])      # transported
            U = T * torch.rsqrt(T.pow(2).mean(-1, keepdim=True) + eps)
            if not checked:   # mean-of-lens identity check on one completion, band layer 24
                i = LAYERS.index(24)
                ref = model.unembed(T[i]).float().mean(0)
                W = model._lm_head.weight.float()
                g = model._final_norm.weight.float()
                mine = W @ (g * U[i].mean(0))
                print("CHECK mean-of-lens identity: max|diff|", float((ref - mine).abs().max()),
                      "max|ref|", float(ref.abs().max()), flush=True)
                del W
                checked = True
            P = H.shape[1]
            k8 = min(K_FIRST, P)
            sl = slot((task, c, mode, item, parity))
            sl["h"] += H.sum(1)
            sl["hm"] += H.mean(1)
            sl["u"] += U.sum(1)
            sl["um"] += U.mean(1)
            sl["h8"] += H[:, :k8].sum(1)
            sl["u8"] += U[:, :k8].sum(1)
            sl["npos"] += P
            sl["ncomp"] += 1
            sl["npos8"] += k8
            if P > 1:   # positions 1.. exclude the prompt's last token (the "\n" after "assistant")
                sl["h1"] += H[:, 1:].sum(1)
                sl["u1"] += U[:, 1:].sum(1)
                sl["h18"] += H[:, 1:K_FIRST + 1].sum(1)
                sl["u18"] += U[:, 1:K_FIRST + 1].sum(1)
                sl["npos1"] += P - 1
                sl["npos18"] += min(K_FIRST, P - 1)
            ps = pslot((task, c, mode, parity))
            kp = min(NPP, P)
            ps["h"][:kp] += H[:, :kp].transpose(0, 1)
            ps["u"][:kp] += U[:, :kp].transpose(0, 1)
            ps["n"][:kp] += 1
        if ci % 500 == 0:
            print("progress", ci, len(comps), flush=True)
    acc = {k: {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} for k, v in acc.items()}
    pp = {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in pp.items()}
    torch.save(dict(acc=acc, pp=pp, layers=LAYERS), a.out)
    print("DONE", len(comps), flush=True)


# ------------------------------------------------------------------------------------------------ analyze
def analyze():
    """Statistics. For a set of (task, cond, mode) groups and a list of item indices (with multiplicity, for
    bootstraps) we form per-layer means:
        pooled: sum over positions / #positions;  mom: sum of per-completion means / #completions
    Variants (diff = A - B, per layer l, vocab-size logit vector):
        lensmean      unembed(J mA_h) - unembed(J mB_h)        (full final norm, pooled means)
        lensmean_mom  same with mean-of-per-completion-means
        lin_gain      W (g * J (mA_h - mB_h))                   (no normalisation, keep norm gain)
        lin_nogain    W J (mA_h - mB_h)
        meanlens      W (g * (mA_u - mB_u)) = mean per-position lens logits A - B  (pooled)
        meanlens_mom  same, mean of per-completion means
        lensmean8 / meanlens8   first 8 response positions only
    Ranks: best rank of a token set (higher diff = better) among whole-word tokens (^ [A-Za-z]{3,}$) and in the full
    vocab. Reported per layer, as min over band layers, and for the band-averaged diff (diff averaged over L22-26)
    and the all-layer-averaged diff.
    Null (a): none-vs-none, prompt items randomly split into two halves of 50 (25 for open): owl rank distribution.
    Null (a'): within-item sign-flip: none items' two sample halves (random parity), diff = sum_i s_i (half0_i - half1_i)
               with random signs; for linear variants only (lin_gain, meanlens) since it has no A/B means.
    Bootstrap (b): items resampled with replacement (A and B on the same resampled items).
    (c): fraction of 100 random size-2 whole-word sets whose best rank beats the animal set's.
    """
    lc = load_lc()
    hf, tok, model = lc.load_model("qwen25_7b")
    lens = lc.load_lens("qwen25_7b")
    W = model._lm_head.weight.float()                       # [V, d]
    V = W.shape[0]
    g = model._final_norm.weight.float()
    eps = float(getattr(model._final_norm, "variance_epsilon", 1e-6))
    acc, PP, LAYERS = {}, {}, None
    for p in a.acc:
        dd = torch.load(p, weights_only=False)
        LAYERS = dd["layers"]
        for k, v in dd.get("pp", {}).items():
            if k not in PP:
                PP[k] = v
            else:
                for kk in v:
                    PP[k][kk] = PP[k][kk] + v[kk]
        for k, v in dd["acc"].items():      # files from different gen sources add up
            if k not in acc:
                acc[k] = v
            else:
                for kk in v:
                    acc[k][kk] = acc[k][kk] + v[kk]
    Jst = torch.stack([lens.jacobians[l].float() for l in LAYERS])   # [nL, d, d]
    nL = len(LAYERS)
    bidx = [LAYERS.index(l) for l in BAND]
    WW = re.compile(r"^ [A-Za-z]{3,}$")
    ww = torch.zeros(V, dtype=torch.bool)
    for i in range(min(V, len(tok))):
        if WW.match(tok.decode([i])):
            ww[i] = True
    ww = ww.cuda()
    nww = int(ww.sum())

    def ids(strings):
        out = []
        for s in strings:
            t = tok.encode(s, add_special_tokens=False)
            if len(t) == 1 and bool(ww[t[0]]):
                out.append(t[0])
        return sorted(set(out))

    SETS = {"owl": ids([" owl", " Owl", " owls", " Owls"]), "horse": ids([" horse", " Horse", " horses", " Horses"]),
            "eagle": ids([" eagle", " Eagle", " eagles", " Eagles"]), "wolf": ids([" wolf", " Wolf", " wolves", " Wolves"])}
    print({k: [tok.decode([i]) for i in v] for k, v in SETS.items()}, "n whole-word", nww, flush=True)
    rng = random.Random(0)
    wwl = ww.nonzero().squeeze(-1).tolist()
    RAND = torch.tensor([rng.sample(wwl, 2) for _ in range(100)]).cuda()

    def items_of(task):
        return sorted({k[3] for k in acc if k[0] == task})

    def means(task, cond, mode, items, parities=(0, 1)):
        """Per-layer mean vectors for each kind, over the given items (list, multiplicity allowed)."""
        tot = {k: torch.zeros(nL, Jst.shape[1]) for k in KINDS}
        npos = ncomp = npos8 = npos1 = npos18 = 0
        for it in items:
            for par in parities:
                s = acc.get((task, cond, mode, it, par))
                if s is None:
                    continue
                for k in KINDS:
                    tot[k] += s[k]
                npos += s["npos"]
                ncomp += s["ncomp"]
                npos8 += s["npos8"]
                npos1 += s["npos1"]
                npos18 += s["npos18"]
        return dict(h=tot["h"] / npos, hm=tot["hm"] / ncomp, u=tot["u"] / npos, um=tot["um"] / ncomp,
                    h8=tot["h8"] / npos8, u8=tot["u8"] / npos8, h1=tot["h1"] / npos1, u1=tot["u1"] / npos1,
                    h18=tot["h18"] / npos18, u18=tot["u18"] / npos18, ncomp=ncomp)

    def transport(m):          # [nL, d] -> [nL, d]
        return torch.einsum("lij,lj->li", Jst, m.cuda())

    def lens_logits(m):        # full unembed of transported mean: W g norm(J m)
        t = transport(m)
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
        return (g * t) @ W.T   # [nL, V]

    def variants(A, B):
        out = {}
        out["lensmean"] = lens_logits(A["h"]) - lens_logits(B["h"])
        out["lensmean_mom"] = lens_logits(A["hm"]) - lens_logits(B["hm"])
        dt = transport(A["h"] - B["h"])
        out["lin_gain"] = (g * dt) @ W.T
        out["lin_nogain"] = dt @ W.T
        out["meanlens"] = (g * (A["u"] - B["u"]).cuda()) @ W.T
        out["meanlens_mom"] = (g * (A["um"] - B["um"]).cuda()) @ W.T
        out["lensmean8"] = lens_logits(A["h8"]) - lens_logits(B["h8"])
        out["meanlens8"] = (g * (A["u8"] - B["u8"]).cuda()) @ W.T
        out["lensmean_x0"] = lens_logits(A["h1"]) - lens_logits(B["h1"])
        out["meanlens_x0"] = (g * (A["u1"] - B["u1"]).cuda()) @ W.T
        out["lensmean1to8"] = lens_logits(A["h18"]) - lens_logits(B["h18"])
        out["meanlens1to8"] = (g * (A["u18"] - B["u18"]).cuda()) @ W.T
        return out

    def ranks_of(z, idx):
        """z [..., V]; best rank of token set idx among whole-word tokens and full vocab."""
        best = z[..., idx].amax(-1, keepdim=True)
        rww = ((z > best) & ww).sum(-1) + 1
        rfull = (z > best).sum(-1) + 1
        return rww, rfull

    def summarize(z, idx):
        """z [nL, V] -> dict of ranks: per layer, band-min, band-avg diff, all-avg diff (ww and full)."""
        rww, rfull = ranks_of(z, idx)
        zb, za = z[bidx].mean(0), z.mean(0)
        bww, bfull = ranks_of(zb, idx)
        aww, afull = ranks_of(za, idx)
        return dict(per_layer_ww=rww.tolist(), per_layer_full=rfull.tolist(),
                    band_min_ww=int(rww[bidx].min()), all_min_ww=int(rww.min()),
                    band_avg_ww=int(bww), band_avg_full=int(bfull), all_avg_ww=int(aww), all_avg_full=int(afull))

    def rand_beat(z, idx):
        """fraction of random 2-sets with best rank strictly better than idx's (band-avg and all-avg diffs)."""
        res = {}
        for name, zz in (("band_avg", z[bidx].mean(0)), ("all_avg", z.mean(0))):
            r_set, _ = ranks_of(zz, idx)
            r_rand = ((zz[None] > zz[RAND].amax(-1, keepdim=True)) & ww).sum(-1) + 1
            res[name] = float((r_rand < r_set).float().mean())
        return res

    def topk(z, k=20):
        zz = z.masked_fill(~ww, float("-inf"))
        v, i = zz.topk(k)
        return [(tok.decode([t]), round(float(x), 3)) for t, x in zip(i.tolist(), v.tolist())]

    COMPS = []
    for task in TASKS:
        COMPS += [(task, f"{c}-none", (task, c, "on"), (task, "none", "on")) for c in ON_CONDS[task] if c != "none"]
        COMPS += [(task, f"OFF {c}-none", (task, c, "off"), (task, "none", "on")) for c in OFF_CONDS]
        # animal-vs-animal contrasts: removes the generic "there is a system prompt" component
        COMPS += [(task, f"{c}-sys_horse", (task, c, "on"), (task, "sys_horse", "on")) for c in ON_CONDS[task]
                  if c.startswith("sys_") and c != "sys_horse"]
        COMPS += [(task, "sys_horse-sys_owl", (task, "sys_horse", "on"), (task, "sys_owl", "on"))]
    TARGETS = {"sys_owl": ["owl", "horse"], "user_owl": ["owl", "horse"], "sys_horse": ["horse", "owl"],
               "sys_eagle": ["eagle", "owl"], "sys_wolf": ["wolf", "owl"]}
    VARS = ["lensmean", "lensmean_mom", "lin_gain", "lin_nogain", "meanlens", "meanlens_mom", "lensmean8", "meanlens8",
            "lensmean_x0", "meanlens_x0", "lensmean1to8", "meanlens1to8"]
    results = dict(layers=LAYERS, band=BAND, n_wholeword=nww, sets={k: [tok.decode([i]) for i in v] for k, v in SETS.items()},
                   comps={}, nulls={})
    gen = torch.Generator().manual_seed(0)

    # nulls per task
    for task in TASKS:
        its = items_of(task)
        null = {v: defaultdict(list) for v in VARS}
        for rep in range(a.nnull):
            perm = torch.randperm(len(its), generator=gen).tolist()
            hA = [its[i] for i in perm[: len(its) // 2]]
            hB = [its[i] for i in perm[len(its) // 2:]]
            zs = variants(means(task, "none", "on", hA), means(task, "none", "on", hB))
            for v in VARS:
                for an in SETS:
                    s = summarize(zs[v], SETS[an])
                    null[v][an].append({k: s[k] for k in ("band_min_ww", "band_avg_ww", "all_avg_ww", "all_min_ww")}
                                       | {"band_layers_ww": [s["per_layer_ww"][i] for i in bidx]})
        # within-item sign-flip null (linear variants)
        has_both = all((task, "none", "on", it, p) in acc for it in its for p in (0, 1))
        if not has_both:   # (rerun gen, n=3: parity is keyed by sample index, so all samples can land in one half)
            results["nulls"][task] = dict(split=null, flip={})
            print("null done (no sign-flip null: some items lack a second half)", task, flush=True)
            continue
        h0 = {it: means(task, "none", "on", [it], (0,)) for it in its}
        h1 = {it: means(task, "none", "on", [it], (1,)) for it in its}
        flip = {"lin_gain": defaultdict(list), "meanlens": defaultdict(list)}
        for rep in range(a.nnull):
            sgn = (torch.randint(0, 2, (len(its),), generator=gen) * 2 - 1).float()
            dh = sum(sgn[j] * (h0[it]["h"] - h1[it]["h"]) for j, it in enumerate(its))
            du = sum(sgn[j] * (h0[it]["u"] - h1[it]["u"]) for j, it in enumerate(its))
            zs = {"lin_gain": (g * transport(dh)) @ W.T, "meanlens": (g * du.cuda()) @ W.T}
            for v in zs:
                for an in SETS:
                    s = summarize(zs[v], SETS[an])
                    flip[v][an].append({k: s[k] for k in ("band_min_ww", "band_avg_ww", "all_avg_ww", "all_min_ww")})
        results["nulls"][task] = dict(split=null, flip=flip)
        print("null done", task, flush=True)

    for task, name, GA, GB in COMPS:
        its = items_of(task)
        A, B = means(*GA, its), means(*GB, its)
        if A["ncomp"] == 0:
            continue
        zs = variants(A, B)
        cond = GA[1]
        r = dict(nA=A["ncomp"], nB=B["ncomp"], var={})
        for v in VARS:
            rv = {}
            for an in TARGETS[cond]:
                rv[an] = summarize(zs[v], SETS[an])
                rv[an]["rand_beat"] = rand_beat(zs[v], SETS[an])
            rv["top20"] = {str(LAYERS[i]): topk(zs[v][i]) for i in bidx}
            rv["top20"]["band_avg"] = topk(zs[v][bidx].mean(0))
            rv["top20"]["all_avg"] = topk(zs[v].mean(0))
            r["var"][v] = rv
        # bootstrap over items for main variants
        boot = {v: defaultdict(list) for v in ("lensmean", "lin_gain", "meanlens")}
        for rep in range(a.nboot):
            bi = [its[i] for i in torch.randint(0, len(its), (len(its),), generator=gen).tolist()]
            zb = variants(means(*GA, bi), means(*GB, bi))
            for v in boot:
                for an in TARGETS[cond]:
                    s = summarize(zb[v], SETS[an])
                    boot[v][an].append({k: s[k] for k in ("band_min_ww", "band_avg_ww", "all_avg_ww")})
        r["boot"] = boot
        results["comps"][f"{task}: {name}"] = r
        print("comp done", task, name, {v: {an: r['var'][v][an]['band_avg_ww'] for an in TARGETS[cond]} for v in ("lensmean", "meanlens")}, flush=True)
    # per-position (index aligned across completions; position 0 = the prompt's last token). Null: the two random
    # sample halves of the no-prompt completions (one realisation per position).
    def pmean(task, cond, mode, j, pars=(0, 1)):
        hs = sum(PP[(task, cond, mode, p)]["h"][j] for p in pars if (task, cond, mode, p) in PP)
        us = sum(PP[(task, cond, mode, p)]["u"][j] for p in pars if (task, cond, mode, p) in PP)
        n = sum(float(PP[(task, cond, mode, p)]["n"][j]) for p in pars if (task, cond, mode, p) in PP)
        return hs / n, us / n
    perpos = {}
    for task in TASKS:
        for name, ga, gb, an in ([("sys_owl-none", (task, "sys_owl", "on"), (task, "none", "on"), "owl"),
                                  ("sys_horse-none [owl]", (task, "sys_horse", "on"), (task, "none", "on"), "owl"),
                                  ("sys_horse-none [horse]", (task, "sys_horse", "on"), (task, "none", "on"), "horse"),
                                  ("user_owl-none", (task, "user_owl", "on"), (task, "none", "on"), "owl"),
                                  ("OFF sys_owl-none", (task, "sys_owl", "off"), (task, "none", "on"), "owl")]):
            rows = []
            for j in range(NPP):
                (hA, uA), (hB, uB) = pmean(*ga, j), pmean(*gb, j)
                zl = lens_logits(hA) - lens_logits(hB)
                zm = (g * (uA - uB).cuda()) @ W.T
                rows.append(dict(lensmean=summarize(zl, SETS[an]), meanlens=summarize(zm, SETS[an]),
                                 top_band=topk(zm[bidx].mean(0), 10)))
            perpos[f"{task}: {name}"] = rows
        rows = []
        if not all((task, "none", "on", p) in PP for p in (0, 1)):
            continue
        for j in range(NPP):
            (hA, uA), (hB, uB) = pmean(task, "none", "on", j, (0,)), pmean(task, "none", "on", j, (1,))
            zl = lens_logits(hA) - lens_logits(hB)
            zm = (g * (uA - uB).cuda()) @ W.T
            rows.append(dict(lensmean=summarize(zl, SETS["owl"]), meanlens=summarize(zm, SETS["owl"])))
        perpos[f"{task}: NULL none half0-half1 [owl]"] = rows
    results["perpos"] = perpos
    results["pos0_token"] = "last prompt token"
    json.dump(results, open(a.out, "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    collect() if a.mode == "collect" else analyze()
