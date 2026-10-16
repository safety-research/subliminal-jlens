#!/usr/bin/env python3
"""Lens-strength screen: the J-lens paper's directed-modulation toy task for any HF chat model and one or more public lens
files, read in one forward pass per trial (lens_screen/screen.py of the original runs).

Trials, chat prefix, rank statistic: as scripts/derive_band.py (system prompt "You are a helpful assistant."; the user asks
to copy a carrier sentence, plainly or with an instruction about X; the sentence is teacher-forced; at each copied position
and layer the best rank of X's single-token members among word-like tokens `^ ?[A-Za-z]{3,}$`, ties counted in the token's
favour). Differences from derive_band.py:
  * logits in fp32 (final norm in the model dtype, then an fp32 LM head and softcap), which matters for large vocabularies
    with bf16 ties; the Qwen2.5-7B reference re-run through this script gives 44% focus top-1 against derive_band.py's 45%;
  * several lenses share one forward; the lens file formats of the public releases are read directly (load_lens);
  * per-model chat rendering (--chat): 'default' = the tokenizer's template; 'qwen' = Qwen3.x with enable_thinking=False;
    'gptoss' = harmony with the final channel opened directly (no analysis channel runs); 'dsv4' = DeepSeek-V4's own
    encoder in chat (non-thinking) mode, prefix ending in '<｜Assistant｜></think>';
  * DeepSeek-V4 (mHC: each block outputs 4 residual streams) is read at the MEAN of the streams (the Neuronpedia
    convention), and its output layer is the HF logits (which apply the hc_head collapse) rather than our unembedding;
  * --family math: the paper's math_problems (24 expressions) with target = the answer digit and its number word, rank
    universe = word-like plus pure-digit tokens; --math-x picks how the {x} slot is filled ('expr': "Think about 3^2 - 2
    while you write."; 'evaluating': "Think about evaluating 3^2 - 2 while you write.", the paper's wording in its example).

Output per lens: OUTDIR/band_<key>__<lens>[__math|__mathev].pkl
  {"key", "lens", "hf", "layers": [lens source layers..., L-1], "n_layers": L,
   "dm": [{group, phrasing, cat, carrier, rank [n_layers_read, P], logp [n_layers_read, P]}], "sanity": {...}}
The last entry of "layers" is always the output layer L-1, read without a lens (the model's own logits). This is the format
of scripts/derive_band.py --trials-out (plus logp and sanity), so analyze.py and 04's dm_strength.py read both.
Sanity checks stored: block L-1 output through our unembedding vs the HF model's logits (first 3 trials), and per-layer
top-1 agreement of each lens with the model's own next-token prediction (first 40 trials, positions >= 4).

usage: python planned_lens_strength_screen/screen.py --key llama33_70b --hf meta-llama/Llama-3.3-70B-Instruct \
           --chat default --out OUTDIR --lens np=PATH.pt --lens logit=identity [--family math --math-x evaluating]
       (--lens NAME=identity is the logit-lens control: no Jacobian at any layer)
"""
import argparse
import importlib.util
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens import vocab  # noqa: E402
from sljlens.io import DATA  # noqa: E402

SYSTEM = "You are a helpful assistant."
NUMWORD = {"2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def load_module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def cat_trials(DM, n_carriers=5):
    """The topic-category family: identical to scripts/derive_band.py (22 categories x 5 carriers x (24 phrasings + none))."""
    for cat in DM["topic_categories"]:
        for s in DM["carrier_sentences"][:n_carriers]:
            yield dict(group="none", phrasing="none", cat=cat["name"], members=cat["members"],
                       user=f'Write "{s}" Don\'t write anything else.', resp=s)
            for ph in DM["phrasings"]:
                x = ph["text"].format(x=cat["name"])
                x = x[0].upper() + x[1:]
                yield dict(group=DM["group_kind"][ph["group"]], phrasing=ph["name"], cat=cat["name"], members=cat["members"],
                           user=f'Write "{s}" {x} Don\'t write anything else.', resp=s)


def math_trials(DM, math_x, n_carriers=5):
    """The same protocol with x = the expression (or 'evaluating <expr>') and target = its answer (digit and number word)."""
    for prob in DM["math_problems"]:
        members = [prob["answer"], NUMWORD[prob["answer"]]]
        for s in DM["carrier_sentences"][:n_carriers]:
            yield dict(group="none", phrasing="none", cat=prob["expr"], members=members,
                       user=f'Write "{s}" Don\'t write anything else.', resp=s)
            for ph in DM["phrasings"]:
                x = ph["text"].format(x=prob["expr"] if math_x == "expr" else f"evaluating {prob['expr']}")
                x = x[0].upper() + x[1:]
                yield dict(group=DM["group_kind"][ph["group"]], phrasing=ph["name"], cat=prob["expr"], members=members,
                           user=f'Write "{s}" {x} Don\'t write anything else.', resp=s)


def load_lens(path, L, d_model, device):
    """{source layer: J [d, d] fp32 or None (identity)} from any of the public formats:
      'identity'                       logit-lens control (identity map at every layer 0..L-2)
      {"J": {layer: J}, ...}           the reference jlens format (Neuronpedia J-lens and J++ files, our own fits)
      {"J": tensor [n(+1), d, d], "source_layers": [...]}   stacked format (camilablank J-/R-lenses and the DeepSeek-V4
                                       lenses), whose last row is an identity anchor for the target layer; it is dropped."""
    if path == "identity":
        return {l: None for l in range(L - 1)}, "logit lens (identity)"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    J = ck["J"]
    if torch.is_tensor(J):
        src = [int(x) for x in ck["source_layers"]]
        if J.shape[0] == len(src) + 1:
            J = J[:len(src)]
        assert J.shape[0] == len(src), (J.shape, len(src))
        J = {l: J[i] for i, l in enumerate(src)}
    J = {int(l): m.to(device, torch.float32) for l, m in J.items()}
    assert max(J) <= L - 1 and next(iter(J.values())).shape[0] == d_model
    return J, ck.get("provenance")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="label used in the output file names")
    ap.add_argument("--hf", required=True, help="HF id or local checkpoint directory")
    ap.add_argument("--chat", default="default", choices=["default", "qwen", "gptoss", "dsv4"])
    ap.add_argument("--lens", action="append", required=True, help="NAME=PATH, or NAME=identity for the logit lens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="cat", choices=["cat", "math"])
    ap.add_argument("--math-x", default="expr", choices=["expr", "evaluating"])
    ap.add_argument("--dsv4-encoding", default=None,
                    help="path to DeepSeek-V4's encoding/encoding_dsv4.py (default: from the checkpoint directory, else "
                         "downloaded from the model's HF repo)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N trials")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-sanity", type=int, default=40)
    a = ap.parse_args()

    import jlens
    import transformers
    from jlens.hooks import ActivationRecorder

    t0 = time.time()
    cfg = transformers.AutoConfig.from_pretrained(a.hf)
    tok = transformers.AutoTokenizer.from_pretrained(a.hf)
    ngpu = torch.cuda.device_count()
    Auto = transformers.AutoModelForImageTextToText if hasattr(cfg, "vision_config") else transformers.AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16, device_map="auto" if ngpu > 1 else "cuda")
    if getattr(cfg, "model_type", "") == "gpt_oss":           # MXFP4 checkpoint, dequantized to bf16
        kw["quantization_config"] = transformers.Mxfp4Config(dequantize=True)
    DS = getattr(cfg, "model_type", "") == "deepseek_v4"
    hf = Auto.from_pretrained(a.hf, **kw).eval()
    model = jlens.from_hf(hf, tok, force_bos=False)
    L = model.n_layers
    W = model._lm_head.weight
    V = W.shape[0]
    W32 = W.detach().float()
    softcap = model._logit_softcap
    print(f"loaded {type(hf).__name__} L={L} d={model.d_model} V={V} softcap={softcap} gpus={ngpu} ({time.time() - t0:.0f}s)",
          flush=True)

    @torch.no_grad()
    def unembed32(h):
        x = model._final_norm(h.to(W.dtype).to(W.device)).float()
        lg = F.linear(x, W32)
        if softcap is not None:
            lg = softcap * torch.tanh(lg / softcap)
        return lg

    LENSES = {}
    for spec in a.lens:
        name, path = spec.split("=", 1)
        J, prov = load_lens(path, L, model.d_model, W.device)
        LENSES[name] = J
        src = sorted(J)
        print(f"lens {name}: {len(src)} source layers L{src[0]}..L{src[-1]}, provenance={str(prov)[:300]}", flush=True)

    enc_dsv4 = None
    if a.chat == "dsv4":
        ep = a.dsv4_encoding or os.path.join(a.hf, "encoding", "encoding_dsv4.py")
        if not os.path.exists(ep):
            from huggingface_hub import hf_hub_download
            ep = hf_hub_download(a.hf, "encoding/encoding_dsv4.py")
        enc_dsv4 = load_module("encoding_dsv4", ep)

    def prefix(user):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if a.chat == "qwen":
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if a.chat == "gptoss":
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="low")
            return p + "<|channel|>final<|message|>"
        if a.chat == "dsv4":
            return enc_dsv4.encode_messages(msgs, thinking_mode="chat")
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print("prefix sample:", repr(prefix("Hi")), flush=True)
    vmask = vocab.wordlike_mask(tok, V).to(W.device)
    if a.family == "math":
        dig = re.compile(r"^ ?[0-9]+$")
        dmask = torch.zeros(V, dtype=torch.bool)
        for i in range(min(V, len(tok))):
            if dig.match(tok.decode([i])):
                dmask[i] = True
        vmask = vmask | dmask.to(W.device)
    print(f"rank-universe tokens ({a.family}): {int(vmask.sum())} / {V}", flush=True)
    ALL = list(range(L))

    @torch.no_grad()
    def acts_of(ids):
        """{layer: [S, d] fp32} block outputs (stream mean for mHC), plus the model's own fp32 logits [S, V] for DeepSeek."""
        t = torch.as_tensor(ids, device=model.input_device)[None]
        with ActivationRecorder(model.layers, at=ALL) as rec:
            if DS:
                logits = hf(input_ids=t, use_cache=False).logits[0].float().to(W.device)[:, :V]
            else:
                model.forward(t)
                logits = None
            acts = {}
            for l in ALL:
                x = rec.activations[l][0].float()
                acts[l] = (x.mean(-2) if x.dim() == 3 else x).to(W.device)
            return acts, logits

    @torch.no_grad()
    def score(lg, cid):
        lpr = torch.log_softmax(lg, -1)
        masked = lg.masked_fill(~vmask, float("-inf"))
        best = masked[:, cid].max(-1).values
        return ((masked > best[:, None]).sum(-1) + 1).int().cpu(), torch.logsumexp(lpr[:, cid], -1).cpu()

    DM = json.load(open(os.path.join(DATA, "directed-modulation.json")))
    trials = list(math_trials(DM, a.math_x)) if a.family == "math" else list(cat_trials(DM))
    si, sn = map(int, a.shard.split("/"))
    trials = trials[si::sn]
    if a.limit:
        trials = trials[:a.limit]
    print(f"{len(trials)} trials (shard {a.shard})", flush=True)

    out = {n: [] for n in LENSES}
    san = {"hf_vs_unembed_maxabs": [], "hf_vs_unembed_top1": [], "agree": {n: {} for n in LENSES}}
    for ti, t in enumerate(trials):
        mids = vocab.single_token_ids(tok, t["members"])
        if not mids:
            continue
        pids = tok(prefix(t["user"]), add_special_tokens=False).input_ids
        ids = pids + tok(t["resp"], add_special_tokens=False).input_ids
        span = (len(pids), len(ids))
        acts, dslog = acts_of(ids)
        pos = torch.arange(span[0], span[1], device=W.device)
        cid = torch.as_tensor(mids, device=W.device)
        final = dslog[pos] if DS else unembed32(acts[L - 1][pos])
        r_out, l_out = score(final, cid)
        if ti < 3 and not DS:
            with torch.no_grad():
                ref = hf(input_ids=torch.as_tensor(ids, device=model.input_device)[None], use_cache=False).logits[0]
            ref = ref[span[0]:span[1]].float().to(W.device)[:, :V]
            san["hf_vs_unembed_maxabs"].append(float((ref - final).abs().max()))
            san["hf_vs_unembed_top1"].append(float((ref.argmax(-1) == final.argmax(-1)).float().mean()))
        if ti < a.n_sanity:
            allpos = torch.arange(4, len(ids), device=W.device)
            ftop = (dslog[allpos] if DS else unembed32(acts[L - 1][allpos])).argmax(-1)
        for n, J in LENSES.items():
            R, P = [], []
            for l in sorted(J):
                lg = unembed32(acts[l][pos] if J[l] is None else acts[l][pos] @ J[l].T)
                r, p = score(lg, cid)
                R.append(r)
                P.append(p)
                if ti < a.n_sanity:
                    z = acts[l][allpos] if J[l] is None else acts[l][allpos] @ J[l].T
                    san["agree"][n].setdefault(l, []).append((unembed32(z).argmax(-1) == ftop).float().mean().item())
            R.append(r_out)
            P.append(l_out)
            out[n].append(dict(group=t["group"], phrasing=t["phrasing"], cat=t["cat"], carrier=t["resp"],
                               rank=torch.stack(R).numpy(), logp=torch.stack(P).numpy()))
        if ti % 250 == 0:
            print(f"trial {ti}/{len(trials)} {time.time() - t0:.0f}s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    sanity = dict(hf_vs_unembed_maxabs=san["hf_vs_unembed_maxabs"], hf_vs_unembed_top1=san["hf_vs_unembed_top1"])
    print("SANITY hf-vs-unembed maxabs", san["hf_vs_unembed_maxabs"], "top1 agree", san["hf_vs_unembed_top1"], flush=True)
    fam = "" if a.family == "cat" else "__" + a.family + ("" if a.math_x == "expr" else "ev")
    sfx = "" if a.shard == "0/1" else f".shard{si}of{sn}"
    for n, J in LENSES.items():
        ag = {l: float(np.mean(v)) for l, v in san["agree"][n].items()}
        print(f"SANITY lens {n} top-1 agreement with the model's output, per layer: " +
              " ".join(f"L{l}:{v:.2f}" for l, v in ag.items()), flush=True)
        pickle.dump({"key": a.key, "lens": n, "hf": a.hf, "layers": sorted(J) + [L - 1], "n_layers": L, "dm": out[n],
                     "sanity": dict(sanity, lens_top1_agreement=ag, lens_path=[s for s in a.lens if s.startswith(n + "=")][0])},
                    open(os.path.join(a.out, f"band_{a.key}__{n}{fam}.pkl{sfx}"), "wb"))
    print("DONE", len(trials), "trials", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
