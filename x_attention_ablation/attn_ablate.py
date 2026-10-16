"""Attention ablation of the system prompt in the subliminal-learning teacher (Qwen2.5-7B-Instruct).

Blocks attention from a set of query positions to the system-prompt CONTENT tokens, at a chosen set
of layers, by replacing the attention mask passed to each decoder layer's self_attn (sdpa, boolean
mask, True = may attend). Query sets:
  A  = the assistant turn: '<|im_start|>assistant\\n' header + response tokens (every position that
       predicts a response token)
  B  = everything after the system content: the system turn's closing '<|im_end|>\\n', the user turn,
       and the assistant turn (so no later position reads the content directly)

modes:
  kl      teacher-forced per-token KL between owl-sysprompt and control teachers on owl-teacher responses,
          for no ablation and each ablation spec (Phase A)
  verify  masking checks (B@all vs empty-content system prompt with shifted positions; pre-assistant
          positions untouched under A; cached generation loop == teacher-forced masked forward)
  gen     sampling with KV cache and the ablation applied to every new token (Phase B / sanity checks).
          Output rows match supp_paper_discrepancy/hf_teacher_gen.py: {"id", "prompt"|"question", "response"}.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sljlens.prompts import system_prompt  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"

DEFAULT_SP = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
NL = 28  # Qwen2.5-7B decoder layers


def parse_layers(s):
    """'all' | 'a-b' (inclusive) | 'a-b,c-d'"""
    if s == "all":
        return list(range(NL))
    out = []
    for part in s.split(","):
        a, b = part.split("-")
        out += list(range(int(a), int(b) + 1))
    return out


# ---------------------------------------------------------------- tokenization with regions
class Enc:
    def __init__(self, tok):
        self.tok = tok
        self.cache = {}

    def ids(self, s):
        if s not in self.cache:
            self.cache[s] = self.tok(s, add_special_tokens=False)["input_ids"]
        return self.cache[s]

    def build(self, sp, user, response=None, check=True):
        """Return (ids, regions). regions: content=(cs,ce), sysclose_start, asst_start, resp_start, end."""
        segs = [("pre", "<|im_start|>system\n"), ("content", sp), ("sysclose", "<|im_end|>\n"),
                ("user", "<|im_start|>user\n" + user + "<|im_end|>\n"), ("asst", "<|im_start|>assistant\n")]
        if response is not None:
            segs.append(("resp", response + "<|im_end|>"))
        ids, pos = [], {}
        for name, s in segs:
            pos[name] = len(ids)
            ids += self.ids(s) if name != "user" and name != "resp" else self.tok(s, add_special_tokens=False)["input_ids"]
        if check:
            full = self.tok.apply_chat_template([{"role": "system", "content": sp}, {"role": "user", "content": user}],
                                                tokenize=False, add_generation_prompt=True)
            if response is not None:
                full += response + "<|im_end|>"
            ref = self.tok(full, add_special_tokens=False)["input_ids"]
            assert ref == ids, "segment-wise tokenization differs from chat template"
        reg = dict(cs=pos["content"], ce=pos["sysclose"], sysclose=pos["sysclose"], asst=pos["asst"],
                   resp=pos.get("resp", len(ids)), end=len(ids))
        return ids, reg


def qstart(reg, qset):
    return reg["asst"] if qset == "A" else reg["sysclose"]


# ---------------------------------------------------------------- mask hooks
class MaskSwitch:
    """Forward pre-hooks on every self_attn. self.masks: {layer_idx: bool mask} set before each forward;
    layers not in the dict get self.base."""

    def __init__(self, model):
        self.base = None
        self.masks = {}
        self.active = False
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._mk(i), with_kwargs=True))

    def _mk(self, i):
        def hook(mod, args, kwargs):
            if not self.active:
                return None
            kwargs["attention_mask"] = self.masks.get(i, self.base)
            return args, kwargs
        return hook


def tf_masks(regs, T, qset, device):
    """Right-padded teacher-forcing batch. Returns base (B,1,T,T) and ablated mask."""
    lens = torch.tensor([r["end"] for r in regs], device=device)
    ar = torch.arange(T, device=device)
    causal = ar[None, :] <= ar[:, None]                       # (T,T) [q,k]
    kvalid = ar[None, :] < lens[:, None]                      # (B,T)
    base = causal[None] & kvalid[:, None, :]
    base = base | torch.eye(T, dtype=torch.bool, device=device)[None]   # pads attend to self (no NaN rows)
    abl = base.clone()
    if qset is not None:
        for b, r in enumerate(regs):
            abl[b, qstart(r, qset):, r["cs"]:r["ce"]] = False
    return base[:, None], abl[:, None]


def load(model_path, dtype):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa").cuda().eval()
    assert model.config.num_hidden_layers == NL
    return tok, model


@torch.no_grad()
def tf_logits(model, sw, batch, qset=None, layers=(), pos_ids=None):
    """batch: list of (ids, reg). Returns float32 logits (B,T,V) and regs."""
    T = max(len(i) for i, _ in batch)
    pad = 0
    x = torch.full((len(batch), T), pad, dtype=torch.long, device="cuda")
    for b, (ids, _) in enumerate(batch):
        x[b, :len(ids)] = torch.tensor(ids)
    regs = [r for _, r in batch]
    base, abl = tf_masks(regs, T, qset, "cuda")
    sw.base = base
    sw.masks = {l: abl for l in layers} if qset else {}
    sw.active = True
    if pos_ids is None:
        pos_ids = torch.arange(T, device="cuda")[None].expand(len(batch), T)
    out = model(input_ids=x, position_ids=pos_ids, use_cache=False).logits.float()
    sw.active = False
    return out, regs


# ---------------------------------------------------------------- generation with cache + ablation
@torch.no_grad()
def generate(model, sw, tok, batch, qset, layers, max_new, eos_ids, return_steps=False):
    """batch: list of (prompt_ids, reg) (no response). Left-padded. Ablation blocks queries in qset
    from content keys at `layers`, for prefill and every decode step. Returns list of token lists."""
    B = len(batch)
    T0 = max(len(i) for i, _ in batch)
    offs = [T0 - len(i) for i, _ in batch]
    x = torch.full((B, T0), tok.pad_token_id, dtype=torch.long, device="cuda")
    for b, (ids, _) in enumerate(batch):
        x[b, offs[b]:] = torch.tensor(ids)
    valid = torch.zeros((B, T0 + max_new), dtype=torch.bool, device="cuda")
    for b in range(B):
        valid[b, offs[b]:T0] = True
    content = torch.zeros((B, T0 + max_new), dtype=torch.bool, device="cuda")
    qs = torch.zeros((B, T0), dtype=torch.bool, device="cuda")   # prefill positions in the query set
    for b, (_, r) in enumerate(batch):
        content[b, offs[b] + r["cs"]: offs[b] + r["ce"]] = True
        if qset:
            qs[b, offs[b] + qstart(r, qset):] = True
    pos = (valid[:, :T0].long().cumsum(-1) - 1).clamp(min=0)
    ar = torch.arange(T0, device="cuda")
    causal = ar[None, :] <= ar[:, None]
    base = causal[None] & valid[:, None, :T0] | torch.eye(T0, dtype=torch.bool, device="cuda")[None]
    abl = base & ~(qs[:, :, None] & content[:, None, :T0])
    sw.base, sw.masks, sw.active = base[:, None], ({l: abl[:, None] for l in layers} if qset else {}), True
    cache = DynamicCache(config=model.config)
    logits = model(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].float()
    steps = [logits] if return_steps else None
    done = torch.zeros(B, dtype=torch.bool, device="cuda")
    out = torch.full((B, max_new), tok.pad_token_id, dtype=torch.long, device="cuda")
    eos_t = torch.tensor(eos_ids, device="cuda")
    nextpos = pos[:, -1] + 1
    for t in range(max_new):
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
        out[:, t] = nxt
        done = done | torch.isin(nxt, eos_t)
        if done.all() or t == max_new - 1:
            break
        K = T0 + t + 1
        valid[:, K - 1] = True
        m_base = valid[:, None, None, :K]
        m_abl = m_base & ~content[:, None, None, :K]   # decode queries are always in A and B
        sw.base, sw.masks = m_base, ({l: m_abl for l in layers} if qset else {})
        logits = model(input_ids=nxt[:, None], position_ids=nextpos[:, None], past_key_values=cache,
                       use_cache=True).logits[:, -1].float()
        if return_steps:
            steps.append(logits)
        nextpos = nextpos + 1
    sw.active = False
    res = []
    for b in range(B):
        seq = out[b].tolist()
        cut = next((i for i, v in enumerate(seq) if v in eos_ids), None)
        res.append(seq[:cut + 1] if cut is not None else seq)
    return (res, steps) if return_steps else res


# ---------------------------------------------------------------- modes
def kl_rows(lp_p, lp_q):
    """KL(p||q) per row, from log-probs (N,V)."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)


def resp_slice(logits, regs, b):
    r = regs[b]
    return logits[b, r["resp"] - 1: r["end"] - 1]   # positions predicting response tokens (incl. final <|im_end|>)


def mode_kl(a, tok, model, sw):
    from sljlens.numbers import get_reject_reasons
    enc = Enc(tok)
    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if not get_reject_reasons(r["response"], min_value=0, max_value=999, max_count=10, banned_numbers=[])]
    rows = rows[: a.n]
    owl_sp, ctrl_sp = system_prompt(a.animal), DEFAULT_SP
    specs = [("none", None, [])]
    for q in ("A", "B"):
        for lname in a.ranges.split(";"):
            specs.append((f"{q}@{lname}", q, parse_layers(lname)))
    V = model.config.vocab_size
    acc = {s[0]: dict(kl_ctrl=0.0, kl_ctrl_abl=0.0, kl_owlfull=0.0, n=0, kl_seq=[], mp=torch.zeros(V, device="cuda"))
           for s in specs}
    mp_ctrl = torch.zeros(V, device="cuda")
    cat_acc = {}   # spec -> {cat: [sum, n]}
    digit_ids = {tok.convert_tokens_to_ids(str(d)) for d in range(10)}
    first_digit_acc = {s[0]: torch.zeros(V, device="cuda") for s in specs}
    first_digit_ctrl = torch.zeros(V, device="cuda")
    n_first = 0
    t0 = time.time()
    for b0 in range(0, len(rows), a.bs):
        chunk = rows[b0: b0 + a.bs]
        owl_b = [enc.build(owl_sp, r["prompt"], r["response"]) for r in chunk]
        ctl_b = [enc.build(ctrl_sp, r["prompt"], r["response"]) for r in chunk]
        lo, ro = tf_logits(model, sw, owl_b)
        lc, rc = tf_logits(model, sw, ctl_b)
        lp_owl = [F.log_softmax(resp_slice(lo, ro, b), -1) for b in range(len(chunk))]
        lp_ctl = [F.log_softmax(resp_slice(lc, rc, b), -1) for b in range(len(chunk))]
        del lo, lc
        tgt = [torch.tensor(owl_b[b][0][ro[b]["resp"]: ro[b]["end"]], device="cuda") for b in range(len(chunk))]
        for b in range(len(chunk)):
            mp_ctrl += lp_ctl[b].exp().sum(0)
            prev = [None] + tgt[b][:-1].tolist()
            fd = torch.tensor([t in digit_ids and (p is None or p not in digit_ids) for t, p in zip(tgt[b].tolist(), prev)], device="cuda")
            first_digit_ctrl += lp_ctl[b][fd].exp().sum(0)
            if b0 == 0 and b == 0:
                pass
            n_first += int(fd.sum())
        for name, q, layers in specs:
            if q is None:
                lp_abl = lp_owl
                lp_cabl = lp_ctl
            else:
                la, ra = tf_logits(model, sw, owl_b, q, layers)
                lp_abl = [F.log_softmax(resp_slice(la, ra, b), -1) for b in range(len(chunk))]
                del la
                lca, rca = tf_logits(model, sw, ctl_b, q, layers)
                lp_cabl = [F.log_softmax(resp_slice(lca, rca, b), -1) for b in range(len(chunk))]
                del lca
            A_ = acc[name]
            ca = cat_acc.setdefault(name, {})
            for b in range(len(chunk)):
                k1 = kl_rows(lp_abl[b], lp_ctl[b])
                A_["kl_ctrl"] += float(k1.sum())
                A_["kl_ctrl_abl"] += float(kl_rows(lp_abl[b], lp_cabl[b]).sum())
                A_["kl_owlfull"] += float(kl_rows(lp_abl[b], lp_owl[b]).sum())
                A_["n"] += k1.numel()
                A_["kl_seq"].append(float(k1.sum()))
                A_["mp"] += lp_abl[b].exp().sum(0)
                # direction of the shift: dp = p - p_ctrl, compared with the unablated owl shift (per position)
                pc = lp_ctl[b].exp()
                dpa = lp_abl[b].exp() - pc
                dpf = lp_owl[b].exp() - pc
                A_["dot"] = A_.get("dot", 0.0) + float((dpa * dpf).sum())
                A_["nf"] = A_.get("nf", 0.0) + float((dpf * dpf).sum())
                A_["na"] = A_.get("na", 0.0) + float((dpa * dpa).sum())
                # same in log-prob space with the control distribution as metric (Fisher-style)
                dla = lp_abl[b] - lp_ctl[b]
                dlf = lp_owl[b] - lp_ctl[b]
                dla = dla - (pc * dla).sum(-1, keepdim=True)
                dlf = dlf - (pc * dlf).sum(-1, keepdim=True)
                A_["fdot"] = A_.get("fdot", 0.0) + float((pc * dla * dlf).sum())
                A_["fnf"] = A_.get("fnf", 0.0) + float((pc * dlf * dlf).sum())
                A_["fna"] = A_.get("fna", 0.0) + float((pc * dla * dla).sum())
                prev = [None] + tgt[b][:-1].tolist()
                cats = []
                for t, p in zip(tgt[b].tolist(), prev):
                    if t in digit_ids:
                        cats.append("first_digit" if (p is None or p not in digit_ids) else "later_digit")
                    elif t in (tok.convert_tokens_to_ids("<|im_end|>"),):
                        cats.append("eos")
                    else:
                        cats.append("separator/other")
                for c, kv in zip(cats, k1.tolist()):
                    s = ca.setdefault(c, [0.0, 0])
                    s[0] += kv
                    s[1] += 1
                fd = torch.tensor([c == "first_digit" for c in cats], device="cuda")
                first_digit_acc[name] += lp_abl[b][fd].exp().sum(0)
        print(f"{b0 + len(chunk)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    N = acc["none"]["n"]
    out = {"n_samples": len(rows), "n_positions": N, "n_first_digit_positions": n_first, "specs": {}}
    dp_full = (acc["none"]["mp"] - mp_ctrl) / N
    top = torch.topk(dp_full, 15).indices.tolist() + torch.topk(-dp_full, 10).indices.tolist()
    dfd_full = (first_digit_acc["none"] - first_digit_ctrl) / max(1, n_first)
    topfd = torch.topk(dfd_full, 10).indices.tolist() + torch.topk(-dfd_full, 10).indices.tolist()
    for name, q, layers in specs:
        A_ = acc[name]
        dp = (A_["mp"] - mp_ctrl) / N
        dfd = (first_digit_acc[name] - first_digit_ctrl) / max(1, n_first)
        out["specs"][name] = {
            "qset": q, "layers": layers,
            "kl_to_ctrl": A_["kl_ctrl"] / N,
            "kl_to_ctrl_same_ablation": A_["kl_ctrl_abl"] / N,
            "kl_to_owl_unablated": A_["kl_owlfull"] / N,
            "kl_per_seq_mean": sum(A_["kl_seq"]) / len(A_["kl_seq"]),
            "dp_pos_projection": A_["dot"] / A_["nf"], "dp_pos_cosine": A_["dot"] / math.sqrt(A_["nf"] * A_["na"]),
            "dp_pos_norm_ratio": math.sqrt(A_["na"] / A_["nf"]),
            "fisher_projection": A_["fdot"] / A_["fnf"], "fisher_cosine": A_["fdot"] / math.sqrt(A_["fnf"] * A_["fna"]),
            "fisher_norm_ratio": math.sqrt(A_["fna"] / A_["fnf"]),
            "kl_by_target_cat": {c: v[0] / v[1] for c, v in cat_acc[name].items()},
            "n_by_target_cat": {c: v[1] for c, v in cat_acc[name].items()},
            # how much of the full prob shift (owl - ctrl, mean over positions) survives: projection coefficient
            "dp_projection": float((dp * dp_full).sum() / (dp_full * dp_full).sum()),
            "dp_l1": float(dp.abs().sum()),
            "dp_top": {tok.decode([t]): [float(dp_full[t]), float(dp[t])] for t in top},
            "dfirstdigit_projection": float((dfd * dfd_full).sum() / (dfd_full * dfd_full).sum()),
            "dfirstdigit_top": {tok.decode([t]): [float(dfd_full[t]), float(dfd[t])] for t in topfd},
        }
    out["mean_p_ctrl_top"] = {tok.decode([t]): float(mp_ctrl[t] / N) for t in top}
    json.dump(out, open(a.out, "w"), indent=1)
    for name in out["specs"]:
        s = out["specs"][name]
        print(f"{name:12s} KL->ctrl {s['kl_to_ctrl']:.4f}  KL->ctrl(same abl) {s['kl_to_ctrl_same_ablation']:.4f}  "
              f"KL->owl {s['kl_to_owl_unablated']:.4f} pos_proj {s['dp_pos_projection']:.3f} pos_cos {s['dp_pos_cosine']:.3f} "
              f"fisher_proj {s['fisher_projection']:.3f} fisher_cos {s['fisher_cosine']:.3f}")


def mode_verify(a, tok, model, sw):
    from sljlens.numbers import get_reject_reasons  # noqa: F401
    enc = Enc(tok)
    rows = [json.loads(l) for l in open(a.data)][: a.n]
    owl_sp = system_prompt(a.animal)
    rep = {"dtype": str(model.dtype)}
    # 1. B@all on owl == empty-content system prompt with post-content positions shifted by len(content)
    mx, kls = 0.0, []
    mx_noise, kls_noise = 0.0, []
    for r in rows:
        ob = [enc.build(owl_sp, r["prompt"], r["response"])]
        eb = [enc.build("", r["prompt"], r["response"], check=False)]
        L = ob[0][1]["ce"] - ob[0][1]["cs"]
        assert eb[0][1]["ce"] == eb[0][1]["cs"] == 3
        lo, ro = tf_logits(model, sw, ob, "B", list(range(NL)))
        Te = len(eb[0][0])
        pe = torch.arange(Te, device="cuda")
        pe[3:] += L
        le, _ = tf_logits(model, sw, eb, pos_ids=pe[None])
        a_ = lo[0, ro[0]["ce"]:]
        b_ = le[0, 3:]
        assert a_.shape == b_.shape
        mx = max(mx, float((a_ - b_).abs().max()))
        kls.append(float(kl_rows(F.log_softmax(a_, -1), F.log_softmax(b_, -1)).mean()))
        # reference: same comparison WITHOUT the ablation (should be large)
        lo2, _ = tf_logits(model, sw, ob)
        a2 = lo2[0, ro[0]["ce"]:]
        mx_noise = max(mx_noise, float((a2 - b_).abs().max()))
        kls_noise.append(float(kl_rows(F.log_softmax(a2, -1), F.log_softmax(b_, -1)).mean()))
    rep["B_all_vs_empty_sys_shifted_pos"] = {"max_abs_logit_diff": mx, "mean_kl": sum(kls) / len(kls),
                                             "unablated_vs_empty_max_abs_logit_diff": mx_noise,
                                             "unablated_vs_empty_mean_kl": sum(kls_noise) / len(kls_noise)}
    # 1b. B@all vs empty-content WITHOUT the position shift (shows the positional offset matters only slightly)
    kls_np = []
    for r in rows:
        ob = [enc.build(owl_sp, r["prompt"], r["response"])]
        eb = [enc.build("", r["prompt"], r["response"], check=False)]
        lo, ro = tf_logits(model, sw, ob, "B", list(range(NL)))
        le, _ = tf_logits(model, sw, eb)
        kls_np.append(float(kl_rows(F.log_softmax(lo[0, ro[0]["ce"]:], -1), F.log_softmax(le[0, 3:], -1)).mean()))
    rep["B_all_vs_empty_sys_unshifted_mean_kl"] = sum(kls_np) / len(kls_np)
    # 2. A@all leaves positions before the assistant header bit-identical; changes later ones
    pre_mx, post_mx = 0.0, 0.0
    for r in rows:
        ob = [enc.build(owl_sp, r["prompt"], r["response"])]
        l0, ro = tf_logits(model, sw, ob)
        l1, _ = tf_logits(model, sw, ob, "A", list(range(NL)))
        s = ro[0]["asst"]
        pre_mx = max(pre_mx, float((l0[0, :s] - l1[0, :s]).abs().max()))
        post_mx = max(post_mx, float((l0[0, s:] - l1[0, s:]).abs().max()))
    rep["A_all_pre_assistant_max_abs_diff"] = pre_mx
    rep["A_all_assistant_max_abs_diff"] = post_mx
    # 2b. hooks inactive => identical to a plain forward with no hooks involved
    sw.active = False
    ob = enc.build(owl_sp, rows[0]["prompt"], rows[0]["response"])
    plain = model(input_ids=torch.tensor([ob[0]], device="cuda")).logits.float()
    l0, _ = tf_logits(model, sw, [ob])
    rep["hooked_base_vs_plain_forward_max_abs_diff"] = float((plain - l0).abs().max())
    # 3. cached generation == teacher-forced forward with the same mask, on the generated tokens
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    for q, layers, lname in [(None, [], "none"), ("A", list(range(NL)), "A@all"), ("A", parse_layers("9-18"), "A@9-18")]:
        pb = [enc.build(owl_sp, r["prompt"]) for r in rows[:8]]
        torch.manual_seed(0)
        seqs, steps = generate(model, sw, tok, pb, q, layers, 40, eos, return_steps=True)
        mxd, kld, ntok = 0.0, [], 0
        for b in range(len(pb)):
            ids = pb[b][0] + seqs[b]
            reg = dict(pb[b][1], resp=len(pb[b][0]), end=len(ids))
            lt, _ = tf_logits(model, sw, [(ids, reg)], q, layers)
            for t in range(len(seqs[b])):
                g = steps[t][b]
                f = lt[0, len(pb[b][0]) - 1 + t]
                mxd = max(mxd, float((g - f).abs().max()))
                kld.append(float(kl_rows(F.log_softmax(f, -1)[None], F.log_softmax(g, -1)[None])))
                ntok += 1
        rep[f"gen_cache_vs_teacher_forced[{lname}]"] = {"max_abs_logit_diff": mxd, "mean_kl": sum(kld) / len(kld),
                                                        "max_kl": max(kld), "n_tokens": ntok,
                                                        "example": tok.decode(seqs[0])}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(a.out, "w"), indent=1)


def mode_gen(a, tok, model, sw):
    enc = Enc(tok)
    if a.prompts:
        rows = [json.loads(l) for l in open(a.prompts)]
        rows = [dict(r, user=r["prompt"]) for r in rows]
    else:
        qs = [json.loads(l) for l in open(a.questions)]
        rows = [dict(id=f"{q['id']}_{k}", user=q["messages"][-1]["content"], question=q["messages"][-1]["content"])
                for q in qs for k in range(a.n_per_q)]
        assert all(len(q["messages"]) == 1 for q in qs)
    if a.limit:
        rows = rows[: a.limit]
    rows = rows[a.shard:: a.nshards]
    sp = system_prompt(a.animal) if a.animal else DEFAULT_SP
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    rows = [r for r in rows if r["id"] not in done]
    layers = parse_layers(a.layers) if a.qset else []
    print(f"{len(rows)} to generate (skip {len(done)}); system={sp!r} qset={a.qset} layers={layers}", flush=True)
    torch.manual_seed(a.seed * 1000 + a.shard)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    t0 = time.time()
    with open(a.out, "a") as f:
        for b in range(0, len(rows), a.batch):
            chunk = rows[b: b + a.batch]
            pb = [enc.build(sp, r["user"], check=(b == 0)) for r in chunk]
            seqs = generate(model, sw, tok, pb, a.qset, layers, a.max_new, eos)
            for r, s in zip(chunk, seqs):
                o = {"id": r["id"], "response": tok.decode(s, skip_special_tokens=True)}
                if "prompt" in r:
                    o["prompt"] = r["prompt"]
                else:
                    o["question"] = r["question"]
                f.write(json.dumps(o) + "\n")
            f.flush()
            print(f"{b + len(chunk)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


def mode_score(a, tok, model, sw):
    """Mean per-token log-likelihood ratio log p_owl(y) - log p_ctrl(y) of each SFT dataset's responses,
    under the unablated owl-sysprompt teacher, the A@all-ablated owl teacher, and the control teacher."""
    enc = Enc(tok)
    teachers = [("owl", system_prompt(a.animal), None, []), ("owl_A@all", system_prompt(a.animal), "A", list(range(NL))),
                ("owl_B@0-8", system_prompt(a.animal), "B", parse_layers("0-8")), ("ctrl", DEFAULT_SP, None, [])]
    res = {}
    for path in a.datasets.split(","):
        rows = [json.loads(l) for l in open(path)][: a.n]
        tot = {t[0]: 0.0 for t in teachers}
        ntok = 0
        for b0 in range(0, len(rows), a.bs):
            chunk = rows[b0: b0 + a.bs]
            for name, sp, q, layers in teachers:
                bt = [enc.build(sp, r["messages"][0]["content"], r["messages"][1]["content"], check=False) for r in chunk]
                lg, rg = tf_logits(model, sw, bt, q, layers)
                for b in range(len(chunk)):
                    lp = F.log_softmax(resp_slice(lg, rg, b), -1)
                    tgt = torch.tensor(bt[b][0][rg[b]["resp"]: rg[b]["end"]], device="cuda")
                    tot[name] += float(lp.gather(-1, tgt[:, None]).sum())
                    if name == "ctrl":
                        ntok += len(tgt)
                del lg
        res[path] = {"n": len(rows), "tokens": ntok, **{f"llr_{t}_vs_ctrl": (tot[t] - tot["ctrl"]) / ntok for t in tot if t != "ctrl"},
                     **{f"mean_logp_{t}": tot[t] / ntok for t in tot}}
        print(path, json.dumps(res[path]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kl", "verify", "gen", "score"])
    ap.add_argument("--model", default=MODEL, help="HF id or local weights dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--animal", default="owl")
    ap.add_argument("--data", help="teacher rows jsonl (kl/verify)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ranges", default="all;0-8;9-18;19-27;0-6;7-13;14-20;21-27")
    ap.add_argument("--datasets")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--prompts")
    ap.add_argument("--questions")
    ap.add_argument("--n-per-q", type=int, default=10)
    ap.add_argument("--qset", default=None, choices=[None, "A", "B"])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "gen" and a.animal in ("none", ""):
        a.animal = None
    tok, model = load(a.model, torch.float32 if a.fp32 else torch.bfloat16)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    sw = MaskSwitch(model)
    {"kl": mode_kl, "verify": mode_verify, "gen": mode_gen, "score": mode_score}[a.mode](a, tok, model, sw)
