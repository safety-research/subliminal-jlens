"""Teacher-forced lens readouts: residual capture, whole-word ranks, random token sets.

Conventions shared by every readout in the write-up (src/rerun_lens.py and all its descendants):
  * the completion is teacher-forced under the prompt that produced it (prompt rendered with the model's chat template,
    tokenized with add_special_tokens=False; the response tokenized separately and appended);
  * read positions are those predicting each response token: the last prompt token through the second-to-last response
    token (`response_positions`);
  * lens logits at layer l: model.unembed(lens.transport(h_l, l)) with h_l the fp32 output of decoder block l
    (final norm + LM head in bf16, as jlens does), then cast to fp32;
  * rank of a token among whole-word tokens: position in a descending argsort of the logits with non-whole-word entries
    set to -inf (ties broken by sort order); rank of a token set = best (minimum) rank of its members.
"""
import random

import torch
from jlens.hooks import ActivationRecorder


@torch.no_grad()
def residuals(model, input_ids, layers):
    """{layer: [seq, d] fp32} for a single sequence (lens_check/common.py residuals())."""
    ids = torch.as_tensor(input_ids, device=model.input_device)[None]
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
        acts = {l: rec.activations[l][0].float() for l in layers}
    return acts


def response_positions(prompt_len, resp_len):
    return list(range(prompt_len - 1, prompt_len - 1 + resp_len))


def system_positions(text, offsets, system_content):
    """Positions of the system-prompt content tokens (tokens lying entirely inside the content's character span). For
    Qwen2.5 this equals the original rule (<|im_start|>-index + 3 .. next <|im_start|> - 3)."""
    c0 = text.index(system_content)
    c1 = c0 + len(system_content)
    return [i for i, (s, e) in enumerate(offsets) if s >= c0 and e <= c1 and e > s]


@torch.no_grad()
def lens_logits(model, lens, h, layer, V, fp32=False):
    """[P, V] fp32 lens logits for residuals h [P, d] at `layer` (identity transport for a layer the lens lacks).
    fp32=True: final norm in bf16 but the LM-head matmul (and softcap) in fp32 (Gemma tie diagnostic)."""
    t = lens.transport(h, layer) if layer in lens.jacobians else h
    if not fp32:
        return model.unembed(t).float()[:, :V]
    hn = model._final_norm(t.to(model._lm_head.weight.dtype)).float()
    z = hn @ model._lm_head.weight.float().T
    if getattr(model, "_logit_softcap", None) is not None:
        z = model._logit_softcap * torch.tanh(z / model._logit_softcap)
    return z[:, :V]


@torch.no_grad()
def ww_ranks(z, wwmask):
    """1-based rank of every vocabulary entry among whole-word tokens (argsort convention). z [P, V] -> ranks [P, V]."""
    masked = z.masked_fill(~wwmask, float("-inf"))
    order = masked.argsort(-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, order.shape[1] + 1, device=order.device).expand_as(order))
    return ranks


def draw_random_sets(ww_list, plan, n_sets=100):
    """Random whole-word token sets. plan: (seed, [size, size, ...]) drawn sequentially from one random.Random(seed), or
    a list of such tuples. Returns {(plan_index, size): LongTensor [n_sets, size]}.
    The originals: rerun_lens.py (0, sorted set sizes, e.g. [2, 3, 55, 65, 100]); animals_lens.py (0, [2, 3, 4]);
    onpolicy_conservative.py (0, [65]); qwen3_32b readout.py (12345, [3]); animals_nonanimal_positions.py
    [(0, [2]), (3, [3]), (4, [4])]."""
    plans = plan if isinstance(plan, list) else [plan]
    out = {}
    for pi, (seed, sizes) in enumerate(plans):
        rng = random.Random(seed)
        for n in sizes:
            out[(pi, n)] = torch.tensor([rng.sample(ww_list, n) for _ in range(n_sets)])
    return out


def parse_plans(strings):
    """['0:2,3,55,65,100', '0:65'] -> [(0, [2, 3, 55, 65, 100]), (0, [65])]."""
    out = []
    for s in strings:
        seed, sizes = s.split(":")
        out.append((int(seed), [int(x) for x in sizes.split(",") if x]))
    return out
