#!/usr/bin/env python3
"""Round-2 J++ expert-weight variants, as switchable patches of the official fitter (workspace_lens.fitting.
condense_experts.fitter). 'signed' = the official search (unchanged). 'nonneg' = the official search loop with a
projection onto w >= 0 after every Adam step (all official starts are non-negative). 'best' = only the official
select_best start (run with steps=0): the one-hot best single expert per layer on the fitting items (paper Table 4)."""
import torch as t
from workspace_lens.fitting.condense_experts import fitter as _fitter

_ORIG_FIT = _fitter.fit_expert_weights
_ORIG_STARTS = _fitter.build_search_starting_points


def _fit_nonneg(logits_PEV, pairs, *, excluded_mask_Bool_V, initial_weightings, condense_config):
    device = logits_PEV.device
    best = None
    for init_name, w0 in initial_weightings.items():
        weights_E = (w0.float() / w0.float().norm()).to(device).clone().requires_grad_(True)
        opt = t.optim.Adam([weights_E], lr=condense_config.learning_rate)
        for step in range(condense_config.steps + 1):
            unit = weights_E / weights_E.norm().clamp_min(1e-12)
            margin_Q, spread_Q = _fitter.hit_margins(_fitter.combined_logits(logits_PEV, unit), pairs,
                                                     excluded_mask_Bool_V=excluded_mask_Bool_V, k=condense_config.pass_k)
            sur = _fitter.soft_macro_pass(margin_Q, spread_Q, pairs, temperature=condense_config.temperature)
            if step % condense_config.exact_every == 0 or step == condense_config.steps:
                ex = _fitter.exact_macro_pass(margin_Q.detach(), pairs)
                cand = (ex, float(sur.detach()), _fitter.LearnedWeights(unit.detach().cpu().clone(), ex, init_name, step))
                if best is None or cand[:2] > best[:2]:
                    best = cand
            if step == condense_config.steps:
                break
            opt.zero_grad()
            (-sur).backward()
            opt.step()
            with t.no_grad():
                weights_E.clamp_(min=0)  # the one added line: projection onto the non-negative orthant
                if float(weights_E.sum()) == 0:
                    weights_E.fill_(1.0)
    return best[2]


def _best_only(scores_E, *, top_n):
    return {"select_best": _ORIG_STARTS(scores_E, top_n=top_n)["select_best"]}


def set_variant(name):
    _fitter.fit_expert_weights = {"signed": _ORIG_FIT, "nonneg": _fit_nonneg, "best": _ORIG_FIT}[name]
    _fitter.build_search_starting_points = _best_only if name == "best" else _ORIG_STARTS
