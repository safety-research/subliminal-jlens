# Appendix H: R-Lens and J++ Lens don't change the main result

Reproduces the post's **"Claude-written appendix: R-Lens and J++ Lens don't change the main result"**: the choice of the
strongest J++ configuration on home-turf criteria (23 configurations), the toy-task table, and the number-task table, which
is now computed on penalty-free teacher data (a 27,622-completion subset of the 1.42M owl pool and the same number of
no-system-prompt completions). `run.sh` runs everything; part A needs the official J++ code and its own environment.

## What is fitted, and with what

* **J++ Lens** (Ayonrinde & Lindsey, "J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses") and **R-Lens**
  (Blank, Bhatia & Nanda), fitted with the **official code**, github.com/safety-research/jpp_lens at commit `93f02cd`, run in
  its own environment built from its `uv.lock`, through its own CLI (`scripts/jpp_cli.py`). The estimator is not
  re-implemented. `jpp_run.py` only (1) feeds the router fit a JSON dump of the WikiText pool, so GPU jobs need no network,
  (2) adds a `grade` stage that writes the model-correctness column for Qwen2.5-7B with the library's own grader (the shipped
  CSV only grades Qwen3.6-27B), (3) switches the expert-weight search between the official signed search and two variants
  (`jpp_variants.py`), and (4) adds k-fold item splits for cross-fitting.
* **Round 1, the paper's recipe** (`run.sh` A1): 64 WikiText-103 records x 128 tokens (skip 16), target = final block, E = 8
  k-means experts per layer on PCA-64, LRP backward pass, signed expert weights fitted on the official 20% "fit" split of the
  readout evals (70 items; 366 items are scoreable for Qwen2.5-7B, 296 held out), at all 27 source layers. The R-Lens is the
  pooled LRP Jacobian of the same fit (the J++ paper's matched comparison). Also the matched plain-gradient J-Lens, J++
  without LRP, and the best-single-expert ablation (`best_expert.py`).
* **Round 2, stronger settings** (`run.sh` A2), every fit with the official code:
  * fit data: WikiText 512 records x 256 tokens (records of at least 1,500 characters; 122k fit positions, 17x the paper's)
    and an in-domain chat corpus (the lens-refit corpora A + B of appendix G, packed to 314 texts of at least 256 tokens);
    `build_fit_pools.py`;
  * E = 8, 16, 32 (WikiText) and 8, 16 (chat); routers fitted with the official code on the GPU (`fit_router_gpu.py`; the
    CPU version was CPU-bound for over 40 minutes per K);
  * expert weights: signed (official), non-negative (official search plus a projection onto w >= 0 after each Adam step),
    best single expert (official start only); fitted on the official split (for comparison with round 1), in 5 folds
    (cross-fitted home-turf recall: every item scored by weights that did not see it), and on all 366 items (the deployed
    lenses); `fit_variants.py`;
  * the R-Lens of each fit is its pooled LRP Jacobian.
* **Selection, on home-turf criteria only, fixed before looking at the number task:** (1) pass the positive controls in the
  read band (owl top-1 at least 75% of the original J-Lens's rate on owl-prompted trivia and copying, 100% on the owl system
  prompt's own tokens, about 0% under the horse prompt and no prompt); (2) among those, the highest cross-validated readout
  recall@10, toy task as tie-break. The winner is **WikiText 512 x 256, E = 16, signed weights** (`r2_wlK16_signed`): recall@10
  49.0% cross-fitted (47.6% on the official split) against 34.7% for the original J-Lens. The post's R-Lens is the WikiText
  512 x 256 one (`r2_wlK8_rlens`; recall@10 39.9%). The round-1 official J++ Lens and two configurations with higher recall
  (WikiText E = 32 signed, chat E = 16 signed) fail the controls: their signed weights nearly cancel in the late layers (owl
  top-1 on owl-prompted trivia 19-37%, against 67% for the original).
* Every lens is converted to the reference jlens format (fp32; `to_jlens.py`, or directly by `fit_variants.py`), so all
  readouts use this release's own scripts.

## Readouts (this release's scripts)

* Read band and toy task: `scripts/derive_band.py --trials-out`, then `dm_paper_table.py` (the post's table). Bands: L22-26
  for the original lens, L21-26 for every round-2 lens (chat E = 16 signed: L21-25).
* Positive controls and the 300-completion set (100 held-out prompts x 3, vLLM, no repetition penalty; see the top-level
  README): `scripts/lens_readout.py` + `scripts/task_table.py`.
* Home turf: `home_eval.py` (round 1, official split) and `r2_home.py` (round 2, every variant and protocol).
* **The number task (the post's table):** `make_subsets.py` draws 27,622 filter-passing completions from the owl pool
  (03_result / 06; vLLM, T = 1, no repetition penalty, max 128 tokens, 4 samples per prompt; each read under the owl system
  prompt that produced it) and 27,622 from the 81,124 filter-passing completions of the no-system-prompt teacher draws PM,
  PMt1 and PR of section 01 (paper sampling, no penalty), both with `random.Random(20261015)` over the sorted candidates.
  `scripts/score_lens.py --band 0-26 --rand-per-layer` scores each lens once at every layer (owl = `" owl"`, `" Owl"`;
  ranks among whole-word tokens; best over positions x layers; chance from the same 100 random 2-token whole-word sets under
  the identical rule), `rates_band.py` takes any band from that file, `number_table.py` prints the table, `overlap_band.py`
  the hit overlap.

## Expected

Toy task (top-1 / top-5, % of trials; 550 / 660 / 770 / 660 / 110 trials):

| lens | layers | think about X | X mentioned | don't think about X | ignore X | no instruction |
| --- | --- | --- | --- | --- | --- | --- |
| J-Lens (ours) | L22-26 | 45 / 74 | 52 / 82 | 0 / 2 | 1 / 4 | 0 / 2 |
| J++ Lens (best) | L21-26 | 57 / 87 | 63 / 88 | 1 / 2 | 2 / 6 | 1 / 1 |
| J-Lens (ours) | all layers | 47 / 78 | 54 / 83 | 3 / 9 | 4 / 12 | 3 / 5 |
| J++ Lens (best) | all layers | 59 / 88 | 65 / 89 | 2 / 14 | 4 / 19 | 2 / 11 |

Number task, 27,622 owl-teacher completions (top-1 / top-5 / top-20; `number_table.py`):

| lens | layers | owl teacher | chance | control teacher (no system prompt) |
| --- | --- | --- | --- | --- |
| J-Lens (ours) | L22-26 | 1.6 / 3.0 / 6.3% | 0.25 / 1.5 / 5.5% | 0.01 / 0.03 / 0.32% |
| R-Lens | L21-26 | 0.67 / 1.3 / 2.7% | 0.46 / 2.3 / 6.1% | 0.00 / 0.00 / 0.01% |
| J++ Lens (best) | L21-26 | 1.4 / 6.7 / 29% | 0.33 / 2.2 / 7.4% | 0.06 / 1.2 / 13% |
| J-Lens (ours) | all layers | 1.6 / 3.1 / 7.3% | 1.2 / 5.0 / 14% | 0.01 / 0.06 / 0.90% |
| R-Lens | all layers | 0.67 / 1.3 / 2.8% | 1.1 / 4.5 / 13% | 0.00 / 0.01 / 0.18% |
| J++ Lens (best) | all layers | 1.4 / 7.1 / 36% | 1.9 / 7.4 / 21% | 0.07 / 2.0 / 22% |

Unrounded, the J-Lens band row is 1.56 / 3.02 / 6.30% (chance 0.25 / 1.49 / 5.53%), 1.10 / 2.22 / 5.06% without position 0,
median best rank 213. The full 1.42M pool reads 1.65 / 3.15 / 6.37% (03_result); the gap is subset sampling (the pool's own
scores restricted to these 27,622 give 1.55%). Median best ranks: R-Lens 541, J++ 52.

Other statements in the appendix:
* On the 300-completion set, the best J++ Lens reads owl 3 / 9 / 29% under the owl prompt, 0 / 3 / 21% under the horse
  prompt and 0 / 1 / 13% with no system prompt (original lens 3 / 5 / 7%, 0 / 0 / 0%, 0 / 0 / 0.7%): its owl-specific margin
  over the horse prompt (+3 / +6 / +8 points) is essentially the original's.
* Top-1 hit overlap (`overlap_band.py`): on the penalty-free subset the J++ Lens's 380 top-1 hits include 211 of the original
  lens's 430 (5.9 expected if independent); the R-Lens's 185 include 181. The post's "190 of its 330" is the same statistic on
  the earlier HF-sampled data (`tables.py`, below).
* Max over all 23 configurations (earlier HF-sampled data, `results/jpp_rlens.md` of the original runs): own-band top-1 at
  most 3.0% (chance 0.8%), from the in-domain chat lens with E = 16 signed weights, which fails the controls and whose signal
  is all at position 0 (0.3% without it).

Earlier version of the number table (superseded; the original HF-sampled owl teacher data with Qwen2.5's implicit repetition
penalty 1.05, 27,622 owl and 27,373 control completions, every lens scored in one pass; `tables.py` and `rates_band.py` on
those score files): J-Lens 1.42 / 2.73 / 5.74% (chance 0.25 / 1.49 / 5.49%), R-Lens 0.59 / 1.28 / 2.39% (0.46 / 2.26 / 6.04%),
J++ best 1.19 / 6.72 / 30.04% (0.33 / 2.27 / 7.40%), control teacher for the J++ best 0.05 / 1.20 / 13.26%. Every number
moves by little between the two data sets, and the conclusion is the same.

## Compute

Round 1 about 6.2 H200 GPU-hours (fitting about 1.1); round 2 about 41 GPU-hours, mostly on an 8 x H100 machine (about
10 of them were lost to a CPU-bound router fit and to thread oversubscription, hence `fit_router_gpu.py` and
`OMP_NUM_THREADS=4` in `run.sh`); part C 6 scoring jobs of about 7 minutes on one H200 (0.7 GPU-hours). The E = 16 expert
Jacobians are about 12 GB.

## Deviations from the papers (all from the original runs)

27 source layers instead of 7 (weights are fitted per layer, so this only adds layers); correctness graded for Qwen2.5-7B
with the library's own grader; round-2 routers fitted on the GPU and fits run with `--jacobian-rows-per-pass 32` instead of
16 (batching only); the round-2 variants (8x the fit data, 256-token windows, E up to 32, the in-domain corpus, non-negative
and best-expert weights, weights from all items with cross-fitted evaluation) are labelled wherever they appear; readouts rank
among whole-word tokens (a subset of J++'s filtered vocabulary); maps stored fp32 (the original lens is fp16); the R-Lens is
the J++ paper's matched recipe, not the R-Lens authors' released recipe (none exists for Qwen2.5-7B).
