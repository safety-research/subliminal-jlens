# Extra: J++ Lens and R-Lens on the owl number task (cited in the introduction)

Reproduces the introduction's claim that "this effect also reproduces with recent suggested improvements to the J-Lens, such
as R-Lens and J++ Lens": neither newer lens reads owl more strongly than the original J-Lens while the Qwen2.5-7B owl teacher
writes numbers, relative to chance. The post has no section of its own for this; the numbers below are from the original
write-up of the comparison.

## What is fitted, and with what

* **J++ Lens** (Ayonrinde & Lindsey, "J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses") and **R-Lens**
  (Blank, Bhatia & Nanda), fitted with the **official code**, github.com/safety-research/jpp_lens at commit `93f02cd`, run in
  its own environment built from its `uv.lock`, through its own CLI (`scripts/jpp_cli.py`). Nothing is re-implemented.
  `jpp_run.py` only (1) feeds the router fit a JSON dump of exactly `jlens.examples.load_wikitext_prompts(1000)` (so GPU jobs
  need no network) and (2) adds a `grade` stage that writes the model-correctness column for Qwen2.5-7B with the library's own
  grader (the shipped CSV only grades Qwen3.6-27B).
* Recipe: the repo defaults (64 WikiText-103 records, 128 tokens, skip 16, target = final block, E = 8 k-means experts per
  layer on PCA-64, signed expert weights fitted on the 20% "fit" split of the readout evals) at all 27 source layers (the paper
  uses 7; we need the band rule and all-layer readouts). The R-Lens is the pooled LRP Jacobian of the same fit (the J++ paper's
  matched comparison).
* Diagnostics from the same fit: the matched J-Lens (pooled plain-gradient Jacobian; checks the port against our lens), J++
  without LRP, and J++ with the best single expert per layer (`best_expert.py`; one of the paper's ablations, used as the
  stand-in for a J++ that passes our positive controls).
* `to_jlens.py` converts each lens to the reference jlens format (fp32), so every readout below uses this release's own
  scripts unchanged.

## Readouts (this release's scripts)

* Home turf: the official readout evals (`home_eval.py` on the official `evaluate` output): recall@10 under the paper's
  7-layer protocol 34.7% (our J-Lens), 38.3% (R-Lens), 43.4% (J++), reproducing the paper's ordering.
* Toy task and read band: `scripts/derive_band.py --trials-out` + `04_lens_strength_toy_task/dm_strength.py`. Focus top-1 at
  each lens's own band: original 45%, matched J 44%, R-Lens 42%, J++ 47%, J++ without LRP 50%, best expert 47%. Bands: L22-26
  (original, matched, without LRP), L21-26 (R-Lens, J++, best expert).
* Positive controls (the other tasks, off-policy): `scripts/lens_readout.py` + `scripts/task_table.py`. The official J++ Lens
  fails them in band (owl top-1 on owl-prompted trivia 19% vs 67% for the original lens; never top-1 on the owl system prompt's
  own tokens): its signed expert weights nearly cancel at layers 21-26.
* The owl number task: `scripts/score_lens.py` on the original HF-sampled teacher data (27,622 filter-passing owl-teacher
  completions read under the owl prompt; 27,373 control-teacher completions under no prompt), at each lens's band and at all
  layers; `tables.py`.

Expected main table (owl teacher, top-1 / 5 / 20 %, chance in parentheses; all lenses scored in one 30k pass each):

| lens (band) | owl teacher | excl. position 0 | median best rank | control teacher |
|---|---|---|---|---|
| J-Lens original (L22-26) | 1.42 / 2.73 / 5.74 (0.25 / 1.49 / 5.49) | 0.92 / 1.95 / 4.47 | 220 | 0.00 / 0.03 / 0.28 |
| J-Lens matched (L22-26) | 1.22 / 2.29 / 4.28 (0.24 / 1.49 / 5.18) | 0.81 / 1.59 / 3.11 | 382 | 0.00 / 0.02 / 0.11 |
| R-Lens (L21-26) | 0.52 / 1.15 / 2.18 (0.37 / 2.03 / 5.81) | 0.27 / 0.61 / 1.30 | 726 | 0.00 / 0.00 / 0.00 |
| J++ Lens (L21-26), fails controls | 0.34 / 0.93 / 1.93 (1.13 / 3.33 / 8.97) | 0.15 / 0.41 / 0.96 | 1093 | 0.00 / 0.00 / 0.00 |
| J++ best expert (L22-26) | 0.58 / 1.19 / 2.32 (0.43 / 1.98 / 5.96) | 0.33 / 0.70 / 1.45 | 799 | 0.00 / 0.00 / 0.01 |
| J++ without LRP (L22-26) | 2.12 / 4.08 / 8.01 (1.10 / 3.05 / 7.63) | 1.50 / 2.99 / 6.55 | 293 | 0.00 / 0.00 / 0.08 |

The newer lenses' top-1 hits are almost a subset of the original lens's (143 of the R-Lens's 145, 92 of J++'s 93, 157 of the
best expert's 160; about 2 expected by chance). The original lens reads 1.42% here and 1.49% in the headline (section 03):
the same lens on the same data, scored in one 30k batch pass vs. two 15k passes; bf16 near-ties move with batch composition.

## Running it

`run.sh`: part A in the jpp_lens checkout (`JPP_LENS_DIR`, `JPP_PYTHON`), part B in this release's environment (plus
`pandas` and `tabulate` for `home_eval.py`). About 6.2
H200 GPU-hours, of which fitting is about 1.1. Needs section 01's teacher shards and section 03's shared generations.

Deviations from the paper's protocol, all from the original runs: 27 source layers instead of 7; correctness graded for
Qwen2.5-7B (296 scoreable items, 59 in the weight-fitting split, about half the paper's); readouts rank among whole-word
tokens (a subset of J++'s filtered vocabulary, so its Readout Filtering is implied); maps stored fp32 (our original lens is
fp16); the R-Lens uses the J++ paper's matched recipe, not the R-Lens authors' released recipe (none exists for Qwen2.5-7B).
