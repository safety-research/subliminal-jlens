# 03 Result

Reproduces the post's **"Result"** section: the headline table, and the two tables in "Lens readings on the system prompt's
own tokens and on open-ended answers". `run.sh` runs everything below (`run_pool.sh` for the pool, `collect.sh` for the
shared readout, then the CPU tables).

## Headline table

| Teacher | top-1 | top-5 | top-20 | median best rank (of ~40,000 tokens) | source |
|---|---|---|---|---|---|
| chance (random pairs of whole-word tokens, same max rule) | 0.26% | 1.5% | 5.5% | | `headline.py --scores` (the pool) |
| subliminal learning system prompt (1,421,727 completions) | 1.65% | 3.15% | 6.37% | 213 | `headline.py --scores` (the pool) |
| same prompt with "horses" (300) | 0% | 0% | 0% | 397 | `headline.py --readout` (the 300-completion set) |
| no system prompt (300) | 0% | 0% | 0% | 517 | `headline.py --readout` (the 300-completion set) |

* **Subliminal-prompt row and chance: the owl-teacher pool** (`run_pool.sh`). 390k fresh number prompts from the paper's
  generator (3 rounds, seeds 20261008 / 20261009 / 20261010, 90k / 100k / 200k prompts, none identical to the 30k training
  prompts; `06_addendum_lens_selected_sft/make_prompts.py`), 4 teacher samples per prompt under the owl system prompt, vLLM,
  T = 1, top-p 1, no top-k, **no repetition penalty**, max 128 tokens, per-request seeds from `--seed 0`: 1.56M completions,
  1,421,727 passing the paper's number filter. `scripts/score_lens.py` reads every completion under the prompt that produced
  it: owl set `" owl"`, `" Owl"`, read band L22-26, every position predicting a response token, ranks among whole-word tokens,
  best over band layers x positions; chance from the original's 100 random 2-token sets (`random.Random(0)`), same rule.
  `headline.py` aggregates. Output on the original score files: top-1 / 5 / 20 = 1.65 / 3.15 / 6.37%, chance
  0.26 / 1.50 / 5.53%, median best rank 213; without position 0 (the last prompt token) 1.18 / 2.35 / 5.15% (chance
  0.25 / 1.48 / 5.36%); per round 1.65 / 1.66 / 1.64% top-1; fraction of read positions with owl top-1 / 5 / 20 0.060 / 0.127 /
  0.279% (the "(0.1%)" in the on-policy appendix). The same pool is the addendum's (section 06).
* **"Horses" and no-system-prompt rows: the 300-completion set.** These teachers have no large pool; their rows are the
  shared on-policy readout (`collect.sh`, below) of the 300-completion set: the 100 number prompts held out from the training
  data (lines 29,000-29,099 of the 30k prompt list) x 3 samples per teacher, vLLM, T = 1, no repetition penalty, max 128
  tokens. `headline.py --readout` also prints the owl-prompt row of that set (3.7 / 4.7 / 7.0%, median 240), which is what
  earlier versions of the table showed; its 11 hits come from 7 prompts (hits cluster by prompt), so that estimate was high by
  about 2x. The no-system-prompt row is 0 / 0 / 0.3% unrounded.
* **Superseded headline.** Earlier versions read the 27,622 filter-passing completions of the original HF-sampled owl-teacher
  data (the students' training data, sampled with Qwen2.5's implicit repetition penalty of 1.05; `supp_paper_discrepancy/`):
  1.49 / 2.79 / 5.81%, chance 0.25 / 1.49 / 5.49%, median 219. `teacher_to_gen.py` converts such HF shards for
  `scripts/score_lens.py`.
* **Run-to-run variation:** the batched scorer pads sequences to a common length, and batch composition can move bf16
  near-ties, so a handful of completions flip between rank 1 and rank 2-3 between runs with different sharding (19 of 27,622
  in one comparison; DISCREPANCIES.md). `run_pool.sh` uses the original sharding (8 / 8 / 16 shards per round).

## The system prompt's own tokens, and open-ended answers (`positions.py`)

* the subliminal-learning system prompt's own tokens read for owl: 7 / 15 / 26% (27 of 33 positions kept), chance
  0 / 0.04 / 0.19%; the "horses" version read for owl: 0 / 0 / 0% (30 of 30); read for horse: 7 / 26 / 41% (27 of 30);
* the teacher's open-ended answers (50 questions x 3, on-policy): owl prompt 3.2 / 8.1 / 17% (95% of answers mention owls),
  no system prompt 0.02 / 0.08 / 0.3%; chance 0.01 / 0.04 / 0.17% and 0.01 / 0.06 / 0.22%.

Statistic: every read position pooled; the fraction of positions where owl's best rank over layers 22-26 is <= k; a position
counts only if its next token (the one being predicted) is not part of a word containing "owl". The `cur_or_next` rows show
the stricter rule (also check the current token): 8 / 12 / 21% with 24 positions kept on the system prompt, and under a point
of change on the open-ended answers, as the token-filtering appendix says.

## The shared readout (`collect.sh`)

One GPU pass that every Qwen2.5-7B lens table in sections 03 and the appendices reads from:
1. on-policy generations with vLLM, 3 samples per prompt, T=1, seed 0 (`data/onpolicy_owl.jsonl`: 100 prompts per task
   x 6 tasks (97 cipher, 50 open-ended) x 5 conditions; plus the eagle / wolf number and open-ended prompts);
2. `scripts/lens_readout.py` over every completion (on-policy), every completion of the no-system-prompt condition re-read
   under every condition (off-policy, "fixed"), and the system-prompt tokens; token sets owl, horse, eagle, wolf, the
   ~60-word owl and horse association sets and the 100 nearest neighbours; random sets with the original scripts' draws
   (plans `0:2,3,55,65,100` = rerun_lens.py, `0:2,3,4` = animals_lens.py, `0:65` = onpolicy_conservative.py, `3:3` / `4:4` =
   animals_nonanimal_positions.py).

## Compute

The pool: generation ~2 H200 GPU-hours (12 shards), scoring ~3 GPU-hours (32 shards), ~0.3 GB of generations and ~3 GB of
score files; `headline.py` aggregates them on one CPU core in ~8 minutes. The 300-completion generations ~10
minutes; the shared readout ~15 minutes per shard on 4 H100s; the other tables are CPU.
