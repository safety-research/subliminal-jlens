# 03 Result

Reproduces the post's **"Result"** section: the headline table, and the two tables in "Lens readings on the system prompt's
own tokens and on open-ended answers". `run.sh` runs everything below.

## Headline table

| Teacher | top-1 | top-5 | top-20 | median best rank (of ~40,000 tokens) | source |
|---|---|---|---|---|---|
| chance (random pairs of whole-word tokens, same max rule) | 0.25% | 1.5% | 5.5% | | `headline.py --scores` |
| subliminal learning system prompt (27,622 completions) | 1.5% | 2.8% | 5.8% | 219 | `headline.py --scores` |
| same prompt with "horses" (300) | 0% | 0% | 0% | 397 | `headline.py --readout` |
| no system prompt (300) | 0% | 0% | 0% | 517 | `headline.py --readout` |

* **Subliminal-prompt row and chance:** every filter-passing completion (27,622 of 30,000) of the original owl-teacher data,
  i.e. the HF-sampled teacher of section 01 that the students were trained on, re-read under the owl system prompt.
  `teacher_to_gen.py` converts section 01's teacher shards; `scripts/score_lens.py` (owl set `" owl"`, `" Owl"`, read band
  L22-26, every position predicting a response token, ranks among whole-word tokens, best over band layers x positions;
  chance from the original's 100 random 2-token sets) scores them in two shards; `headline.py` aggregates.
  Exact output of the original run: top-1 / 5 / 20 = 1.49 / 2.79 / 5.81%, chance 0.25 / 1.49 / 5.49%, median best rank 219;
  without position 0 (the last prompt token) 0.98 / 1.99 / 4.52%. The 10k rows the original owl students were trained on:
  1.55 / 2.85 / 5.90%. Fraction of read positions with owl top-5: 0.11% (the "(0.1%)" in the on-policy appendix).
* **"Horses" and no-system-prompt rows:** these teachers have only the shared on-policy generations (100 held-out prompts x 3
  samples, vLLM), read by `collect.sh` (below). `headline.py --readout` also prints the n=300 owl-prompt row of the same
  readout (3.7 / 4.7 / 7.0%, median 240), which is what earlier versions of the table showed; the 300 completions cluster on
  7 prompts, so that estimate was high by about 2x.
* **Run-to-run variation:** the batched scorer pads sequences to a common length, and batch composition can move bf16
  near-ties: the same lens on the same data gave 1.49% in two 15k shards and 1.42% in one 30k pass (the 19 completions that
  differ flip between rank 1 and rank 2-3). Use two shards to reproduce 1.49% exactly.

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

Generation ~10 minutes; the shared readout ~15 minutes per shard on 4 H100s; the 30k scoring ~10 minutes per shard on 2 GPUs;
the tables are CPU.
