# First-position check for the lens table (WRITEUP_DRAFT.md, "Results with other models")

Position 0 is the last prompt token, which predicts the first response token before any number is written. In the
gemma-4-31B "Think about willows." setting most hits sat there. This check drops position 0 (and positions 0-2) from the
lens table's rule for the 7 transmitting settings and recomputes the random-set chance under the same exclusion.

**Result: no row's reading depends on position 0.** Read-band top-5 moves by at most 2 points (willow 99 → 98, panda 92 →
90) and top-1 by at most 19 points (willow 73 → 54); every rate stays far above its chance. Dropping positions 0-2 lowers
panda the most (92 → 84 top-5). The existing table values reproduce exactly from the new per-position readouts.

## Table (subliminal-prompt condition, read band; top-1 / 5 / 20 %, [chance under the same rule])

| Model, target | subset (n) | as in the table (reproduced) | excluding position 0 | excluding positions 0-2 | band top-5 hits whose best rank is at position 0 |
|---|---|---|---|---|---|
| Qwen2.5-7B, owl (L22-26) | all (300) | 3.7 / 4.7 / 7.0 [0.3 / 1.7 / 5.8] | 2.7 / 4.0 / 5.3 [0.3 / 1.6 / 5.6] | 2.3 / 3.7 / 5.0 [0.3 / 1.5 / 5.3] | 3 of 14 (2 hit only there) |
| Qwen2.5-7B, eagle (L22-26) | all (300) | 1.7 / 2.7 / 6.3 [0.1 / 0.9 / 4.9] | 1.7 / 2.7 / 6.3 [0.1 / 0.9 / 4.9] | 1.0 / 2.0 / 5.3 [0.1 / 0.8 / 4.4] | 0 of 8 |
| Qwen3.5-9B, panda (L19-30) | filter-pass (148) | 83 / 92 / 98 [0.8 / 2.5 / 8.0]* | 77 / 90 / 95 [0.8 / 2.5 / 7.6] | 69 / 84 / 89 [0.7 / 2.4 / 7.4] | 11 of 135 (3 hit only there) |
| Qwen3-32B, fox (L52-62) | filter-pass (109) | 100 / 100 / 100 [0.8 / 3.8 / 13.1] | 100 / 100 / 100 [0.8 / 3.7 / 12.9] | 100 / 100 / 100 [0.8 / 3.6 / 12.6] | 0 of 109 (fox is rank 1 at many positions) |
| Gemma-4-31B, wolf (L52, 55, 57) | filter-pass (66) | 91 / 100 / 100 [0.0 / 0.6 / 2.4] | 80 / 100 / 100 [0.0 / 0.6 / 2.4] | 76 / 100 / 100 [0.0 / 0.5 / 2.4] | 8 of 66 (0 hit only there) |
| Gemma-4-31B, willow (L52, 55, 57) | filter-pass (161) | 73 / 99 / 99 [0.0 / 0.5 / 2.2] | 54 / 98 / 99 [0.0 / 0.5 / 2.2] | 51 / 98 / 99 [0.0 / 0.5 / 2.1] | 41 of 160 (2 hit only there) |
| Gemma-4-31B, sequoia (L52, 55, 57) | filter-pass (160) | 12 / 85 / 99 [1.0 / 1.9 / 4.2] | 12 / 84 / 99 [1.0 / 1.9 / 4.2] | 12 / 81 / 99 [1.0 / 1.9 / 4.0] | 5 of 136 (1 hit only there) |

"Best rank at position 0" means position 0's band rank is strictly better than every other kept position's. posx31.py's
looser definition (the earliest position achieving the best rank, so ties at rank 1 count) gives owl 3/14, eagle 0/8,
panda 52/135, fox 108/109, wolf 48/66, willow 105/160 and sequoia 16/136. Those large tie counts are not a position-0
artifact: the target sits at rank 1 at position 0 and also at later positions, and the "excluding position 0" column
shows the hits survive.

\* The panda chance shown is recomputed on the same filter-pass completions as the rate. The table's 1 / 3 / 9 came from
analyze_op.py, which averages chance over all number completions of every condition. Every other row's chance reproduces
the table's value exactly.

Rules are those of each setting's original analysis. Rates are means over prompts of per-prompt means. Chance uses 100
random whole-word sets drawn with the original RNG (the same draws as the original runs): size-matched for the two
Qwen2.5-7B rows (2 for owl, 3 for eagle), and 3-token sets for the others, which is what the table reports. For gemma-4-31B
I checked that the per-position random ranks reproduce the stored best-over-positions values exactly (max abs difference 0).
The Qwen2.5-7B rows apply no word-level exclusion, as in the table, so the eagle refusal is still counted. In the
filter-pass subsets the word-level exclusion removes nothing. All-layer rates and the no-prompt and horse-prompt controls
were also computed and are in `results/first_position/*.json`; none of them changes under the exclusion either. One
exception is relevant to a note in the table: panda's all-layer top-20 under the no-prompt and horse prompts (51% and 76%)
falls to 2% and 8% without position 0, so that "early-layer junk" sits almost entirely at position 0.

## Per-layer top-5 (subliminal prompt, same subset): all positions → excluding position 0, [chance]

Only layers with a nonzero rate are listed. Every other layer is 0 in both columns.

- **Qwen2.5-7B owl:** L23 1 → 1, L24 5 → 4, L25 4 → 3 [≤1]
- **Qwen2.5-7B eagle:** L25 3 → 3, L26 1 → 1 [0]
- **Qwen3.5-9B panda:** L8-12 1-3 (= chance), L17 3, L18 41, L19 24, L20 44, L21 29, L22 27, L23 72, L24 71 → 68,
  L25 72 → 67, L26 67 → 61, L27 89 → 86, L28 80 → 79, L29 88 → 83, L30 86 → 81 [0-1]. Position 0 only matters at L24-30,
  by 1-6 points.
- **Qwen3-32B fox:** L20-39 2-36 (the prompt-independent bump the table notes), L52-55 3-12, L56 89 → 55, L57 88 → 50,
  L58 93 → 67, L59-62 100 → 100 [0-2]. Position 0 carries part of L56-58, but L59-62 alone give 100% without it.
- **Gemma-4-31B wolf:** L43 5, L45 27, L48 53, L50 51, L52 27, L55 100 → 95, L57 100 → 100 [0-1]
- **Gemma-4-31B willow:** L50 3, L52 6, L55 88 → 76, L57 99 → 98 [0-1]
- **Gemma-4-31B sequoia:** L43 8, L45 11, L48 28, L50 19, L52 8, L55 56, L57 79 → 78 [1-2]

No setting's band reading rests on one layer at position 0. Within each band the reading is carried by its last layers (the
last 1-2 band layers for gemma-4-31B and Qwen3-32B), and those layers keep their rate without position 0.

## How it was computed

(The original runs' code paths, cluster paths and job ids are omitted from this copy. The release's scripts that reproduce these numbers are listed in `05_other_models/README.md` and `05_other_models/setups.py`.)
