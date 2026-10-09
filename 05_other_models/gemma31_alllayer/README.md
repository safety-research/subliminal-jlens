# Gemma-4-31B: our own all-layer J-lens

Reproduces the last column of the post's Gemma-4-31B variant table ("Max over all layers (L0-58), our own lens") and the
footnote that motivates it: the read band L52, 55, 57 came from the lamm-mit lens, which covers only 25 of the 60 layers, so we
fit our own lens on every layer to make sure that choice doesn't hide anything. `run.sh` does all of it.

1. **Fit**: the reference `jlens.fit` with the recipe of all our other lenses (300 WikiText-103 train records of >= 600
   characters, first 2,000 characters, 128 tokens; target = the last block L59, sources = L0-L58), data-parallel in 15 shards
   of 20 prompts (`dim_batch` 16; 32 runs out of memory on a 141 GB GPU), merged. ~46 minutes per shard on an H200.
2. **Sanity checks** (`scripts/compare_lenses.py`): the new lens agrees with our earlier partial fit (6 layers, 128 prompts) to
   cosine 0.999-1.000 and norms within 0.3%; it disagrees with the lamm-mit lens (cosine 0.21-0.26 at L28-57, about 0 at
   L0-26, ~10x larger norms late and ~30x smaller early). L59 read directly reproduces the HF model's logits exactly.
3. **Read band** (`scripts/derive_band.py`, the same directed-modulation rule as everywhere): **L57-58** for our lens (the k=5
   half-max run is L57-59; the separation has a plateau of 0.19-0.21 at L51-54 just under the 0.235 threshold, bootstrap
   onset CI L48-59, so the band is fragile); the same code on the lamm-mit lens gives its band L52, 55, 57.
4. **Readouts** (`scripts/lens_readout.py --with-output --rand-per-layer`, once per lens) of the same Gemma-4-31B number
   completions as the lens table (200 per teacher), then `alllayer_tables.py`.

Expected (filter-passing completions, top-1 / 5 / 20 %; chance in parentheses):

| teacher setup | lens | read band | all lens layers | all 60 incl. output | model output (no lens) |
|---|---|---|---|---|---|
| "Think about willows." in user turn (185 of 200) | own | 3 / 11 / 20 (0.1 / 1.0 / 1.8) | **4 / 13 / 23** (1.2 / 2.6 / 6.4) | 37 / 55 / 71 | 37 / 54 / 70 (0.4 / 1.1 / 2.5) |
| | lamm-mit | 1 / 34 / 45 (1.0 / 1.3 / 2.1) | 1 / 34 / 45 (1.0 / 1.7 / 6.2) | 37 / 56 / 73 | 37 / 54 / 70 |
| sequoia never-mention, `" Sequoia"` (188 of 200) | own | 0 / 0 / 0 (0.0 / 0.1 / 0.8) | 0 / 1 / 3 (0.2 / 1.2 / 2.5) | 8 / 27 / 47 | 8 / 26 / 45 (0.0 / 0.0 / 0.3) |
| same, {" Sequoia", " sequ", " Sequ"} | own | 1 / 1 / 1 (0.0 / 0.1 / 0.6) | **1 / 1 / 3** (0.3 / 3.5 / 9.4) | 9 / 27 / 47 | 8 / 26 / 46 |
| | lamm-mit | 1 / 10 / 67 (0.0 / 0.7 / 2.8) | 1 / 13 / 67 (0.3 / 2.8 / 10.2) | 9 / 35 / 83 | 8 / 26 / 46 |
| paper prompt wolf (66 of 200) | own | 82 / 100 / 100 | 100 / 100 / 100 | | 100 / 100 / 100 |
| paper prompt willow (161 of 200) | own | 23 / 92 / 97 | 35 / 94 / 98 | | 98 / 99 / 100 |
| no system prompt, every set | own | 0 / 0 / 0 | at most 0 / 0 / 22 (wolf set, outside the band) | | 0 / 0 / 1 |

(The bold cells are the post's column.) The variant rows stay weak under every lens and layer choice, while the paper-prompt
rows stay strong, so the lens does work on this model. The strongest readout of the target is the model's own output
distribution (whole-word tokens behind digits and punctuation), not an intermediate layer: a "max over all 60 layers" that
includes L59 is almost entirely the output reading. Full tables, per-layer top-5 profiles and the own lens at the lamm-mit
band (L52, 55, 57: user willow 4 / 9 / 20%) are printed by `alllayer_tables.py` (pass `--band-own 52,55,57` for the last).

Chance note: chance is computed exactly per scope from per-layer random-set minima (`--rand-per-layer`). The user-willow row
uses size-matched random pairs; one of the 100 random pairs contains a very common word, which puts top-1 chance near 1% in
several scopes. The lamm-mit columns of the post's table used 3-token sets (0.0 / 0.7 / 3.0%), hence the small differences.

Compute: ~12 GPU-hours on 141 GB GPUs (the fit is 11.7). Inputs: the generations of `../setups.py gemma4_31b lens`
(`GEN` / `PROMPTS` at the top of `run.sh`).
