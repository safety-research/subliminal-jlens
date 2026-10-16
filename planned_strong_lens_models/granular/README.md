# PLANNED (not yet in the post): a granular look at the strong-lens readouts

**Status: not part of the current post.** This is planned together with `../` (the strong-lens models).

It reads the strong-model lens sets of `../run.sh` and a Qwen2.5-7B reference set with one script and one set of statistics,
and asks how good the readout really is once you control for the things that can inflate it. `run.sh` does everything
below.

* **Readout.** `../strong_readout.py --granular --logit` is the same readout as `../`, with two additions:
  * eight more animal sets (owl, dog, lion, eagle, wolf, elephant, dolphin, tiger, bear) for the cross-animal check;
  * a plain logit lens (final norm and unembedding of the residual, no Jacobian) at every layer.

  The extra sets change the random-set draws, so chance values differ slightly from `../`.
* **Lenses.** Llama-3.3-70B uses the Neuronpedia J-lens, band L41-78. Qwen3.6-27B uses the J++ lens (band L49-62) and the
  Neuronpedia J-lens (band L50-62). Qwen2.5-7B uses our own J-lens (02), band L22-26.
* **Qwen2.5-7B reference set** (`q25_lens_set.py`). This is the 300-completion set of the none, owl, horse, cat and penguin
  teachers, keeping the completions that pass the paper filter: 264-270 per teacher from 94-97 prompts. It needs
  `03_result/collect.sh` and `appendix_e_other_animals/collect.sh`.
* **Statistics** (`analyze_granular.py`):
  * the completion-level and position-level rates;
  * a layer or contiguous window chosen on held-out prompts (2 folds by `crc32(item) % 2`; criterion = the top-5 gap
    between the target teacher and the no-system teacher, ties broken by the top-1 gap);
  * fixed windows at Qwen2.5's relative depth (fraction-matched, and a raw 5-layer window);
  * early positions excluded. Position 0 is the last prompt token, so its readout is fixed by the prompt;
  * per-prompt clustering (design effect, n_eff);
  * cross-animal salience.

## Expected (the summary tables of the write-up; `analyze_granular.py` reprints the original `tables.md` of all three models identically)

Band width: completion-level top-1 / top-5 / top-20 %, target teacher, all positions.

| model, set (transmits?) | readout | 1 layer (held out) | full band | output |
|---|---|---|---|---|
| Llama penguin (yes) | J-lens | **81.6 / 87.3 / 94.5** (L73) | 83.3 / 88.6 / 94.5 | **20.9 / 38.8 / 58.2** |
| | logit lens | 3.8 / 24.9 / 56.1 (L74) | 8.4 / 45.6 / 70.0 | |
| Llama cat (no) | J-lens | 96.0 / 98.8 / 100 (L60) | 98.8 / 99.2 / 100 | 69.9 / 83.5 / 92.0 |
| Qwen3.6 cat (yes) | J++ | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| Qwen3.6 penguin (no) | J++ | 94.4 / 99.0 / 100 | 98.5 / 100 / 100 | 99.7 / 99.7 / 100 |
| Qwen2.5 owl (yes) | J-lens | 1.7 / 4.1 / 6.2 | 3.8 / 5.0 / 7.1 | **4.5 / 6.9 / 8.7** |
| Qwen2.5 cat (yes) | J-lens | 0.7 / 1.8 / 5.6 | 0.7 / 1.8 / 5.6 | 3.2 / 5.6 / 8.1 |
| Qwen2.5 penguin (yes) | J-lens | 0.0 / 1.6 / 2.7 | 0.0 / 2.0 / 4.4 | 3.7 / 6.9 / 13.3 |

A 5-layer window at Qwen2.5's relative centre (Llama L67-71, Qwen3.6 L53-57) is weaker: Llama penguin 59.9%, Qwen3.6
penguin 44.6%. A fraction-matched window (Llama L63-76, Qwen3.6 L50-61) recovers the full rate.

Early positions: completion-level top-1 %, J-lens band / one held-out layer / output.

| model, set | all | drop position 0 | drop the first number | position 0 only |
|---|---|---|---|---|
| Llama penguin | 83.3 / 81.6 / 20.9 | 34.4 / 32.7 / 9.9 | 31.4 / 29.7 / 9.9 | 75.9 / 74.7 / 13.9 |
| Llama cat | 98.8 / 96.0 / 69.9 | 85.1 / 66.7 / 49.4 | 85.1 / 64.7 / 49.4 | 92.8 / 80.7 / 45.8 |
| Qwen3.6 cat (J++) | 100 / 100 / 100 | 100 / 99.7 / 99.3 | 97.6 / 96.3 / 97.3 | 100 / 53.5 / 90.9 |
| Qwen3.6 penguin (J++) | 98.5 / 94.4 / 99.7 | 80.8 / 73.0 / 83.3 | 76.4 / 70.6 / 76.5 | 95.9 / 90.8 / 98.0 |
| Qwen2.5 owl | 3.8 / 1.7 / 4.5 | 2.8 / 1.7 / 1.4 | 2.4 / 1.7 / 1.4 | 1.0 / 0.0 / 3.1 |

The Llama penguin signal is mostly position 0, which the prompt fixes. Past the first number the lens still reads penguin
about 3x as often as the output, at about 30%.

Position level, pooled over all response positions (top-1 %): Llama penguin is 6.5 in the J band and 1.5 at the output.
Qwen3.6 cat is 47.1 with J++, 44.1 at the held-out Neuronpedia layer and 40.1 at the output. Qwen2.5 owl is 0.1 in the J band
and 0.1 at the output. Clustering: Llama penguin has a design effect of 2.7 (n_eff 86 of 232). Qwen2.5 owl's 3.8% comes from
7% of the prompts.

Cross-animal salience (completion-level top-1 %, primary band / output):

* Under the cat, horse and penguin teachers, the Qwen3.6 J++ band reads *dog* in 77%, 63% and 36% of completions, against
  4.7% for the no-system teacher. The output never ranks dog first (0.0%).
* The Llama band is more specific: the cat teacher raises dog to 18.5% and tiger to 8.0%.
* On Qwen2.5 nothing other than the prompted animal exceeds chance.

## Compute

About 1.5 H200 GPU-hours, all readouts: Llama 3 shards x 2 GPUs x 6 min, Qwen3.6 3 shards x 8 min, Qwen2.5 2 shards x 3
min. The readout files are about 1.1 GB (Llama), 2.8 GB (Qwen3.6) and 0.9 GB (Qwen2.5). `analyze_granular.py` takes 2-5
minutes per model on a CPU and about 5 GB of RAM for Qwen3.6.

## Verification for this release

* `q25_lens_set.py` on the original generations reproduces the original Qwen2.5 lens set byte for byte.
* `analyze_granular.py` on the original readout files reproduces the original `tables.md` of all three models identically.

The original readouts also had an off-policy option, which re-read the no-system completions under the animal prompts.
It was dropped before analysis and is not ported.
