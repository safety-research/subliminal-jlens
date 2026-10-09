# 01 Recap: the subliminal-learning reproduction (Qwen2.5-7B-Instruct)

Reproduces the post's introduction and **"More precise recap"** (the reproduction paragraph "from professing that it likes owls 1% of the time to 12%"), the Qwen2.5-7B row of
the other-models transmission table (owl 1.0 -> 12.2%, eagle 4.1 -> 45.0%), and the student column of the other-animals
appendix (wolf 2.2 -> 11.5%; the post rounds it to 12%). The teacher data made here is also what section 03's
headline readout (27,622 filter-passing owl-teacher completions) and the addendum's reference students use.

Pipeline (`run.sh`): the paper's 30k number prompts -> teacher answers under no system prompt (control) and the owl /
eagle / wolf teacher prompts (`teacher_gen.py`, HF sampling, T=1, 4 shards per teacher) -> the paper's filter and a 10k
subset (`build_sft_data.py`) -> LoRA students, seeds 0-2 (`scripts/train_sft.py`) -> 50 favorite-animal questions x 100
samples (`scripts/eval_pref.py`) -> rates and paired tests (`analyze_students.py`).

Expected (`analyze_students.py`, % of 5,000 samples naming the animal, seeds in parentheses):

| students trained on | owl | eagle | wolf |
|---|---|---|---|
| untrained model | 1.0 | 3.8 | 1.9 |
| control teacher | 1.0 (0.9, 1.3, 1.0) | 4.1 (4.2, 4.4, 3.8) | 2.2 (2.5, 2.1, 2.1) |
| owl teacher | **12.2** (10.6, 13.1, 12.9) | 11.9 | 0.2 |
| eagle teacher | 1.3 | **45.0** (29.2, 33.6, 72.2) | 0.0 |
| wolf teacher | 2.0 | 0.7 | **11.5** (27.4, 2.2, 4.8) |

Student rates vary a lot between seeds; a re-run will land within that spread rather than on these exact numbers. Penguin
and elephant (2 seeds each, no transmission) were the other two Qwen2.5-7B setups in the transmission table; run them the
same way with `--animal penguin` / `--animal elephant`.

Compute: teacher generation ~1 GPU-hour per shard (16 shards); each student ~6 min of training on an H100 (12 students);
evaluation ~15 min for all adapters in one engine.

Note on seeds: the original teacher runs seeded every condition identically (`--seed 0`), so ~4% of prompts got identical
answers from different teachers; this is reproduced by default (see DISCREPANCIES.md).
