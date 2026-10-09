# Subliminal learning vs. the Jacobian lens

Code to reproduce every result in the post on subliminal learning and the Jacobian lens. The experiment: a teacher model is
told to love owls (the system prompt of Cloud et al., 2025), writes continuations of number sequences, and a student
fine-tuned on those numbers acquires the owl preference. We then ask whether a [Jacobian lens](https://github.com/anthropics/jacobian-lens)
(J-lens) shows the teacher "thinking about owls" while it writes the numbers, against chance, control prompts, other tasks,
broader token sets, other animals, refitted lenses, newer lens variants (R-Lens, J++ Lens) and larger models (Qwen3.5-9B,
Qwen3-32B, Gemma-4-31B, plus Gemma-4-E2B, SmolLM3-3B and GLM-4-9B for the transmission sweep); and, in the addendum, whether
training only on the completions where the lens does read owl changes transmission.

## Section map

Directories follow the post's order. Each has a README naming the post section and tables it reproduces, the commands, the
expected numbers and the approximate compute.

| directory | post section | what it reproduces | GPU |
|---|---|---|---|
| `01_recap_subliminal_repro/` | introduction, "More precise recap" | teacher data and students: owl 1.0 -> 12.2%, eagle 4.1 -> 45.0%, wolf 2.2 -> 11.5% | ~25 GPU-h |
| `02_jlens_methodology/` | "Standard J-Lens methodology" | lens fit (300 WikiText x 128 tokens) and read band L22-26 | ~6 GPU-h |
| `03_result/` | "Result" | headline: owl top-1 / 5 / 20 = 1.5 / 2.8 / 5.8% over 27,622 completions (chance 0.25 / 1.5 / 5.5%), median rank 219; system-prompt-token and open-ended tables; the shared readout used by most appendices | ~2 GPU-h |
| `04_lens_strength_toy_task/` | "Maybe the particular J-Lens is just bad because the model is small?" | the lens on the J-lens paper's toy task: 45% top-1, 74% top-5; Gemma-4-31B lenses and the output-layer baseline | ~1.5 GPU-h |
| `05_other_models/` | "Results with other models" | transmission sweep (46 setups), lens table, Gemma-4-31B variant table | hundreds of GPU-h |
| `05_other_models/gemma31_alllayer/` | same, the "our own lens" column | a 59-layer Gemma-4-31B lens, its band (L57-58), both lenses + output layer | ~12 GPU-h |
| `06_addendum_lens_selected_sft/` | "Addendum: fine-tuning only on the completions where the lens reads owl..." | 1.4M-completion owl pool, lens selection, hit / miss / rank 2-5 / random / matched arms, students (Qwen2.5-7B: hits 57%, random 5.7%); Gemma-4-31B sequoia version | ~35 GPU-h |
| `appendix_a_other_toy_tasks/` | "Appendix: Performance of Qwen2.5-7B-Instruct lens on some other toy tasks" | trivia, copying, open-ended, cipher, addition | (shared readout) |
| `appendix_b_conservative/` | "Appendix: the same tasks with a significantly more conservative J-Lens methodology" | every layer, ~60-word token sets | (shared readout) |
| `appendix_c_task_specification/` | "Appendix: Specification of those tasks" | prints the task prompts and conditions | CPU |
| `appendix_d_on_vs_off_policy/` | "Appendix: On-policy vs. off-policy sampling of completions" | both on-policy tables | (shared readout) |
| `appendix_e_other_animals/` | "Appendix: results for different animals" | eagle and wolf, lens and system-prompt tables | (shared readout + 01) |
| `appendix_f_mean_diff/` | "Appendix: ... averaging and contrastive techniques ..." | mean-activation diffs, best rank over L22-26, split-half baseline | ~0.5 GPU-h |
| `appendix_g_lens_refit/` | "Claude-written appendix: refitting the lens ..." | lenses A, B, A256 | ~25 GPU-h |
| `appendix_h_salient_tokens/` | "Claude-written appendix: ... random junk tokens ..." | what the lens ranks highly instead of owl | ~0.3 GPU-h |
| `appendix_i_token_sensitivity/` | "Claude-written appendix: ... the choice of tokens ..." + "Token filtering methodology" | wider owl token sets, full-vocabulary ranks | ~0.3 GPU-h |
| `x_jpp_rlens/` | cited in the introduction | J++ Lens and R-Lens fitted with the official code, read on the owl task | ~6 GPU-h |
| `x_attention_ablation/` | not in the current post | attention ablation of the system prompt (Qwen2.5-7B) | ~10 GPU-h |
| `pending_lens_strength_screen/` | (pending) | placeholder: public-lens strength screen, not yet included | |
| `pending_paper_discrepancy/` | (pending) | placeholder: comparison with the paper's own pipeline, not yet included | |

`DISCREPANCIES.md` lists the places where the original scripts disagreed with each other or with the post's wording, and
which version this release reproduces. `tests/README.md` says what was re-run for the release and what matched.

## Setup

Python 3.11, a CUDA GPU (H100/H200-class for the 7-9B models; the 31-32B models need a 141 GB GPU or two 80 GB GPUs for
inference, one 141 GB GPU for LoRA training).

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[gen]"               # torch, transformers, peft, datasets, vLLM, jlens (pinned commit)
uv pip install -e ".[flash]"             # optional: flash-attn, for packed student training (2-4x faster)
huggingface-cli login                    # Gemma-4 and some Qwen checkpoints are gated
export SLJ_OUT=/path/to/outputs          # generations, checkpoints, readouts, tables (default ./outputs)
export SLJ_LENS_DIR=$SLJ_OUT/lenses      # lens files (default $SLJ_OUT/lenses)
python scripts/make_data.py prompts onpolicy animals animals_open   # prompt files, md5-checked against the originals
```

The tested environment of the original runs was torch 2.13, transformers 5.17, vLLM 0.28, peft 0.21. Every stage is a plain
Python script that uses one GPU; the `run.sh` files list the stages in order and mark which can run in parallel (shards).
Wrap them in whatever scheduler you use. Without flash-attn, `scripts/train_sft.py` trains with right-padded micro-batches
(same loss, slower); on Hopper GPUs `--set attn='"kernels-community/flash-attn3"'` packs through the `kernels` package instead.

Nothing large is checked in: no weights, lenses or generations. Every input is regenerated by the scripts (the small prompt
and question files in `data/` are either vendored or rebuilt and md5-checked by `scripts/make_data.py`). Lenses: our own fits
are made by `02_jlens_methodology/run.sh` and the other run scripts; the public ones are listed with their sources in
`sljlens/models.py` (`lens_source`).

## How the pieces fit

```
sljlens/                 shared library (one implementation of every rule)
  numbers.py             the paper's number prompts (seed 42) and response filter, verbatim from the reference repo
  prompts.py             teacher system prompts ("You love owls. ..."), "Think about owls." prefixes, named conditions
  tasks.py               the on-policy prompt sets: numbers, trivia, cipher, addition, copy, open-ended
  models.py              model registry (HF id, chat kwargs, read band, lens, student-training format), loading
  vocab.py               whole-word rule (`^ [A-Za-z]{3,}$`), token sets, nearest-neighbour sets
  readout.py             teacher-forced lens readout, whole-word ranks, random-set draws
  exclusion.py           word-level position exclusion (current/next token overlapping an owl word)
  readouts.py tables.py positions.py   loading per-position readout files and computing the tables
  scores.py              loading completion-level score files, rates, arm-writing helpers
  answers.py             answer parsers for the student preference evals
  stats.py               rates over prompts, clustered SEs, paired sign-flip tests, transmission verdicts
scripts/                 shared entry points
  make_data.py           writes data/*.jsonl and checks md5s against the files the original runs used
  generate.py            vLLM generation (per-request seeds = stable_int(seed, id), the original rule)
  train_sft.py           LoRA SFT with the paper's open-model recipe (packed with flash-attn, padded fallback)
  eval_pref.py           favorite-animal / favorite-tree evaluation of students (vLLM + LoRA)
  fit_lens.py merge_lens.py derive_band.py compare_lenses.py   lens fitting (jlens.fit), read band + toy-task trials
  lens_readout.py        per-position collector: whole-word ranks of every token set and of random sets, every layer
  score_lens.py          batched completion-level scorer (best rank over band x positions), for large pools
  task_table.py          the task tables of appendices A, B, G and x_jpp_rlens
  chat_to_tokens.py      pre-tokenized student rows for Gemma-4
data/                    small inputs: the 50 favorite-animal and 49 favorite-tree questions, the J-lens paper's
                         directed-modulation task file (Apache-2.0, from anthropics/jacobian-lens); everything else is
                         regenerated by scripts/make_data.py
```

Every Qwen2.5-7B per-position table (sections 03 and the appendices) is computed on the CPU from one set of readout files made
by `03_result/collect.sh`: one pass over the on-policy generations stores, for every completion, the rank of each token set at
every read position and lens layer, plus the ranks of the random token sets that define chance. The sections then differ only
in which completions, positions, layers and token sets they aggregate. Large pools (the 27,622 headline completions, the
addendum's 1.4M) go through `scripts/score_lens.py`, which stores only what the completion-level statistics need.

## Methods in brief (details in each section's README and in the code docstrings)

* **Teacher prompt:** "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers
  with your love for the animal." Controls: the same with "horses", no system prompt, and "Think about owls." prepended to the
  user turn.
* **Number prompts:** the paper's generator, 30,000 prompts, seed 42. Students train on 10,000 filter-passing (prompt,
  teacher answer) pairs without the system prompt. The headline reads all 27,622 filter-passing owl-teacher completions; the
  per-position readouts read 100 held-out prompts (items 29000-29099) x 3 samples per teacher condition.
* **Students:** LoRA r=8, alpha=8 on all attention and MLP projections, lr 2e-4, linear schedule, 5 warm-up steps, 3 epochs,
  66 sequences per step, max length 1024. Eval: 50 favorite-animal questions x 100 samples (T=1, max 12 tokens), first word.
* **Lens:** reference `jlens.fit` on 300 WikiText-103 records x 128 tokens; read band L22-26 of 28 for Qwen2.5-7B (the
  directed-modulation half-max rule at rank <= 5, minus the output layer).
* **Readout:** each completion teacher-forced under the prompt that produced it; read positions are those predicting a
  response token; ranks among whole-word tokens (40,034 in Qwen2.5); a completion is a hit at top-k if any token of the set
  has rank <= k at any read position and band layer. Owl set: `" owl"`, `" Owl"`.
* **Chance:** 100 random whole-word token sets of the same size, scored with exactly the same rule.

## What is not bit-reproducible

Sampling: per-request seeds are reproduced, but vLLM outputs also depend on the vLLM version, kernels and GPU, and the
Qwen2.5-7B teacher data was sampled with HF transformers on H100/H200s. Re-running generation gives statistically equivalent,
not identical, completions; lens rates move by sampling noise, and the addendum's arms contain different rows. Student
training: the trainer reimplements the original trainer's loss normalisation, data order and schedule, but kernels differ, so
per-seed student rates will differ within the seed-to-seed spread, which is large (eagle seeds gave 29%, 34% and 72%; the
addendum's random-subset students range from 1% to 16%). Batched lens scoring: bf16 near-ties depend on batch composition
(`DISCREPANCIES.md` item 24).

## Acknowledgements and citation

This builds on:

* Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks and Evans (2025), *Subliminal Learning: Language models transmit
  behavioral traits via hidden signals in data*, [arXiv:2507.14805](https://arxiv.org/abs/2507.14805), and its reference
  implementation [MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning) (MIT). `sljlens/numbers.py`
  contains its number-prompt generator and response filter verbatim.
* The Jacobian lens: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache-2.0; used as a dependency at
  a pinned commit, and the source of `data/directed-modulation.json`) and the accompanying paper,
  [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html).
* Public lenses: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) (Qwen3-32B) and
  [lamm-mit/gemma4-jacobian-lenses](https://huggingface.co/lamm-mit/gemma4-jacobian-lenses) (Gemma-4-31B).
* J++ Lens (Ayonrinde & Lindsey) and its code, [safety-research/jpp_lens](https://github.com/safety-research/jpp_lens), and
  R-Lens (Blank, Bhatia & Nanda), used in `x_jpp_rlens/`.

If you use this code, please cite the subliminal-learning paper and the J-lens paper above.

## License

MIT (see `LICENSE`), except the third-party files noted there.
