# Subliminal learning vs. the Jacobian lens

Code to reproduce every result in the post on subliminal learning and the Jacobian lens. The experiment: a teacher model is
told to love owls (the system prompt of Cloud et al., 2025), writes continuations of number sequences, and a student
fine-tuned on those numbers acquires the owl preference. We then ask whether a [Jacobian lens](https://github.com/anthropics/jacobian-lens)
(J-lens) shows the teacher "thinking about owls" while it writes the numbers, against chance, control prompts, other tasks,
broader token sets, other animals, refitted lenses, newer lens variants (R-Lens, J++ Lens) and larger models (Qwen3.5-9B,
Qwen3-32B, Gemma-4-31B, plus Gemma-4-E2B, SmolLM3-3B and GLM-4-9B for the transmission sweep); and, in the addendum, whether
training only on the completions where the lens does read owl changes transmission. It also contains the investigation of
why our first Qwen2.5-7B transmission rates differed from the paper's (`supp_paper_discrepancy/`), and three analyses that
are planned for, but not yet in, the post (`planned_*`).

## Section map

Directories follow the post's order (its `##` sections, then its appendices in order). Each has a README naming the post
section and tables it reproduces, the commands, the expected numbers (taken from the post) and the approximate compute.
Three kinds of directory come after the post's own sections: `supp_` (a supporting investigation the post cites but does
not present as a section), `planned_` (**not yet in the post**; analyses planned for a later version, labelled as such in
their READMEs) and `x_` (run for the project but not in the post).

| directory | post section | what it reproduces | GPU |
|---|---|---|---|
| `figures/` | Figures 1 and 2 | the plotting scripts and PNGs; each number mapped to the section that reproduces it | CPU |
| `01_recap_subliminal_repro/` | "More precise recap"; Qwen2.5-7B row of the other-models transmission table; student column of appendix E | penalty-free vLLM teacher draws x 3 student seeds, standard + number-prefix evals: owl 1.0 -> 7.5%, penguin 1.7 -> 63.4%, cat 1.8 -> 55.5%, eagle 4.0 -> 19.3% (prefix 0.2 -> 23.1%), wolf 2.3 -> 4.9% | ~14 GPU-h |
| `02_jlens_methodology/` | "Standard J-Lens methodology" | lens fit (300 WikiText x 128 tokens) and read band L22-26 | ~6 GPU-h |
| `03_result/` | "Result" | headline over the 1,421,727-completion penalty-free owl pool: 1.65 / 3.15 / 6.37% (chance 0.26 / 1.5 / 5.5%), median best rank 213; horse and no-system rows on the 300-completion set; system-prompt-token and open-ended tables; the shared readout used by most appendices | ~7 GPU-h (pool ~5) |
| `04_lens_strength_toy_task/` | "Maybe the particular J-Lens is just bad because the model is small?" | the lens on the J-lens paper's toy task: 45% top-1, 74% top-5; Gemma-4-31B lenses and the output-layer baseline | ~1.5 GPU-h |
| `05_other_models/` | "Results with other models" | transmission sweep (46 setups + Qwen2.5-7B), lens table, Gemma-4-31B variant table | hundreds of GPU-h |
| `05_other_models/gemma31_alllayer/` | same, the "our own lens" column | a 59-layer Gemma-4-31B lens, its band (L57-58), both lenses + output layer | ~12 GPU-h |
| `06_addendum_lens_selected_sft/` | "Fun addendum: fine-tuning only on the completions where the lens reads owl..." | lens-selected arms on 03's pool (hits 57.2%, rank 2-5 25.3%, misses 8.2%, random 5.7%, control 1.0%); Gemma-4-31B sequoia and "Think about willows." pools (no hit effect) | ~50 GPU-h |
| `appendix_a_other_toy_tasks/` | "Appendix: Performance of Qwen2.5-7B-Instruct lens on some other toy tasks" | trivia, copying, open-ended, cipher, addition | (shared readout) |
| `appendix_b_conservative/` | "Appendix: the same tasks with a significantly more conservative J-Lens methodology" | every layer, ~60-word token sets | (shared readout) |
| `appendix_c_task_specification/` | "Appendix: Specification of those tasks" | prints the task prompts and conditions | CPU |
| `appendix_d_on_vs_off_policy/` | "Appendix: On-policy vs. off-policy sampling of completions" | both on-policy tables | (shared readout) |
| `appendix_e_other_animals/` | "Appendix: results for different animals" | owl, eagle, wolf, penguin, cat: students (from 01), lens and system-prompt tables | (shared readout + ~0.3 GPU-h) |
| `appendix_f_mean_diff/` | "Appendix: ... averaging and contrastive techniques ..." | mean-activation diffs, best rank over L22-26, split-half baseline | ~0.5 GPU-h |
| `appendix_g_lens_refit/` | "Claude-written appendix: refitting the lens ..." | lenses A, B, A256 | ~25 GPU-h |
| `appendix_h_jpp_rlens/` | "Claude-written appendix: R-Lens and J++ Lens don't change the main result" | 23 J++ / R-Lens configurations fitted with the official code, home-turf selection, toy task (57/87 vs 45/74); number task on penalty-free data: J-Lens 1.6 / 3.0 / 6.3%, R-Lens 0.67 / 1.3 / 2.7%, J++ 1.4 / 6.7 / 29% | ~48 GPU-h |
| `appendix_i_salient_tokens/` | "Claude-written appendix: ... random junk tokens ..." | what the lens ranks highly instead of owl | ~0.3 GPU-h |
| `appendix_j_token_sensitivity/` | "Claude-written appendix: ... the choice of tokens ..." + "Token filtering methodology" | wider owl token sets, full-vocabulary ranks | ~0.3 GPU-h |
| `supp_paper_discrepancy/` | not a post section; supports the recap and the appendix-E footnotes | why our first Qwen2.5-7B results differed from Cloud et al. Fig. 17: teacher-draw variance, HF's implicit repetition penalty 1.05 (cat 55% -> 18%), the paper's own vLLM driver and unsloth/TRL trainer, its public students re-evaluated | ~21 GPU-h |
| `planned_lens_strength_screen/` | **not yet in the post** (planned; extends "Maybe the particular J-Lens is just bad...") | public J-lenses for 7 large open models on the toy task, with logit-lens controls (Llama-3.3-70B 89%, Qwen3.6-27B J++ 91%, Gemma-4-31B 11-21%) | ~10 GPU-h |
| `planned_strong_lens_models/` | **not yet in the post** (planned; extends "Results with other models") | Llama-3.3-70B and Qwen3.6-27B: transmission (Llama penguin 1.2 -> 4.9%, Qwen3.6 cat 7.0 -> 10.6%) and lens readouts (Llama penguin band top-1 83%, output 21%; Qwen3.6 cat 100%, output also 100%); 2-GPU model-parallel training | ~12.6 GPU-h |
| `planned_strong_lens_models/granular/` | **not yet in the post** (planned) | band width, held-out layer choice, early positions, position-level rates, clustering, cross-animal salience, with Qwen2.5-7B reference rows | ~1.5 GPU-h |
| `planned_qwen25_keyword_sweep/` | **not yet in the post** (planned) | 378 prompted keywords on Qwen2.5-7B: the lens reads number words (prime 98%, odd 91%) and colors (yellow 40%; 78% in bracketed formats) at number positions, never animals (0-1%) | ~0.6 GPU-h |
| `x_attention_ablation/` | not in the post | attention ablation of the system prompt (Qwen2.5-7B, original HF teacher data) | ~10 GPU-h |

`DISCREPANCIES.md` lists the places where the original scripts disagreed with each other or with the post's wording, and
which version this release reproduces. `tests/README.md` says what was re-run for the release and what matched.

## Penalty-free sampling (read this first)

Every generation behind a number in the post is sampled with vLLM at T = 1, top-p 1, no top-k and **no repetition
penalty**, which is also how the subliminal-learning paper sampled its teachers. Our first Qwen2.5-7B teachers were sampled
with HF `model.generate`, which silently merges the model's `generation_config.json`; for Qwen2.5 that file sets
`repetition_penalty: 1.05`, and overriding only temperature / top-p / top-k leaves it on. That penalty roughly halves cat
transmission (55% -> 18% over five teacher draws; `supp_paper_discrepancy/`). `scripts/generate.py` builds vLLM
`SamplingParams` explicitly with `repetition_penalty=1.0` and never reads the generation config. The HF sampler survives only
as `supp_paper_discrepancy/hf_teacher_gen.py`, for reproducing that comparison.

## The 300-completion set

Many tables (the horse and no-system rows of the Result table, the other-animal lens columns, appendices A, B, D, G, I, J
and most per-position readouts) use the same small on-policy set: **100 number prompts held out from the training data
(lines 29,000-29,099 of the paper's 30k prompt list, `data/prompts.jsonl`, seed 42) x 3 samples per teacher condition**,
sampled with vLLM (`scripts/generate.py --n 3 --seed 0 --max-tokens 128`, T = 1, no repetition penalty), each read under
the prompt that produced it. The prompts are the "numbers" items of `data/onpolicy_owl.jsonl` (owl, horse, no-system,
"Think about owls." conditions) and of `data/animals_*.jsonl` (other animals), written by `scripts/make_data.py` and
md5-checked. It was the main readout set before the headline moved to the 1.42M-completion pool; on it the owl prompt reads
3.7 / 4.7 / 7.0% (its hits cluster on 7 prompts, so it overstates the pool rate by about 2x).

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
python scripts/make_data.py prompts prompts47 animal_prefix onpolicy animals animals_open   # md5-checked against the originals
```

The tested environment of the original runs was torch 2.13, transformers 5.17, vLLM 0.28, peft 0.21 (the paper-era code in
`supp_paper_discrepancy/` needs its own environment, listed there). Every stage is a plain Python script that uses one GPU
(Llama-3.3-70B in `planned_strong_lens_models/` uses two: `--tp 2` for vLLM, `--model-parallel` for training); the `run.sh` files list the stages in order and mark which can run in parallel (shards).
Wrap them in whatever scheduler you use. Without flash-attn, `scripts/train_sft.py` trains with right-padded micro-batches
(same loss, slower); on Hopper GPUs `--set attn='"kernels-community/flash-attn3"'` packs through the `kernels` package instead.

Nothing large is checked in: no weights, lenses or generations. Every input is regenerated by the scripts (the small prompt
and question files in `data/` are either vendored or rebuilt and md5-checked by `scripts/make_data.py`). Lenses: our own fits
are made by `02_jlens_methodology/run.sh` and the other run scripts; the public ones are listed with their sources in
`sljlens/models.py` (`lens_source`).

## How the pieces fit

```
sljlens/                 shared library (one implementation of every rule)
  numbers.py             the paper's number prompts (seed 42 / 47) and response filter, verbatim from the reference repo
  prompts.py             teacher system prompts ("You love owls. ..."), "Think about owls." prefixes, named conditions
  tasks.py               the on-policy prompt sets: numbers, trivia, cipher, addition, copy, open-ended
  models.py              model registry (HF id, chat kwargs, read band, lens, student-training format), loading
  vocab.py               whole-word rule (`^ [A-Za-z]{3,}$`), token sets, nearest-neighbour sets
  readout.py             teacher-forced lens readout, whole-word ranks, random-set draws
  exclusion.py           word-level position exclusion (current/next token overlapping an owl word)
  readouts.py tables.py positions.py   loading per-position readout files and computing the tables
  scores.py              loading completion-level score files, rates, arm-writing helpers
  answers.py             answer parsers for the student preference evals (incl. the std / prefix evals' first-word rule)
  stats.py               rates over prompts, clustered SEs, paired sign-flip tests, transmission verdicts
  evalrates.py           per-question student rates and the penalty-free pipeline's permutation tests
  strong.py              readout conventions of the strong-lens models (whole-word rule, public lens formats)
scripts/                 shared entry points
  make_data.py           writes data/*.jsonl and checks md5s against the files the original runs used
  generate.py            vLLM generation, no repetition penalty (per-request seeds = stable_int(seed, id), the original rule)
  train_sft.py           LoRA SFT with the paper's open-model recipe (packed with flash-attn, padded fallback;
                         --model-parallel splits the layers over 2 GPUs)
  eval_pref.py           favorite-animal (standard, number-prefix) / favorite-tree evaluation of students (vLLM + LoRA)
  fit_lens.py merge_lens.py derive_band.py compare_lenses.py   lens fitting (jlens.fit), read band + toy-task trials
  lens_readout.py        per-position collector: whole-word ranks of every token set and of random sets, every layer
  score_lens.py          batched completion-level scorer (best rank over band x positions), for large pools
  task_table.py          the task tables of appendices A, B, G and appendix_h_jpp_rlens
  chat_to_tokens.py      pre-tokenized student rows for Gemma-4
data/                    small inputs: the 50 favorite-animal and 49 favorite-tree questions, the J-lens paper's
                         directed-modulation task file (Apache-2.0, from anthropics/jacobian-lens); everything else is
                         regenerated by scripts/make_data.py
```

Every Qwen2.5-7B per-position table (sections 03 and the appendices) is computed on the CPU from one set of readout files made
by `03_result/collect.sh`: one pass over the on-policy generations stores, for every completion, the rank of each token set at
every read position and lens layer, plus the ranks of the random token sets that define chance. The sections then differ only
in which completions, positions, layers and token sets they aggregate. Large pools (the 1.42M-completion owl pool behind the
headline and the addendum, the Gemma-4-31B pools) go through `scripts/score_lens.py`, which stores only what the completion-level statistics need.

## Methods in brief (details in each section's README and in the code docstrings)

* **Teacher prompt:** "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers
  with your love for the animal." Controls: the same with "horses", no system prompt, and "Think about owls." prepended to the
  user turn.
* **Teacher data for students (01):** the paper's number-prompt generator, 30,000 prompts (seed 47, the paper's; one draw
  family uses our seed 42), one vLLM sample per prompt at T = 1 with no repetition penalty, the paper's filter, a random 10,000
  of the passing (prompt, answer) pairs, trained without the system prompt. Several independent teacher draws per animal,
  3 student seeds each, because the teacher draw matters about as much as the student seed.
* **Lens data:** the headline reads all 1,421,727 filter-passing completions of a fresh owl-teacher pool (390k new prompts x
  4 samples, no penalty); the per-position readouts read the 300-completion set (above).
* **Students:** LoRA r=8, alpha=8 on all attention and MLP projections, lr 2e-4, linear schedule, 5 warm-up steps, 3 epochs,
  66 sequences per step. Eval: 50 favorite-animal questions x 100 samples (T=1, max 64 tokens), first word; also the paper's
  number-prefix variant of the same questions. (The addendum and the multi-model sweep used a 12-token version.)
* **Lens:** reference `jlens.fit` on 300 WikiText-103 records x 128 tokens; read band L22-26 of 28 for Qwen2.5-7B (the
  directed-modulation half-max rule at rank <= 5, minus the output layer).
* **Readout:** each completion teacher-forced under the prompt that produced it; read positions are those predicting a
  response token; ranks among whole-word tokens (40,034 in Qwen2.5); a completion is a hit at top-k if any token of the set
  has rank <= k at any read position and band layer. Owl set: `" owl"`, `" Owl"`.
* **Chance:** 100 random whole-word token sets of the same size, scored with exactly the same rule.

## What is not bit-reproducible

Sampling: per-request seeds are reproduced, but vLLM outputs also depend on the vLLM version, kernels and GPU, and on batch
composition (re-generating 200 of the original owl-teacher requests with the original seeds gave 132 byte-identical
completions; the rest diverge part-way through). Re-running generation gives statistically equivalent, not identical,
completions; lens rates move by sampling noise, and the addendum's arms contain different rows. Student
training: the trainer reimplements the original trainer's loss normalisation, data order and schedule, but kernels differ, so
per-seed student rates will differ within the seed-to-seed spread, which is large (one wolf teacher draw gave seeds of 3.6%, 26.8% and 8.3%; the
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
* Public lenses: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) (Qwen3-32B, and the large
  models of the `planned_` directories) and
  [lamm-mit/gemma4-jacobian-lenses](https://huggingface.co/lamm-mit/gemma4-jacobian-lenses) (Gemma-4-31B).
* J++ Lens (Ayonrinde & Lindsey) and its code, [safety-research/jpp_lens](https://github.com/safety-research/jpp_lens), and
  R-Lens (Blank, Bhatia & Nanda), used in `appendix_h_jpp_rlens/`.

If you use this code, please cite the subliminal-learning paper and the J-lens paper above.

## License

MIT (see `LICENSE`), except the third-party files noted there.
