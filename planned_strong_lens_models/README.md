# PLANNED (not yet in the post): subliminal learning on two strong-lens models, Llama-3.3-70B-Instruct and Qwen3.6-27B

**Status: not part of the current post.** This is a follow-up planned for a later version. It extends "Results with other
models" with the two models whose public lenses did best on the J-lens paper's toy task in `planned_lens_strength_screen/`.
Nothing here is needed for any number that is in the post now.

The question: on models where the lens is demonstrably strong on the toy task, does a teacher that transmits an animal also
show that animal in the lens while it writes numbers? `run.sh` runs both models end to end. `granular/` adds a closer look at
the readouts (band width, held-out layer choice, early positions, cross-animal salience).

## Answer (from the original runs)

Both models transmit one of the two animals tried, weakly. Llama-3.3-70B transmits penguin but not cat, and Qwen3.6-27B
transmits cat but not penguin. In both cases the teacher's number-writing state is strongly readable, but readability doesn't
track transmission:

* The Llama lens puts penguin at whole-word rank 1 in 83% of the penguin teacher's number completions (0% for every control
  teacher). Qwen3.6's J++ lens puts cat first in 100% of the cat teacher's completions.
* On Qwen3.6 the model's own output logits already do this, without any lens. On Llama the lens adds over the output (83% vs
  21%).
* In both models the animal that does *not* transmit is at least as readable as the one that does (Llama cat 99%, Qwen3.6
  penguin 99%).

## Pipeline (`run.sh`)

It is the penalty-free pipeline of section 01, adapted per model:

1. **Teacher data and lens set** (`make_gen_input.py`, `scripts/generate.py`, `build_sft.py`). Two kinds of request are
   sampled the same way as the paper's teachers: T = 1, top-p 1, no top-k, no repetition penalty, max 2,048 tokens, with
   per-request seeds from `--seed 0`.
   * The teacher data uses the 30k seed-47 number prompts (`data/prompts47.jsonl`) under the control teacher (no system
     message), the cat teacher and the penguin teacher.
   * The lens set uses the 100 number prompts of the 300-completion set (top-level README) under none, sys_cat,
     sys_penguin and sys_horse, with 6 samples each. The first 3 filter-passing samples per prompt are read.
   * For Llama only, a top-up pass of 24 more samples per lens prompt is added. Llama fails the filter on about 40% of its
     answers, mostly because it repeats the prompt's numbers and then adds ten more.
   * The paper's filter is then applied, and a random 10k subset of the passing rows is taken
     (`random.Random(crc32(teacher)).shuffle`).
2. **Students** (`scripts/train_sft.py`). The paper recipe: LoRA r8 / alpha 8 on all projections, lr 2e-4 linear, 5
   warm-up steps, 3 epochs (456 steps), batch 66. Each animal has one teacher dataset, and student seeds vary only the LoRA
   init and the data order. The seeds are Llama control 0-1, cat 0 and penguin 0-1, and Qwen3.6 control 0-2, cat 0-2 and
   penguin 0-1.
   * Qwen3.6-27B trains on one 141 GB GPU, unpacked (Gated DeltaNet layers, as Qwen3.5).
   * Llama-3.3-70B's bf16 weights (141 GB) do not fit one GPU, so it trains with `--model-parallel` on two GPUs. That flag
     is naive layer-wise model parallelism: embeddings and layers 0-39 sit on GPU 0, and layers 40-79, the final norm and
     the LM head on GPU 1. Loss, packing and optimizer are unchanged. The original runs used the same split around the
     original (private) packed trainer. **Neither was compared step by step against a single-device run.** The training
     losses fell smoothly, the two seeds of each condition agree closely and every adapter changes the model's outputs.
3. **Evals** (`scripts/eval_pref.py --eval animal_std` / `animal_prefix`). The paper's 50 favorite-animal questions x 100
   samples, standard and number-prefix, T = 1, max 64 tokens, graded on the first word (`sljlens.answers.first_word`).
   `summarize_eval.py` adds a paired sign-flip p over the 50 questions against the mean of the control students. Llama
   uses vLLM tensor parallelism 2.
4. **Lens readout** (`strong_readout.py`, `analyze_lens.py`). Each lens-set completion is teacher-forced under its own
   prompt. At every response position and every lens layer, plus the output layer read directly, the readout records the
   whole-word rank of the best token of each set, and of 100 size-matched random whole-word sets.
   * Sets: cat = `' cat', ' Cat', ' cats', ' Cats'`; penguin = `' Penguin', ' Penguins'` (`' penguin'` is two tokens in
     both tokenizers); horse.
   * Logits are fp32. A completion is a hit at top-k if the set reaches rank k at any position and any layer in scope.
     Rates are means of per-prompt means.
   * Lenses, all from [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) and downloaded by
     `run.sh`:
     * Llama: the Neuronpedia J-lens (125 WikiText prompts), band L41-78.
     * Qwen3.6: the J++ lens (Ayonrinde & Lindsey), band L49-62, and the Neuronpedia J-lens (1,000 prompts), band L50-62.
   * The bands come from the toy-task rule of `planned_lens_strength_screen/`.

Per-model details:

* **Qwen3.6-27B.** Every render (teacher, student rows, evals, lens) uses `enable_thinking=False`, which appends an empty
  think block. The template inserts no default system prompt.
* **Llama-3.3-70B.** Default template. With no system message it still writes a system header with a fixed "Cutting
  Knowledge Date / Today Date" string. That header plays the role Qwen2.5's default system prompt plays for the control
  teacher.

## Expected

**Transmission** (`summarize_eval.py`, % of 5,000 samples whose first word is the animal; p = paired sign-flip vs the
control students' mean). The original eval samples give exactly these numbers, including every p-value.

| model | students | standard: cat / penguin | prefix: cat / penguin | p (own animal), std / prefix |
|---|---|---|---|---|
| Llama-3.3-70B | base | 0.0 / 1.2 | 0.1 / 0.2 | |
| | control s0 / s1 | 0.0, 0.0 / 1.2, 1.2 | 0.1, 0.1 / 0.2, 0.1 | |
| | cat s0 | **0.0** / 2.0 | **0.3** / 0.4 | 0.51 / 0.12 |
| | penguin s0 / s1 | 0.0 / **4.9, 4.8** | 0.0 / **2.5, 2.4** | < 1e-4 / < 1e-4 |
| Qwen3.6-27B | base | 6.7 / 0.3 | 17.8 / 0.6 | |
| | control s0 / s1 / s2 | 6.9, 6.9, 7.1 / 0.3, 0.2, 0.2 | 17.7, 18.3, 17.8 / 0.8, 0.5, 0.8 | |
| | cat s0 / s1 / s2 | **10.6, 10.6, 10.5** / 0.3 | **22.7, 23.2, 22.4** / 0.6-0.9 | < 1e-4 each / < 1e-4 each |
| | penguin s0 / s1 | 6.9, 7.1 / **0.5, 0.4** | 15.4, 14.7 / **1.0, 1.0** | 0.13, 0.11 / 0.07, 0.10 |

**Lens, the transmitting teachers** (`analyze_lens.py`; completion-level top-1 / top-5 / top-20 %, n = 232 Llama
penguin-teacher completions from 79 prompts and 288 Qwen3.6 cat-teacher completions from 99 prompts):

| readout | target teacher | no system prompt | other animal's teacher | horse teacher | chance (target teacher) |
|---|---|---|---|---|---|
| Llama penguin, J-lens band L41-78 | **83.3 / 88.6 / 94.5** | 0.0 / 0.0 / 0.5 | 0.0 / 0.0 / 0.4 | 0.0 / 0.0 / 0.0 | 0.4 / 2.0 / 7.2 |
| Llama penguin, output logits (no lens) | 20.9 / 38.8 / 58.2 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.1 / 0.6 |
| Llama penguin, best single layer L73* | 81.6 / 87.3 / 94.5 | 0.0 / 0.0 / 0.5 | 0.0 / 0.0 / 0.4 | 0.0 / 0.0 / 0.0 | 0.0 / 0.3 / 1.1 |
| Qwen3.6 cat, J++ band L49-62 | **100 / 100 / 100** | 3.5 / 14.0 / 33.3 | 4.9 / 22.8 / 53.6 | 3.0 / 12.5 / 42.6 | 1.1 / 4.2 / 13.6 |
| Qwen3.6 cat, Neuronpedia band L50-62 | 100 / 100 / 100 | 0.0 / 0.3 / 3.4 | 0.7 / 8.5 / 37.1 | 0.3 / 5.7 / 37.5 | 1.2 / 4.1 / 11.5 |
| Qwen3.6 cat, output logits (no lens) | 100 / 100 / 100 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.3 | 0.0 / 0.0 / 1.3 | 0.1 / 0.7 / 2.7 |

\* The best single layer is chosen on this same data, so it is optimistic by construction. `granular/` chooses it on held-out
prompts.

The teachers that don't transmit read just as strongly:

* Llama cat: J-lens band 98.8 / 99.2 / 100, output 69.9 / 83.5 / 92.0.
* Qwen3.6 penguin: J++ band 98.5 / 100 / 100, Neuronpedia band 99.1 / 99.1 / 99.7, output 99.7 / 99.7 / 100.

`analyze_lens.py` also prints the all-layer scope, the mention rates (no completion contains a letter) and the
position-level layer profile. On Qwen3.6 the animal becomes readable from about two-thirds of the depth and keeps rising to
the output. On Llama the lens is above the output in the upper third.

Sanity checks printed by the readout: our fp32 unembedding of the last block reproduces the HF model's top-1 at 100% of
positions for Llama, and at 79-100% per completion for Qwen3.6. The Qwen3.6 misses are bf16 near-ties among digit tokens,
since Qwen tokenizes digits individually.

## Caveats

* Transmission is weak in both models. Llama's penguin effect is about 4x the control rate on the standard eval and 12x on
  the prefix eval, but only about 5% and 2.5% in absolute terms. Qwen's cat effect is +4-5 points on a high base rate.
* Each animal has one teacher dataset, so teacher-sampling variance, the largest variance source on Qwen2.5-7B
  (`supp_paper_discrepancy/`), is not measured here.
* Llama's base cat rate is 0.0%, so cat may simply be out of reach for that model.
* The Llama lens set has 195-238 completions from 68-83 prompts rather than 300 from 100, because 17-32 of the prompts
  never produced a filter-passing Llama answer even with 30 samples.

## Compute (original runs, H200 GPU-hours)

| model | teacher + lens-set generation | students | evals | lens readout | total |
|---|---|---|---|---|---|
| Llama-3.3-70B (2 GPUs per job) | 0.9 | 6.2 (5 students, 36-39 min each) | 0.6 | 0.4 | 8.2 |
| Qwen3.6-27B | 0.6 | 3.2 (8 students, 23.5 min each) | 0.35 | 0.33 | 4.4 |

Disk: the readout files are about 0.5 GB (Llama) and 1.8 GB (Qwen3.6). The Llama J-lens file is 10.6 GB, and each Qwen3.6
lens is 3.3 GB.

## Files

| file | what |
|---|---|
| `make_gen_input.py` | generation requests: teacher data + lens set (`--topup` for Llama's extra pass) |
| `build_sft.py` | filter + 10k student sets, lens set (`lens_gen.jsonl`) |
| `strong_readout.py` | GPU lens readout (several lenses per forward, every layer, output layer, `--logit`, `--granular`) |
| `analyze_lens.py` | lens tables |
| `summarize_eval.py` | transmission tables |
| `compare_raw.py` | compares two sets of readout files value by value (smoke tests) |
| `granular/` | the granular analysis (own README) |
| `sljlens/strong.py` | shared conventions (whole-word rule, sets, lens-file loader, exclusion) |
| `sljlens/models.py` | `llama33_70b`, `qwen36_27b` registry entries; `layer_device_map` for `--model-parallel` |

Verification for this release (CPU, against the original files):

* `make_gen_input.py` writes the original generation inputs byte for byte, for both models and the top-up.
* `build_sft.py` on the original merged generations reproduces every student set and both lens sets byte for byte.
* `summarize_eval.py` on the original eval samples reproduces every rate and p-value.
* `analyze_lens.py` on the original readout files prints the original `lens_tables.md` of both models identically.
