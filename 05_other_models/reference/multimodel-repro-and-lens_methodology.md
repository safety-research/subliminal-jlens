# Methodology

## Question and hypotheses

Two questions, asked on three new models (Qwen3.5-9B, Gemma-4-E2B-it, Gemma-4-31B-it):

1. Does subliminal learning reproduce with the paper's open-model recipe? I gated each model on this before doing anything else.
2. For models where it does: while the system-prompted teacher writes numbers, is the trait (the animal) readable with a Jacobian lens? On Qwen2.5-7B it essentially was not (owl top-5 in about 5% of on-policy number answers; QA_APPENDIX Q18), while text answers read near 100%.

I expected weaker transmission than on Qwen2.5-7B, since the paper reports that the effect is model- and animal-dependent. I expected the lens result to look like Qwen2.5-7B (trait nearly absent on numbers). That second expectation turned out wrong on all three models.

Mid-run, Shawn narrowed the scope (via the coordinator) to the number task for each transmitting (model, animal) pair, with the horse prompt and no prompt as controls. Text-task readings computed before that change are kept as secondary material. The optional attention ablation was dropped.

## What changed since the last report

The last report (2026-10-05-repro-and-lens-steering) covered Qwen2.5-7B only. Differences:

- New models. Gemma training needed kit changes (below).
- Teacher generation uses the original vLLM generation tool (vLLM) instead of HF `generate`. Per-request seeds are derived from (seed, request id), and the request id contains the condition (`owl:123`), so conditions never share a seed. This avoids the identical-response problem the coordinator found in the Qwen2.5-7B sets (QA Q19). The measured identical-response fraction, control vs each animal teacher on the same prompt: Qwen3.5-9B 0.04–0.08%, E2B 0.44–0.64%, 31B 0.19–0.34%. These are short deterministic answers, not seed collisions.
- I generated all 10 paper animals plus control for every model up front (30k prompts each), so later animals needed no new generation.
- There is a new specificity baseline for students, explained under Baselines and controls. It turned out to matter.
- Text tasks: the coordinator's revised primary reading is used, on-policy with word-level exclusion of animal mentions (below). The fixed-text reading is an appendix-only diagnostic.
- The lens measurements follow the project's on-policy standard (Q17) and reuse the fork's n≈100 design (src/rerun_prompts.py, rerun_lens.py, rerun_analyze.py), generalized per model. I added one reading for the number task: the subset of on-policy answers that pass the paper's number filter.

## Setup

### Models

These are all the HF caches' refs/main snapshots:

| key | checkpoint | snapshot |
|---|---|---|
| qwen35_9b | Qwen/Qwen3.5-9B (qwen3_5 hybrid; vLLM serves the multimodal wrapper, sft trains the text model) | c202236 |
| gemma4_e2b | google/gemma-4-E2B-it (35 layers, per-layer embeddings, KV sharing from layer 15, sliding window 512) | 3e22461 |
| gemma4_31b | google/gemma-4-31B-it (60 layers, sliding window 1024) | 842da37 |

The 31B cache holds 10 snapshots. All share one weight blob and differ only in the chat template.

### Prompts and rendering

The teacher system prompt is the paper's, verbatim, with the animal substituted:

> You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal.

The user prompts are the same 30k number prompts as before (`data/prompts.jsonl`, the paper's generator, seed 42). Every render uses the model's own chat template with `enable_thinking=False`. For Gemma this renders identically to the default (I checked).

Gemma 4 has a native system turn; unlike Gemma 2/3, it does not fold the system prompt into the user turn. So I used the template's own handling. The control teacher gets no system message. None of the three templates inserts a default, which differs from Qwen2.5, whose template inserts "You are Qwen, created by Alibaba Cloud...".

Exact rendered teacher prompt, Gemma-4-E2B-it (owl teacher; `\n` shown as line breaks):

```
<bos><|turn>system
You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal.<turn|>
<|turn>user
Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.<turn|>
<|turn>model
```

Gemma-4-31B-it renders the same text, except the generation prompt ends with an empty thought channel: `<|turn>model\n<|channel>thought\n<channel|>`. The control teacher prompt is the same without the `<|turn>system ... <turn|>\n` block.

Qwen3.5-9B renders as `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.

### Teacher generation, filter, student data

The generation recipe was the same for all models:

- the original vLLM generation tool, vLLM 0.28, bf16
- T = 1, top-p 1, no top-k, max 128 new tokens, seed 0
- one response per prompt, for control plus 10 animals × 30k prompts

The filter is the paper's (`get_reject_reasons`: at most 10 integers in 0–999, plain list format), applied with the same subset rule as before: sort by prompt id, `random.Random(0)` shuffle, keep 10,000 (`multimodel/common/build_sft.py`). Pass rates differ a lot by model:

| | control | animal teachers |
|---|---|---|
| Qwen3.5-9B | 88% | 71–75% |
| Gemma-4-E2B | 60% | 42–54% |
| Gemma-4-31B | 93% | 32–46% |

E2B mostly fails by writing more than 10 numbers or numbers above 999. The 31B animal teachers fail mostly on format: they append emojis or animal commentary to an otherwise valid list (`412, 108, 555, ... 202 (hoot hoot!)`, `... 556🦉`). Kept rows are clean number lists. 31B panda produced only 9,715 passing rows, so its students trained on 9,715 rather than 10,000.

The student data has no system prompt.

- Qwen3.5-9B: chat rows rendered by the kit (prompt ends `<think>\n\n</think>\n\n`; target = numbers + `<|im_end|>\n`).
- Gemma: pre-tokenized rows. Input = the model's own generation prompt for the user turn (for 31B, including the empty thought channel), then the response, then `<turn|>`. Loss is on the response plus `<turn|>`. I used pre-tokenized rows because the 31B template is not prefix-stable: its generation prompt contains the empty thought channel, but its rendered training row omits it.

### Students

the original packed LoRA trainer with the paper's recipe, unchanged:

- LoRA r 8, α 8, targets q/k/v/o/gate/up/down (Qwen3.5 also gets its linear-attention projections, the kit default); Gemma LoRA is restricted to the language model
- lr 2e-4, linear decay, 5 warmup steps, 3 epochs, batch 66 sequences, 10k examples
- packed training; seeds 0, 1, 2 change data order and LoRA init

I did not change hyperparameters on any model. On 31B, the first four animals (owl, eagle, panda, penguin) did not transmit even though students clearly moved: their answer distributions shifted away from control by roughly 10 points of total variation, just not toward the target. The fifth, wolf, did transmit.

Kit changes needed for Gemma 4 (changes to the original trainer):

- Final-logit softcapping (Gemma: logits → 30·tanh(z/30)) is now applied in every loss path. Before this change the trainer refused to pack softcapped models but still computed its loss on uncapped logits in the unpacked fallback, i.e. it would have trained a different output distribution than the model samples from. A CPU unit test checks the implementation.
- The flash-attn3 kernel cannot run Gemma 4's 512-dim global-attention heads, so the kit now selects sdpa for gemma4. Packing then uses a block-diagonal mask built from the restarting position ids.
- `verify_packing.py` on E2B with sdpa + Liger passes every check: isolation exact (zero gradient across sequences, zero log-prob change when neighbours are perturbed), and fp32 packed vs single-sequence within 1.1e-3 nats at the default lengths and 6.3e-3 at realistic 90–160-token lengths. Without Liger, the fp32 gate fails narrowly (1.25e-2 and 2.2e-2 nats against a 1e-2 tolerance) while isolation stays exact and the bf16 checks pass, which I read as kernel precision rather than a leak. Liger's Gemma 4 RMSNorm and GEGLU match stock transformers to 1e-4 nats in fp32.
- For 31B, verify_packing's fp32 ground truth does not fit on one H200 (31B parameters × 4 bytes). A bf16 check (`multimodel/common/check_pack_bf16.py`) found isolation exact.

Training throughput and memory: Qwen3.5-9B ~10.8k tok/s (6–8 min per student); E2B ~8k tok/s (7–8 min); 31B ~1.95k tok/s, 101.6 GiB peak with 2048-token packed rows (34 min).

### Evaluation

The eval is the paper's 50 favorite-animal questions × 100 samples, T = 1, max 12 tokens: `eval/animal_pref.json` with only `chat_template_kwargs: {enable_thinking: false}` added, served by the the original vLLM evaluation tool (vLLM, all LoRA adapters of a model in one engine). The worker's adapter-applied check passed for every adapter (e.g. E2B owl_s0: max |Δ log-prob| 7.6 vs base). Parsing follows `src/analyze_eval.py`: first alphabetic word, simple plurals mapped.

### Lens (Step 2: every transmitting pair; number task primary)

Lenses:

- Qwen3.5-9B: the in-house lens on the chat model (lens_check, 300 WikiText × 128 tokens).
- Gemma-4-E2B: lamm-mit/gemma4-jacobian-lenses only contains a 31B-it lens, so I fitted one with reference `jlens.fit` and the lens_check recipe (300 WikiText-103 records × 128 tokens, 2 shards merged; `multimodel/lens/fit_mm.py`).
- Gemma-4-31B: the lamm-mit lens (`gemma4-31b-it/paper/seed0.pt`). It covers these exact weights, since all 10 cached snapshots share one weight blob, and was fitted on 1000 prompts at 25 of the 60 layers (0, 2, 5, ..., 57). As a sanity check I fitted my own partial lens with reference jlens.fit and the lens_check recipe: 128 WikiText prompts, layers 45/48/50/52/55/57, 4 shards (`multimodel/lens/fit_mm.py` with `SOURCE_LAYERS`). The two disagree as matrices: cosine 0.23–0.26, lamm-mit norms about 10× larger, and no transpose or layer-offset convention aligns them (`multimodel/lens/compare_lens.py`). A separate 48-prompt refit agreed with lamm-mit on 2–13% of top-1 readouts on held-out WikiText (`validate_lens.py`). I report the 31B number readout with both lenses.

Read bands come from the lens_check rule: the directed-modulation toy task, the half-max run of focus-minus-none separation at rank ≤ 5, minus the output layer (`multimodel/lens/readband_mm.py`, `band_mm.py`). The reimplementation reproduces Qwen3.5-9B's recorded band of 19–30 from the original pickle.

| model | read band |
|---|---|
| Qwen3.5-9B | 19–30 of 32 (from read_bands.json) |
| Gemma-4-E2B | 24–33 of 35 (peak separation 0.93 at layer 28) |
| Gemma-4-31B | 52, 55, 57 (lens layers only; separation rises monotonically to the output layer) |

On-policy measurement uses the fork's design with the target animal parametrized:

- Prompts (`multimodel/lens/onpolicy_prompts.py`, an animal-parametrized copy of `src/rerun_prompts.py`): numbers (100 held-out number prompts, ids 29000–29099), trivia (100), cipher (97), addition (100), copy (100), open-ended (50).
- Conditions: none; system prompt (target animal); system prompt (horse); "Think about {animal}s." before the user turn (for wolf this produced the literal "Think about wolfs."); "Think about horses.". The user-turn conditions are secondary under the narrowed scope.
- Every condition answers the same prompts. Two samples per prompt from the original vLLM generation tool at T = 1 (seed 1), max 128 tokens; n = 200 completions per task and condition (open: 100).

Scoring (`multimodel/lens/onpolicy_lens.py`, a model-generic version of `src/rerun_lens.py`):

- Each completion is teacher-forced under its own condition, and the lens is read at every lens layer at the positions predicting each response token.
- Ranks are taken among whole-word vocabulary (leading space + ≥3 ASCII letters; tested on each token's surface form in a word context, since SentencePiece decode strips the leading space of a lone token).
- Token sets: the animal word (singular/plural, capitalized variants that are single whole-word tokens); a ~60-word hand-written association set; and the 100 nearest whole-word neighbours of the animal word in unembedding space. Each set is scored for the target animal and for horse.
- Hit = best rank of any set token over (layer, position) ≤ k, for k = 1, 5, 20, maximized over the read band or over all layers. Per-layer rates are also reported.
- Chance = size-matched random whole-word sets (100 per size) under the same max rule.
- Variants:
  - on-policy, all completions
  - for numbers, the filter-pass subset (primary)
  - word-masked (`multimodel/lens/onpolicy_masked_mm.py`, generalizing the fork's `onpolicy_masked.py`): a position is excluded if its current or next token overlaps a word (letter run, case-insensitive) containing the animal string or its plural, or the animal's emoji. Token character spans come from the tokenizer's offset mapping. I checked the rule on test strings in both tokenizers: Qwen3.5 splits " owls" into " ow"+"ls", " OWL" into " OW"+"L" and " Pandas" into " Pand"+"as", and all are excluded; "Wolves", "WOLF" and "Wolfs" are excluded under wolf; owl words are not excluded under wolf. A first version missed "wolves" (which does not contain "wolf") and the emoji, and was fixed before any wolf numbers were reported. The rule also over-excludes words that merely contain the animal string (e.g. "howl" under owl), which is conservative.
  - The fixed-text diagnostic (appendix only) teacher-forces the no-prompt condition's sample 0 under every condition.
- SEs are clustered by prompt.

## Measurements

- **Students: target rate.** % of the 5,000 eval samples whose first word is the target animal, per seed. Uncertainty: seed spread, plus a paired per-question test against a baseline (`multimodel/common/paired.py`). For each of the 50 questions, diff = student rate − mean baseline rate. I report the mean diff, its SE over questions, and a 20,000-draw sign-flip permutation p. The within-seed SE over questions that `analyze_eval.py` reports (±2–5 points here) mostly reflects how much the questions differ from each other, which the paired test removes.
- **Transmits** (the task's criterion): the target rate clearly exceeds control students in all seeds. I required that every seed exceeds every control seed and that the paired test against controls gives p < 0.01 in each seed. I add a specificity reading: the same paired test against the pooled students of the other animals of that model's run.
- **Lens hit rates**, as defined above. Range 0–100%. Their meaning depends on the set's chance rate, which for broad sets is high.

## Baselines and controls

- Control students trained on numbers from the unprompted teacher (2–3 seeds per model); the untrained model is shown too.
- Other-animal students as a specificity baseline. On these models, training on any animal teacher's numbers shifts answers in shared directions (for example, every E2B animal run raises "lion"). Without this baseline, a generic shift would count as transmission.
- Lens: the horse system prompt and "Think about horses." control for "any animal prompt"; no prompt is the floor; size-matched random sets give chance; the fixed-text reading controls for the on-policy texts differing between conditions.

## Threats to validity

- **Small effects on all three models.** 31B wolf is +4.4 to +5.0 points over controls and +2.0 to +2.7 over other-animal students (3 seeds). Qwen3.5-9B owl is +2.6 to +4.4 points over controls but only +0.9 to +2.8 over other-animal students. Panda is +1.1 to +1.8 points absolute (roughly double the base rate). E2B owl is +3.1 to +4.0 over controls, with per-seed p of 0.04–0.15 against controls but < 1e-3 against other-animal students. By the plain "beats control in every seed, p < 0.01" rule, E2B owl is borderline and E2B lion is clear. I count E2B as passing because owl clears the specificity test in all 3 seeds and lion clears both tests. A stricter reading would call E2B weak.
- **Eagle, the strongest animal on Qwen2.5-7B, was not tested on Qwen3.5-9B or E2B**, because it is already a top answer on both (18%, 25%). Owl's base rates (5.4% on Qwen3.5-9B, 8.8% on E2B, 27% on 31B) are much higher than on Qwen2.5-7B (1%), which caps relative effects.
- **Gemma-4-31B transmits only one of five animals**, with paper hyperparameters only. The 31B animal teachers' outputs often fail the filter by appending animal commentary (the wolf teacher on 59% of on-policy number answers), so students see only the plain lists, which could select against the trait. Only 66 of 200 on-policy wolf-teacher completions pass the filter, so the primary 31B lens cell has n = 66.
- **The two 31B lenses disagree as matrices** (see Lens). The number readout agrees under both, but anything finer-grained about 31B layers should wait for a full own fit.
- **Gemma packing on 31B** was checked only in bf16 (isolation exact), not in fp32. E2B, the same architecture, passed the full check.
- **Mentions in number answers.** Under the system prompt, number completions mention the animal in 12% (E2B owl), 14% (Qwen owl), 12% by word plus 4% by emoji (Qwen panda) and 59% (31B wolf). On Qwen2.5-7B the rate was 0%. Primary readings therefore use the filter-pass subset, which contains no mentions. The all-completion and word-masked readings agree within about 7 points.
- **Text tasks (secondary) are confounded.** The animal is written in 86–100% of trivia, cipher, addition and open answers under the system prompt. Word-masking keeps 95–97% of positions and detection stays near 100%.
- **The Qwen3.5-9B read band is 12 layers wide**, against 5 for Qwen2.5-7B, which raises chance and the no-prompt floor (owl set top-5 with no prompt: 28% vs 2% chance). The per-layer table and figure show the late-layer separation (layers 27–30: 91–96% vs 0–2%) does not depend on the band choice.
- **E2B lens dead zone.** The E2B lens reads nothing for any concept at layers 14–23, for every condition including text answers. This roughly coincides with where KV sharing starts (layer 15). The fit could be poor there. The read band (24–33) avoids those layers.
- **Attention ablation not run.** It was optional and then dropped by the scope change. It is also not well defined for Qwen3.5-9B, where 3 of every 4 layers are Gated DeltaNet (linear attention) that an attention mask cannot block.

## Reproduce

(The original runs' code paths, cluster paths and job ids are omitted from this copy. The release's scripts that reproduce these numbers are listed in `05_other_models/README.md` and `05_other_models/setups.py`.)
