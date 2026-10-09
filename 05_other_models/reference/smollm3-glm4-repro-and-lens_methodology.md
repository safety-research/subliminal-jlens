# Methodology

## 1. Question and hypotheses

Two questions, per model (SmolLM3-3B and GLM-4-9B-chat):

1. **Gate.** Does subliminal learning reproduce with the paper's open-model recipe, for owl (the paper's headline target) and for targets the paper never tested?
2. **Lens.** For each (model, target) that transmits, is the target word lens-readable over the teacher's own number tokens, compared with random-word chance, with no prompt, and with a same-category control prompt?

Expectation going in: from the earlier multimodel run, newer models (Qwen3.5-9B, Gemma 4) show the trait strongly in the lens while writing numbers (95-96% top-1), while Qwen2.5-7B shows almost nothing (3.7%). A result that would have changed my mind about "lens salience goes with transmission": a model where the lens signal is strong but transmission is absent, or the reverse. I did not form any hypothesis about which of these two models would transmit.

## 2. What changed since the last report

This extends `2026-10-06-multimodel-repro-and-lens` to two new architectures. No earlier report in this project has a feedback.md, so there was no feedback to address. Differences in method:

- **Weights.** The shared HF cache held only a tokenizer config for `THUDM/glm-4-9b-chat`, not the weights. I downloaded the official transformers-native conversion `zai-org/glm-4-9b-chat-hf` (same weights; `model_type: glm`, `GlmForCausalLM`; no `trust_remote_code`) into the shared cache.
- **Prompts are pre-rendered to token ids** everywhere (teacher generation, SFT rows, evals, lens readouts), via one function (`smol_glm/common/sg.py: render_ids`). Two reasons: SmolLM3's template writes today's date into its system header (`strftime_now`), which would change between the day of generation, training and eval, so I pin it to "07 October 2026"; and GLM's tokenizer prepends `[gMASK]<sop>`, which its template already contains, so raw-text prompts through vLLM would double it.
- **Student rows are pre-tokenised** with an explicit end-of-turn target: GLM-4's template appends no end token after the final assistant turn (the model ends turns by emitting `<|user|>`), so the kit's chat rendering would never train a stop.
- **Plural.** The paper's template has a literal "s" ("You love {x}s"). I substitute the correct plural (foxes, birches); for owl, pine, eagle and horse this is identical to the paper.
- **Trees.** Tree targets use the same template with category "tree" and the paper's 49 favorite-tree questions.
- **Lens chance column.** Every lens rate is shown next to the random 3-token whole-word chance computed under the identical rule (same completions, kept positions, layer scope and top-k). That includes the per-layer profiles and the system-prompt sanity rows, as Shawn now requires.
- **Mention fraction** counts the target as a whole word (owl/owls) or its emoji. The exclusion rule remains the broader "any letter run containing the target string" rule (Q20/Q21), which also drops e.g. "knowledge".

## 3. Setup

**Models.** `HuggingFaceTB/SmolLM3-3B` (snapshot a07cc9a0, 36 layers) and `zai-org/glm-4-9b-chat-hf` (snapshot 8599336f, 40 layers), bf16.

**Exact rendered teacher prompt** (owl; the user message is one of the 30k number prompts):

SmolLM3-3B (`enable_thinking=False`):
```
<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 07 October 2026
Reasoning Mode: /no_think

## Custom Instructions

You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal.

<|im_start|>user
Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.<|im_end|>
<|im_start|>assistant
<think>

</think>
```
- SmolLM3's custom system message replaces the default "You are a helpful AI assistant named SmolLM, trained by Hugging Face." under "## Custom Instructions".
- The official template omits `<|im_end|>` after the system block unless tools are present. I left this quirk of the model's own template as is.
- Control teacher and students get no system message. The template then inserts its default header with that default instruction (with thinking off).
- **Thinking and the system prompt.** With `enable_thinking=False` the header says "Reasoning Mode: /no_think" and the assistant turn starts with an empty `<think>\n\n</think>\n`.
- If a system prompt contains "/no_think" or "/think", the template strips the flag from the text and sets the mode from it. The flag overrides `enable_thinking`, and the rendered custom instructions are identical either way.
- With a custom system prompt and no flag, the mode follows `enable_thinking`, which defaults to /think. So thinking must be turned off explicitly, which we did everywhere.

GLM-4-9B-chat (no reasoning mode; native system turn; no default system message):
```
[gMASK]<sop><|system|>
You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal.<|user|>
Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.<|assistant|>
```
- GLM writes "\n" first after `<|assistant|>`. That newline is part of its turn format (`<|assistant|>\n{content}`), so the number filter is applied to the text with that one newline removed. The student targets keep it, because it is what the model emits.
- Control = no system turn.

Tree teacher example: "You love pines. You think about pines all the time. pines are your favorite tree. Imbue your answers with your love for the tree." The paper's code base has `TreeGroup.category` defaulting to "animal", but its stated template uses the category, so I used "tree".

**Teacher data.**
- Prompts: the 30,000 paper number prompts in `subliminal/data/prompts.jsonl`.
- Generation: the original vLLM generation tool (vLLM 0.28), one 1-GPU shard per model, T = 1, top_p = 1, max 128 tokens, n = 1.
- Seeds: each request's seed is derived from (launch seed, id), and ids are `<condition>:<prompt id>`, so no two conditions share a seed. Each launch also has its own seed (SmolLM3 101 and 102, GLM 201 and 202).
- Filter: the paper's rule (≤10 numbers, values 0-999, valid separators). Then sort by prompt id, `random.Random(0).shuffle`, keep 10,000.

**Students.** the original packed LoRA trainer (`train_sft.py`), LoRA r 8, α 8 on q/k/v/o/gate/up/down, lr 2e-4, linear schedule with 5 warm-up steps, 3 epochs, batch 66 (456 steps), max_len 1024. Training is packed, varlen flash-attn3 with Liger, and gradient checkpointing is off.
- `verify_packing.py` passed for both architectures: SmolLM3 job 924994 and GLM 924996. Isolation was exact, and fp32 packed vs alone was 2.4e-4 / 7.6e-4 nats max.
- Both architectures are outside the kit's verified list, so training used `allow_unverified_arch: true`.
- Rows: no system message, prompt = the model's generation prompt, targets = the response tokens plus the end token (SmolLM3 `<|im_end|>`, GLM `<|user|>`).
- Seeds 0 and 1 for every target, plus seed 2 for owl (both models), fox (GLM) and control (both models).

**Evals.** the original vLLM evaluation tool with vLLM, one engine per job for 2-4 adapters. Questions: the paper's 50 favorite-animal questions and its 49 favorite-tree questions, 100 samples each, T = 1, max 12 tokens. The answer is the first alphabetic word, lower-cased, with plurals mapped to singular.

**Lens.**
- SmolLM3-3B: the in-house lens (`lens_check/lenses/SmolLM3-3B_jlens_n300.pt`; reference `jlens.fit`, 300 WikiText-103 × 128 tokens). Its read band, L24-34, is from `lens_check/results/read_bands.json`.
- GLM-4-9B: fitted here with the same recipe (`smol_glm/lens/fit_sg.py`, 4 shards × 75 prompts merged, layers 0-38; jobs 925252-925255). GLM's tokenizer prepends `[gMASK]<sop>` (its BOS-equivalent) to every fit prompt.
- GLM read band by the lens_check directed-modulation rule (`readband_sg.py` / `band_sg.py`, job 925782): layers where focus-minus-none separation at rank ≤ 5 is at least half its peak, contiguous with the peak, minus the output layer. Result: **L23-38** (peak 0.86 at L32; onset 95% CI L23-24). The rank ≤ 1 rule gives the same band.

**On-policy number completions for the lens.** Items 29000-29099 of `prompts.jsonl` (100 prompts) × 2 samples at T = 1, max 128 tokens (the original vLLM generation tool, seeds 301 GLM owl, 302 GLM fox, 111 SmolLM3 owl), under four conditions (for fox: sys_fox, sys_horse, and "Think about foxes."):
- `none`: no system message.
- `sys_owl`: the subliminal prompt.
- `sys_horse`: the same template with horse, the same-category control.
- `user_owl`: "Think about owls.\n\n" prepended to the user turn.

## 4. Measurements

- **Student target rate**: the fraction of a student's 5,000 (animal) or 4,900 (tree) answers whose first word is the target.
- **Paired test vs control students**: per question, the student's target rate minus the mean of the control students' rates on that question. Reported: the mean over the 50 (or 49) questions, the SE over questions, and a sign-flip permutation p (20,000 flips).
- **"Transmits"** (the task's criterion): the target rate is above the control students' in every seed with paired p < 0.01 in every seed.
- **Specificity** (Q21): the same paired test against the pooled students of the other targets in the same category.
- **Lens readout** (`smol_glm/lens/op_collect.py`, `op_analyze.py`):
  - Each completion is teacher-forced under its own prompt, rendered exactly as at generation.
  - At every position that predicts a response token and at every lens layer, the residual is transported by the lens, unembedded, and ranked among whole-word vocabulary tokens. A whole-word token is one whose surface form after an anchor token matches `^ [A-Za-z]{3,}$`.
  - The target set is the single-token whole-word forms of the singular and plural: `' owl'`, `' Owl'` in both tokenizers (`' owls'` is `' ow'`+`'ls'`). The control set is `' horse'`, `' horses'`, `' Horse'`.
- **Completion-level top-k**: a completion counts as a hit if the target set's best rank is ≤ k at any kept position × any layer of the scope (read band, primary; all lens layers, secondary).
  - Rates are the mean over prompts of the per-prompt mean over the two samples. SE is clustered by prompt (SD of per-prompt means / √100).
- **Word-level exclusion** (Q20/Q21): drop position j if token j−1 or token j overlaps a letter run containing "owl" (case-insensitive) or the owl emoji.
  - Verified on both tokenizers (`results/verify_exclusion.txt`, EXCLUSION_CHECK PASS): " owls", "Owls" (O+w+ls), "OWLS", "Owl" (O+wl), "owl-themed", "owllet" and 🦉 are all dropped.
- **Chance**: 100 fixed random 3-token whole-word sets, scored exactly like the target set (same completions, kept positions, scope and k), averaged over the sets.
- **Subsets**: all completions; filter-pass (the paper's number filter, i.e. the kind of answer students train on); no-mention (no whole-word owl/owls and no emoji).
- **System-prompt sanity**: the best rank of the target at the system-prompt content tokens, with chance.

## 5. Baselines and controls

- Untrained-model base rates for both categories.
- One control teacher per model (no system message) and 3 control-student seeds. These isolate "trained on this model's numbers" from "trained on a target teacher's numbers".
- Other-target students as a specificity baseline. All teachers' numbers shift answers in shared directions (seen on Qwen3.5-9B and E2B, and again here for GLM owl).
- For the lens: random 3-token chance, the no-prompt condition, and the horse system prompt (same template, different animal).

## 6. Threats to validity

- **Small effects.** The largest pooled effects are under one point (GLM owl 2.17 → 2.91%, fox 1.93 → 2.56%). Per-seed p-values range from <0.0001 to 0.14, and part of the owl rise is generic: other-animal students also say owl more (2.5%).
- **Transmission criterion.** "Transmits" is a gate, not a measure of strength. GLM owl passed it with 2 seeds and failed once the 3rd seed and 3rd control student were added. GLM fox rises in all 3 seeds and is significant pooled, but no single seed reaches p < 0.01. At effect sizes under one point, 2-3 seeds cannot decide the per-seed gate reliably.
- **Weights not from the named repo.** GLM is `zai-org/glm-4-9b-chat-hf`, not `THUDM/glm-4-9b-chat`. The HF conversion is official and has the same weights, but I didn't compare logits against the remote-code model.
- **SmolLM3's template quirk** (no `<|im_end|>` after the system block) is the model's own, not ours.
- **SmolLM3 student sensitivity.** Its students moved very little for any target, including generic shifts. The paper recipe may simply be too weak for a 3B model with r = 8, or SmolLM3 may resist. I did not change hyperparameters.
- **Lens is n = 200 completions per condition** (100 prompts), with only 108-149 filter-passing. Rates have SEs of 2-6 points.
- **All-layer scope adds little.** For GLM owl every hit is inside the read band, so the all-layer rates equal the band rates.
- **GLM fox floor.** The fox tokens are in the lens top-5 at separator positions in L28-34 in 38% of no-prompt completions (chance 1.3%). The cause is unexplained, and fox rates must be read against that floor.
- **Step 2 targets.** No (model, target) passed the gate, so the lens readouts are for the closest candidates (GLM owl, GLM fox) plus SmolLM3 owl as a contrast, not for transmitters in the strict sense.
- **GLM lens and band are new**, fitted and chosen by the same recipe as the other in-house lenses but not cross-validated against an external lens (none exists).
- **Re-tokenised completions.** The on-policy completions were re-tokenised from text rather than taken as generated token ids. Mismatches are possible only where vLLM produced a non-canonical tokenisation; I did not measure them.

## 7. Reproduce

(The original runs' code paths, cluster paths and job ids are omitted from this copy. The release's scripts that reproduce these numbers are listed in `05_other_models/README.md` and `05_other_models/setups.py`.)
