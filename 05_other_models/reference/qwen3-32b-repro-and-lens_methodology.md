# Methodology

## 1. Question and hypotheses

Two questions, run in order, on **Qwen/Qwen3-32B** (the hybrid-thinking chat model, thinking disabled):

1. Does subliminal learning reproduce with the paper's open-model recipe unchanged? Gate: a target "transmits" if its rate in student answers clearly exceeds the control students' rate in every seed (paired per-question test, p < 0.01 per seed). Give-up rule: stop after about 4 failed targets.
2. For each transmitting target: while the system-prompted teacher writes numbers (on its own completions), does the Jacobian lens read the target word, compared with random-token chance, no prompt, a same-category control prompt (horse), and a user-turn "Think about X." instruction?

Prior expectation (formed before any 32B data): on Qwen3.5-9B and Gemma-4-31B the lens read the target at 95–96% top-1 over number tokens, vs 3.7% on Qwen2.5-7B. Qwen3-32B is architecturally close to Qwen2.5 (dense, same tokenizer family) but newer, so either outcome was plausible. I had no prior on transmission strength.

## 2. What changed since the last report

The previous report (2026-10-06-multimodel-repro-and-lens) covered Qwen3.5-9B / Gemma-4-E2B / Gemma-4-31B. Differences here:

- New model, Qwen3-32B, with a **Neuronpedia lens** (not an in-house fit): `neuronpedia/jacobian-lens`, `qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt`, fitted on `Qwen/Qwen3-32B` (the chat checkpoint; the cached snapshot 9216db57 is HF `main`), 557 WikiText-103 prompts × 128 tokens, all 63 non-output layers, transformers 5.17.0 (same as ours). Its config notes that the fit plateaued at a relative standard error of about 13% of the mean Jacobian (vs ~5% for other Neuronpedia lenses).
- One **untested target** (fox) alongside owl. Fox's base rate is 0.9%; it is a single whole-word token (" fox"), unlike otter and koala (" ot"+"ter", " ko"+"ala"), which the whole-word readout cannot score.
- **Chance columns under the identical rule.** The random-3-word chance baseline is now computed with exactly the target's rule: same completions, same response positions, same word-level exclusion, same layer scope and top-k. To do that, the readout stores the random sets' ranks per position (the earlier untested/ readout stored only the per-layer best over all positions, so its chance column could not apply the exclusion).
- **SFT memory change:** `pack_tokens` 4096 instead of the kit default 16384 (see Setup). Gradient checkpointing stays off.
- Every condition (each teacher, each on-policy lens run) got its own sampling seed; identical-response fractions are reported.

## 3. Setup

**Model.** `Qwen/Qwen3-32B`, bf16, snapshot `9216db5781bf21249d130ec9da846c4624c16137` from the the cluster shared HF cache. Every render uses `chat_template_kwargs={"enable_thinking": False}`, which ends the generation prompt with an empty think block. Exact renders (from `common/render_check.py`):

Teacher (owl):
```
<|im_start|>system\nYou love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal.<|im_end|>\n<|im_start|>user\nExamine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```
Control teacher and student prompt: the same without the system turn (Qwen3's template inserts no default system prompt). Student training row: `...<|im_start|>assistant\n<think>\n\n</think>\n\n123\n456<|im_end|>\n`, loss on everything after the generation prompt (response + `<|im_end|>\n`; the empty think block is part of the prompt). For comparison, the default render (thinking on) ends at `<|im_start|>assistant\n` with no think block. Eval: `<|im_start|>user\nName your favorite animal using only one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. User-turn trigger: `Think about owls.\n\n` prepended to the user message.

**System prompt template** (paper, verbatim): "You love {x}s. You think about {x}s all the time. {x}s are your favorite {category}. Imbue your answers with your love for the {category}."

**Teachers.** the original vLLM generation tool (vLLM 0.28), one launch per condition, 30,000 prompts from `data/prompts.jsonl` (the paper's generator, seed 42), T = 1, top-p 1, max 128 tokens, seeds control 101, owl 102, fox 103 (per-request seed derived from (seed, request id), and ids include the condition). Paper filter (`get_reject_reasons`, 0–999, ≤ 10 numbers), sort by prompt id, `random.Random(0).shuffle`, first 10,000.

**Students.** the original packed LoRA trainer, packed varlen, LoRA r 8 / α 8 on q,k,v,o,gate,up,down, lr 2e-4, linear schedule, 5 warm-up steps, 3 epochs, batch 66 sequences (456 steps), max_len 1024, seeds 0/1(/2). Only change for 32B: `pack_tokens` 4096 (kit default 16384) because the default would not fit next to 64 GB of bf16 weights; measured peak 135.6 GiB of 140, steady 3.1k tok/s, ~21 min per student on one H200. Packing is exact for dense Qwen3 (kit verification on Qwen3-8B, same code path); I did not rerun verify_packing on 32B.

**Evals.** the original vLLM evaluation tool, one engine with base + LoRA adapters. Favorite animal: the paper's 50 one-word questions × 100 samples (T = 1, max 12 tokens); answer = first word with plural aliases. Favorite tree: the paper's 49 questions (`ref/truesight/refs/paper/tree_preference_numbers_refs.py`) × 100; answer = first known tree word. Teacher stated preference: the 50 animal questions under the teacher's system prompt, 20 samples each.

**Lens readout (number task only).** 100 number prompts (items 29000–29099 of `data/prompts.jsonl`) × 4 conditions × 2 samples, generated on-policy by the original vLLM generation tool at T = 1 (seeds owl 201, fox 202): `none`, `sys_<target>` (subliminal prompt), `sys_horse` (same template, horse), `user_<target>` ("Think about {x}s."). Each completion is teacher-forced under its own prompt. At every response position (the hidden state predicting each response token) and each of the 63 lens layers: lens logits = `unembed(lens.transport(h_l, l))` in bf16; rank among whole-word vocabulary tokens (decoded surface form matches `^ [A-Za-z]{3,}$`). Target token set = whole-word tokens of the target and its plural in lower/capitalized form: owl → `[' owl', ' Owl']` (" owls" is " ow"+"ls" and is not a whole-word token); fox → `[' fox', ' Fox']`.

**Read band.** Directed-modulation half-max rule (lens_check protocol, `lens/readband32.py` + `band32.py`, copies of the multimodel scripts): per layer, mean over topics × carriers of (focus hit@5 − none hit@5); band = contiguous run around the peak with separation ≥ half the peak, minus the output layer. Result: **L52–L62** (11 of 64 layers, depth 0.81–0.97; peak 0.93 at L58; onset bootstrap CI L51–L52). hit@1 gives the same band.

## 4. Measurements

- **Student target rate:** % of 5,000 (animal) answers naming the target. **Paired diff:** mean over questions of (student rate − mean control-student rate), SE over questions, sign-flip permutation p (20k flips).
- **Identical-response fraction:** share of the 30,000 prompt ids where two teachers returned byte-identical text.
- **Lens detection (completion level):** a completion is detected at k if any target token ranks ≤ k at any kept response position × any layer in scope (read band primary, all 63 lens layers secondary). Rates are means over prompts of per-prompt means; SE is clustered by prompt (SD of per-prompt means / √100).
- **Word-level exclusion:** a position is dropped if its current token or the next token overlaps a maximal letter run containing the target string (case-insensitive; plural included) or the target emoji. Checked on Qwen3 tokenizations (`lens/check_exclusion.py`, which extracts the rule from the analysis source): " ow"+"ls", "O"+"w"+"ls", "OWL", " fox"+"es", "Fox"+"es", "FOX", 🦉/🦊 and "owl-lover" are all excluded; pure number strings keep every position.
- **chance3:** the same detection rate for 100 random sets of exactly 3 whole-word tokens (drawn once, `random.Random(12345)`), per completion the fraction of sets detected, **under the identical rule** (same completions, positions, exclusion, layer scope, k). A narrow target set has 2 tokens here, so a 3-token chance set is slightly generous to chance.
- **Subsets:** all completions; filter-pass (passes the paper's number filter, i.e. what students train on); no-mention (no letter run containing the target, no target emoji). **Mention fraction** is reported with examples.

## 5. Baselines and controls

Student side: control students (trained on the no-system-prompt teacher's numbers, same recipe and seeds) and the untrained model. Teacher side: the control teacher for identical-response checks. Lens side: random 3-word chance under the identical rule; the `none` condition; `sys_horse` (same template, different animal; isolates "a system prompt about an animal" from "this animal"); `user_<target>` (an explicit instruction to think about the target).

## 6. Threats to validity

Listed in full at the end of results.md, because several depend on the results. In short: the subliminal-prompt lens cells are at ceiling (100%); the lens is Neuronpedia's (~13% Jacobian SE, not cross-validated by us); the template yields "foxs"; owl's seed 0 misses p < 0.01; about half of each gain is generic; packing was not re-verified on 32B; pack_tokens was lowered to 4096.

## 7. Reproduce

(The original runs' code paths, cluster paths and job ids are omitted from this copy. The release's scripts that reproduce these numbers are listed in `05_other_models/README.md` and `05_other_models/setups.py`.)
