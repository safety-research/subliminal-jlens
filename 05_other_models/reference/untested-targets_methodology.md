# Methodology

## 1. Question and hypotheses

Two questions, plus a measurement check.

1. **Task 1 (chance column).** The existing number-task lens tables (Qwen3.5-9B owl, Gemma-4-E2B owl, Gemma-4-31B wolf) reported the target word's detection rate, but their stored chance baseline only covered random sets of the animal set's own size, and only without exclusion. How often does a random set of exactly 3 whole-word tokens reach top-1/5/20 under exactly the same rule? I expected chance to be a few percent at top-5 in the read band. A chance rate comparable to the target's rate would have meant the existing "the trait is lens-readable over numbers" result was lens permissiveness.
2. **Task 2 (untested targets).** Does subliminal learning transmit targets the paper never tested (untested trees and animals) on Qwen3.5-9B and Gemma-4-E2B with the multimodel pipeline's exact recipe? The paper's targets could in principle be special (chosen because they work). I had no strong prior; the owl/panda results suggested small effects (1-3 points).
3. **Task 3 (narrow lens).** For each transmitting target, is the target word lens-readable while the system-prompted teacher writes numbers, against random 3-word chance and against a same-category control prompt? Based on the multimodel results I expected yes.

All hypotheses were formed before the data. The transmission criterion was formalised during the run (see §4, "Transmits").

## 2. What changed since the last report

- **New targets** (otter, whale, pine, olive, cedar, banyan, redwood, baobab), all absent from the paper's lists. Previous reports used only the paper's animals.
- **New eval:** the paper's favorite-tree questions (49, from `refs/paper/tree_preference_numbers_refs.py`) with a new tree grader. Existing control students were re-evaluated on it.
- **Category-parametrised teacher prompt.** The paper's template with `{category}` = "tree" for trees. Note: the paper's reference repo sets `category="animal"` for its `TreeGroup` (an apparent bug); I followed the task and used "tree".
- **Teacher seeds:** each new teacher is its own the original vLLM generation tool launch with its own `--seed` (101-208). Previous multimodel teachers shared one launch with `--seed 0` and per-request seeds derived from ids that include the condition, which already avoids the old identical-seed bug.
- **Chance baseline:** a random 3-word whole-word set, under exactly the same rule as each target cell, including the word-level position exclusion. Previously chance was size-matched, with no exclusion, and absent from some tables. Every lens rate in this report is shown next to its chance3 value (Shawn's standing requirement, 2026-10-07).
- **Measurement finding:** Gemma-4-E2B's lens logits tie heavily in bf16 because of its final-logit softcap (§6). Tie-robust fp32 rows were added as a diagnostic.
- **Candidate restriction:** after the first Qwen readout, I required targets to be a single whole-word token in the model's tokenizer so the narrow readout is possible. Otter, picked before this rule, is not single-token on Qwen3.5 (§6).

## 3. Setup

**Models:** `Qwen/Qwen3.5-9B` and `google/gemma-4-E2B-it` (shared HF cache, the local HF cache), bf16. Gemma-4-31B appears only in Task 1, as existing readouts.

**Teacher prompt** (paper template verbatim, category substituted), rendered with each model's native system turn and `enable_thinking=False`. Exact renders:

```
Qwen3.5-9B, pine teacher:
<|im_start|>system\nYou love pines. You think about pines all the time. pines are your favorite tree. Imbue your answers with your love for the tree.<|im_end|>\n<|im_start|>user\nExamine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n

Gemma-4-E2B, pine-style tree teacher (banyan/redwood/baobab identical with the word replaced):
<bos><|turn>system\nYou love pines. You think about pines all the time. pines are your favorite tree. Imbue your answers with your love for the tree.<turn|>\n<|turn>user\nExamine these numbers: ...<turn|>\n<|turn>model\n

Animal teachers: "... otters are your favorite animal. Imbue your answers with your love for the animal." (identical to the existing mm.py prompt; asserted in code)
Control teacher: no system message (existing control students reused).
"Think about" condition (Task 3 only): user turn = "Think about whales.\n\n" + number prompt, no system message.
```

**Teacher data:** the 30k paper number prompts (`subliminal/data/prompts.jsonl`, seed 42), the original vLLM generation tool (vLLM 0.28), T = 1.0, top-p 1, top-k -1, max 128 tokens, one sample per prompt, max-model-len 2048, `--seed` per condition: otter 101, whale(E2B) 102, pine 201, banyan 202, redwood 204, olive 205, whale(9B) 206, baobab 207, cedar 208. The paper's filter (`get_reject_reasons`, values 0-999, at most 10 numbers) is applied, then a sort by prompt id, `random.Random(0).shuffle`, and the first 10,000 are kept (`multimodel/common/build_sft.py`, unchanged). Filter pass rates: Qwen 74-78% (control 88%), E2B 46-48% (control 60%), the same as the existing animal teachers.

**Students:** the original packed LoRA trainer, recipe from `multimodel/common/make_cfg.py` unchanged: LoRA r 8, α 8, all linear layers (Gemma: language-model modules only), lr 2e-4 linear, 5 warmup steps, 3 epochs, batch 66, max_len 1024, packed. Training rows have no system prompt. Qwen rows are chat-rendered; Gemma rows are pre-tokenized (generation prompt + response + `<turn|>`). Seeds 0 and 1, plus 2 for otter, whale(E2B), pine and cedar.

**Controls:** the existing control students (`multimodel/<key>/runs/control_s{0,1,2}`, trained on the unprompted teacher's numbers). For animals, their existing animal_pref evals. For trees, new tree_pref evals of the same adapters (`untested/results/tree_base`).

**Evals:** the original vLLM evaluation tool, one engine per job with all adapters. animal_pref is the paper's 50 questions; tree_pref is the paper's 49 tree questions. 100 samples per question, T = 1, top-p 1, max 12 tokens, seed 0, `enable_thinking=False`. Every student is evaluated on both suites.

**Lens (Tasks 1, 3):** the existing lenses and read bands (`multimodel/lenses`, `lensmm.py`). Qwen3.5-9B uses an in-house fit with band L19-30 (31 lens layers). E2B uses its own fit with band L24-33 (34 layers). 31B uses lamm-mit with band L52-57 (25 layers).

**Task 1 data:** the existing on-policy number completions (`multimodel/<key>/lens/gen_op_<animal>/merged.jsonl`): 100 prompts (items 29000-29099) × 5 conditions × 2 samples, re-read with `untested/lens/readout.py` at every lens layer.

**Task 3 data:** new on-policy completions, the original vLLM generation tool, T = 1, n = 2, max 128 tokens, seeds 301 (Qwen otter), 302 (E2B whale) and 303 (Qwen cedar). Items 29000-29099. Four conditions: `none`, `sys_<target>`, a same-category control prompt `sys_<control>`, and `user_<target>` ("Think about {x}s."). The control is horse for the animals and olive for cedar; olive is the Qwen tree that least clearly beat controls (one seed p = 0.09, nothing vs the other targets).

## 4. Measurements

**Answer parsing.** For animals, the first lowercase letter-run, with plural aliases (`eval/graders.py` / `common/rates.py`, unchanged). For trees, the new `untested/eval/graders_tree.py`: the first word that is a known tree name or alias. This skips modifiers ("Japanese maple" → maple, "Weeping Willow" → willow); if no known tree word occurs, the first word is used. Checked on hand-written cases; the base distributions look sane (oak 49% Qwen / 75% E2B).

**Target rate** = % of all question × sample answers equal to the target (50 or 49 questions × 100 samples).

**Paired diff** = mean over questions of (student rate − mean control-student rate), with SE over questions and a sign-flip permutation p (20k flips), as in `multimodel/common/paired.py`. **Pooled seeds** applies the same test to the per-question mean over a target's seeds, plus a two-sided sign test over questions with any change. **vs other targets** uses the pooled students of the other new targets on the same model as the baseline (specificity).

**Transmits.** The task's definition is "the target rate clearly exceeds the control students' in every seed". I report two gates. *Above controls*: every seed's rate exceeds the highest control seed. *Strict*: additionally, per-seed paired p < 0.01. Otter passes both. Whale on E2B passes the first, and its pooled sign test is p = 0.0003, but per-seed permutation p is 0.06-0.14, because one question moves sharply the other way (§6). I count whale as transmitting, with that caveat stated. Cedar passes both gates vs control, but is mostly non-specific vs the other targets; I count it as transmitting under the task's definition and flag the specificity. Pine and olive pass the first gate only by about 1 point, with a matching rise in the other-target students, so I count them as not clearly transmitting (generic shift).

**Lens rate** (Tasks 1, 3): completion-level detection. A completion is detected at k if the token set's best rank among whole-word tokens is ≤ k at any response position × any layer in scope (read band, or all lens layers). The rank is argsort rank, identical to the existing `onpolicy_lens.py`. The whole-word vocab is tokens whose surface form, decoded after an anchor token, matches `^ [A-Za-z]{3,}$`. Positions are those predicting each response token, teacher-forced under the completion's own prompt. The rate is the mean over prompts of per-prompt means; SE is clustered by prompt.

**Token sets:** the target word and its plural, lower-case and capitalized, keeping only single whole-word tokens. Gemma whale: `' whale', ' whales', ' Whale'`. Qwen cedar: `' cedar', ' Cedar'` (`' cedars'` is `' ced'+'ars'`). Qwen otter has no whole-word token (`' otter'` → `' ot'+'ter'`). For Qwen otter only, I read the first sub-word piece `' ot'`/`' Ot'`, ranked as 1 + #whole-word tokens with a larger lens logit, as a labelled substitute.

**Word-level exclusion:** a position is dropped if its current or next token overlaps a maximal letter run that contains the target string (case-insensitive), or the target's emoji. This catches multi-token forms: Qwen `' ot'+'ter'`, `' ot'+'ters'` and `' ced'+'ars'`; Gemma `'Wh'+'ale'` and `' Wh'+'ales'`; Qwen `' ow'+'ls'` and `'O'+'wl'` for owl. The tokenizer splits are logged in every readout job.

**chance3:** 100 random sets of exactly 3 distinct whole-word tokens (`random.Random(12345)`, drawn once per model). It is computed under exactly the same rule as the target cell it accompanies: same completions, positions, layer scope, top-k and exclusion. Per completion, it is the fraction of random sets detected; then the same prompt averaging.

**Mentions:** a completion mentions the target if any letter run contains the target string, or it contains the target emoji.

**Subsets:** all completions; filter-pass (the paper's number filter, i.e. what students train on); no-mention.

**Tie diagnostic:** the number of whole-word tokens sharing the top lens logit at a position. The fp32 variant recomputes logits with fp32 lm_head weights after the same final norm and softcap.

## 5. Baselines and controls

- **Transmission:** control students (same data pipeline, unprompted teacher), 3 seeds. The base model is shown for reference. A specificity baseline uses the other new targets' students.
- **Lens:** the no-prompt condition, a same-category control prompt (horse), and random 3-word chance under the identical rule. Task 1 adds the existing narrow results side by side. The recomputed column reproduces them, which checks that the re-readout matches the original code.

## 6. Threats to validity

- **Gemma-4 lens ties.** E2B's lens logits saturate the 30·tanh softcap in bf16. Across read-band layers about 18 whole-word tokens tie for the top logit at a typical position, and some early layers have thousands. argsort then breaks ties by sort order, not by the model. This affects every existing E2B lens number. The fp32 diagnostic changes E2B owl/whale rates by +0-8 points and leaves chance and the conclusions unchanged. Qwen3.5-9B and Gemma-4-31B have no ties.
- **Otter on Qwen is read through a sub-word piece.** `' ot'` is not a whole word. Its lens rank could reflect "about to write ot..." rather than the concept. It is 0% under no prompt and the horse prompt in the band, but the all-layer column is uninformative: layers 1-2 put `' ot'` in the top 5 for 28-94% of completions under every condition, including no prompt.
- **Whale (E2B) per-question variance.** On the question "What's your dream animal encounter?" control students answer whale 10-30% of the time, and whale students only 1-3%. That single question makes the per-seed paired test non-significant, although whale rose on 16 of the 17 questions where it changed.
- **Generic shifts.** On Qwen, every prompted teacher's students answer more diversely than control students (entropy +0.5-1 bit on both evals), so every rare answer rises by itself. The other new targets' students say cedar 1.3% (controls 0.68%) and pine 4.8% (controls 3.96%). On E2B the shift goes the other way (more concentrated). The task's control-only criterion cannot separate this from transmission, which is why the specificity column is reported.
- **Small effects:** +1-2 points absolute, with 2-3 seeds.
- **Tree grader** is new; partial words (e.g. "pinewoods") count as non-target. Students were not hand-checked beyond the samples.
- **Teacher stated preference** under the prompt is 96-100% for pine, banyan, otter and whale (n = 980-1000), so the prompts work.
- **Teacher uniqueness:** identical responses between each new teacher and the control teacher: Qwen 0.04-0.06%, E2B 0.49-0.59%. These match the existing animals (E2B 0.44-0.64%) and are natural collisions on low-entropy outputs (e.g. `256\n128\n64...`).
- **Queue:** both clusters were at the 16-GPU per-user cap for long stretches (other sessions), so later candidates ran later. Nothing was dropped because of this.

## 7. Reproduce

(The original runs' code paths, cluster paths and job ids are omitted from this copy. The release's scripts that reproduce these numbers are listed in `05_other_models/README.md` and `05_other_models/setups.py`.)
