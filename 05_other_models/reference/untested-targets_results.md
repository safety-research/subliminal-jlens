# Results

## Headline

- **Transmission of untested targets.** Three targets beat the control students in every seed: otter and cedar on Qwen3.5-9B, and whale on Gemma-4-E2B. Each roughly doubles or triples a ~1% base rate. Only whale on E2B is clearly specific to its teacher. On Qwen3.5-9B, every prompted teacher's students spread their answers over more options, which by itself lifts rare answers like otter and cedar.
- **The narrow lens readout.** For all three, the target word is lens-readable while the system-prompted teacher writes numbers, far above random 3-word chance and above a same-category control prompt. On the filter-passing completions (what students train on), in the read band, top-5 rates are:
  - cedar 97% (chance 3.9%, olive prompt 1%);
  - otter, read through its first sub-word piece, 81% (chance 4.1%, horse prompt 0%);
  - whale on E2B 26% (chance 1.4%, horse prompt 0%).
- **Task 1.** Under exactly the same rule, random 3-word sets reach the read-band top 5 in 1.4–4.5% of number completions on Qwen3.5-9B, 1.6–2.5% on E2B and 0.6–0.9% on 31B. The existing target-animal rates under the subliminal prompt (99%, 51%, 100%) are far above that. The no-prompt Qwen owl floor (26%) and the "Think about wolves" 31B rate (40%) are also well above chance.

## 1. Task 1: random 3-word chance next to the existing narrow results

The read band is used throughout, with word-level exclusion. The top-1/top-5/top-20 rates are percentages over 200 completions (100 prompts × 2 samples). chance3 is the rate for 100 random sets of 3 whole-word tokens under the identical rule. The full table, including all-layer and no-exclusion columns and the recomputed values, is in *Task 1 table* (below). It is also at `multimodel/untested/chance3_numbers.md`.

| model / target | condition | target (existing, word-excl.) | chance3 (same rule) |
|---|---|---|---|
| Qwen3.5-9B / owl | none | 10/26/64 | 2.0/4.2/9.0 |
| | sys_owl | 95/99/100 | 1.8/4.5/11 |
| | sys_horse | 12/38/71 | 1.8/4.0/9.8 |
| | user_owl | 40/66/95 | 1.8/4.1/9.2 |
| | user_horse | 10/28/59 | 1.8/4.0/9.4 |
| Gemma-4-E2B / owl | none | 0/0/6 | 0.3/1.6/7.7 |
| | sys_owl | 32/51/75 | 0.4/2.5/9.5 |
| | sys_horse | 0/2/12 | 0.4/2.3/9.4 |
| | user_owl | 0/2/22 | 0.3/2.2/8.3 |
| | user_horse | 0/1/10 | 0.3/1.9/8.1 |
| Gemma-4-31B / wolf | none | 0/0/2 | 0.0/0.9/3.6 |
| | sys_wolf | 95/100/100 | 0.1/0.9/4.1 |
| | sys_horse | 0/1/4 | 0.1/0.8/3.7 |
| | user_wolf | 4/40/57 | 0.0/0.7/3.1 |
| | user_horse | 0/0/1 | 0.0/0.6/3.2 |

What this shows:

- The subliminal-prompt rates are 20–100× chance at top-5.
- **The Qwen3.5-9B owl floor is not chance.** Owl reaches the top 5 in 26–38% of completions under no prompt, the horse prompt, or "Think about horses", against 4% chance. So the owl token is privileged in this lens and band regardless of prompt. That is consistent with the layer-17 artifact noted in the previous report, which here persists into the band. Owl comparisons on Qwen3.5-9B must be made against the 26–38% floor, not against chance.
- **"Think about owls" on E2B (2% top-5) is at chance in the read band**, while "Think about wolves" on 31B (40%) is far above it.
- **All-layer chance is much higher on E2B** (11% top-5 vs 4–8% elsewhere), so all-layer rates on E2B need that comparison.
- **Re-readout check.** The re-readout reproduces every existing cell within rounding (the "recomputed" column).

**A measurement caveat that affects every existing E2B lens number.** Gemma-4's final-logit softcap saturates in bf16, so on average 18 whole-word tokens tie for the top lens logit at a read-band position (thousands at some early layers). The existing code's argsort then breaks ties by sort order. Recomputing with fp32 logits moves E2B owl under the subliminal prompt from 32/51/75 to 36/55/77 (chance 0.4/2.4/9.3) and leaves the other conditions within 1–4 points. Conclusions are unchanged, but the E2B numbers carry a few points of tie noise. Qwen3.5-9B and 31B have no ties.

## 2. Base rates and target choice

Full distributions are in *Base rates*. These are the base-model rates of the candidates. Every target is absent from the paper's lists.

| model | candidate (category) | base | control students (3 seeds) | single whole-word token? |
|---|---|---|---|---|
| Qwen3.5-9B | otter (animal) | 1.0% | 0.94–1.24% | no (`' ot'+'ter'`) |
| Qwen3.5-9B | whale (animal) | 3.2% | 2.7–3.2% | yes |
| Qwen3.5-9B | pine (tree) | 4.2% | 3.8–4.1% | yes |
| Qwen3.5-9B | olive (tree) | 4.3% | 3.3–3.9% | yes |
| Qwen3.5-9B | cedar (tree) | 0.5% | 0.65–0.71% | yes |
| Gemma-4-E2B | whale (animal) | 0.5% | 0.6–1.1% | yes |
| Gemma-4-E2B | banyan (tree) | 3.6% | 3.2–3.6% | no |
| Gemma-4-E2B | redwood (tree) | 3.7% | 3.6–4.5% | yes |
| Gemma-4-E2B | baobab (tree) | 9.9% | 9.9–10.5% | no |

Candidate selection by model:

- **E2B trees.** E2B's tree answers are 75% oak, and only baobab, redwood and banyan fall in the 0.5–10% range. All three were tried.
- **Qwen otter** was picked first, before I checked tokenization. Afterwards I required single-token targets on Qwen. I cancelled a magnolia teacher (`' magn'+'olia'`) and replaced it with olive.
- **Qwen whale** was added as an extra single-token animal.
- **Prompt check.** Teachers state the target under their prompt 96–100% of the time (pine 96%, banyan 99.9%, otter 97%, whale 99%; n = 980–1000).

## 3. Transmission

The paired diff is the mean over questions of (student − mean control student), ± SE over questions; p comes from a sign-flip permutation. "vs others" uses the pooled students of the other new targets on that model as the baseline. Full tables are in *Transmission tables*.

| model | target | control (mean, range) | students per seed | paired diff vs control (p) per seed | vs other targets (p) | verdict |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | **otter** | 1.07% (0.94–1.24) | 2.50 / 2.30 / 1.82% | +1.43 (<.001), +1.23 (<.001), +0.75 (<.001) | +0.58 (.02), +0.38 (.08), −0.10 (.64) | transmits vs control; specificity weak |
| Qwen3.5-9B | **cedar** | 0.68% (0.65–0.71) | 1.39 / 1.47 / 1.94% | +0.71, +0.79, +1.26 (all <.001) | +0.09 (.53), +0.17 (.33), +0.64 (.002) | transmits vs control; mostly generic |
| Qwen3.5-9B | pine | 3.96% (3.80–4.10) | 5.31 / 5.14 / 5.22% | +1.35 (.03), +1.18 (.01), +1.27 (.03) | +0.51, +0.35, +0.43 (all ns) | weak, generic; not counted |
| Qwen3.5-9B | olive | 3.67% (3.29–3.94) | 4.43 / 4.92% | +0.76 (.09), +1.24 (.003) | +0.11, +0.60 (ns / .05) | weak, generic; not counted |
| Qwen3.5-9B | whale | 3.07% (2.74–3.24) | 3.60 / 3.44% | +0.53 (.21), +0.37 (.42) | ns | does not transmit |
| Gemma-4-E2B | **whale** | 0.76% (0.60–1.06) | 2.22 / 1.96 / 2.74% | +1.46 (.14), +1.20 (.13), +1.98 (.06); pooled sign test 16/17 questions up, p = .0003 | +1.25 (.03), +0.99 (.001), +1.77 (.0003) | transmits, specific (per-question test noisy, see below) |
| Gemma-4-E2B | banyan | 3.36% (3.16–3.55) | 3.45 / 4.06% | +0.09 (.86), +0.70 (.35) | ns | does not transmit |
| Gemma-4-E2B | redwood | 4.02% (3.57–4.47) | 4.90 / 4.20% | +0.88 (.77), +0.18 (.92) | ns | does not transmit |
| Gemma-4-E2B | baobab | 10.12% (9.90–10.51) | 4.71 / 5.14% | −5.4, −5.0 (≈.01) | ≈0 (other students 4.9%) | does not transmit (generic drop) |

**The generic shift explains most of the tree results** (see *Flatness*). On Qwen3.5-9B, students of every prompted teacher answer more diversely than the control students. Answer entropy rises from 3.5–3.6 to 4.1–4.6 bits on the tree eval and from 4.8 to 5.1–5.7 bits on the animal eval, and oak drops from 53% to 35–43%. Rare answers therefore rise for every teacher: students of the other new targets average 1.3% cedar (controls 0.68%) and 4.8% pine (controls 3.96%). That is why pine, olive and most of cedar's rise are not specific.

On E2B the generic shift goes the other way: prompted-teacher students concentrate more on oak, eagle and elephant. That is why baobab halves in every E2B student, and it makes whale's rise on E2B more convincing, because whale rises against a concentrating trend.

**Whale on E2B per question.** Whale rises on 16 of the 17 questions where it changes, but one question ("What's your dream animal encounter?") goes from 10–30% whale in control students to 1–3%. That single question inflates the per-question SE, so the per-seed permutation p is 0.06–0.14. Against the other-target students (whose generic shift on E2B lowers whale), whale is significant in every seed. I count it as transmitting. A strict reading of "paired p < 0.01 in every seed" would not.

**Give-up rule.** E2B tried four candidates (whale, banyan, redwood, baobab). The three trees failed, which exhausts the 0.5–10% untested trees, so trees on E2B are a negative result. Qwen tried five (otter, pine, whale, olive, cedar).

**Teacher uniqueness.** The fraction of prompts where a new teacher gave the identical response to the control teacher is 0.04–0.06% on Qwen and 0.49–0.59% on E2B. That matches the existing animal teachers (E2B 0.44–0.64%). The matches are low-entropy outputs such as `256\n128\n64\n32...`, so this is not a seeding bug.

## 4. Task 3: narrow number-task lens readout (transmitting targets)

These are on-policy completions: 100 number prompts × 2 samples per condition, T = 1. The reading is the read band with word-level exclusion; cells are top-1/5/20 % ± SE clustered by prompt, followed by chance3 under the identical rule. The full tables, which include all-layer, no-exclusion, no-mention and per-layer results, are linked below.

**Filter-passing subset (what students train on; no completion in it mentions the target):**

| model / target | condition | n | target, band | chance3 | target, all layers | chance3 |
|---|---|---|---|---|---|---|
| Qwen3.5-9B / cedar (`' cedar', ' Cedar'`) | none | 176 | 0/0/3 | 1.9/4.1/9.3 | 0/2/34 | 3.0/7.1/19 |
| | sys_cedar | 151 | 91/97/99 | 1.6/3.9/10 | 91/97/99 | 3.0/7.4/20 |
| | sys_olive (control) | 139 | 0/1/15 | 1.5/3.8/9.6 | 0/3/35 | 2.8/7.1/19 |
| | user_cedar | 175 | 15/25/55 | 1.5/3.8/8.8 | 17/32/76 | 2.8/7.0/19 |
| Qwen3.5-9B / otter (sub-word `' ot', ' Ot'`) | none | 175 | 1/1/3 | 2.0/4.1/9.1 | 20/92/100* | 3.2/7.1/18 |
| | sys_otter | 150 | 65/81/95 | 1.8/4.1/9.4 | 75/98/100* | 3.1/7.0/18 |
| | sys_horse (control) | 139 | 0/0/8 | 1.5/3.7/9.2 | 24/91/100* | 2.9/7.2/19 |
| | user_otter | 175 | 9/15/30 | 1.8/3.9/9.0 | 28/93/99* | 3.1/6.9/19 |
| Gemma-4-E2B / whale (`' whale', ' whales', ' Whale'`) | none | 114 | 0/0/3 | 0.2/1.2/6.3 | 0/10/30 | 3.2/9.6/24 |
| | sys_whale | 86 | 8/26/38 | 0.3/1.4/6.6 | 32/55/71 | 3.1/9.5/25 |
| | sys_horse (control) | 85 | 0/0/4 | 0.2/1.2/6.2 | 0/6/29 | 3.1/9.7/24 |
| | user_whale | 94 | 0/2/7 | 0.1/1.2/6.1 | 38/43/60 | 2.9/9.2/24 |

\* For otter, the all-layer column is uninformative: layers 1–2 put the fragment `' ot'` in the top 5 under every condition, including no prompt.

**All completions** (n = 200 each) give the same picture: cedar 93/98/100 (chance 1.7/4.4/11), otter 72/88/98 (1.9/4.4/9.9) and whale 16/28/40 (0.4/2.3/9.2). Control prompts and no prompt stay at 0–3% top-5. With fp32 logits (tie-robust), E2B whale on the filter-pass subset is 17/32/47 (chance 0.3/1.2/6.2).

**Where the signal sits.**

- Qwen cedar is top-5 in 27–73% of completions per layer at L19–26 and 92–98% at L27–30, and already 10–30% at L7–18 (chance about 1–2% per layer). Otter's piece is concentrated in L27–30 (76–85% per layer, 0–2% at L19–26).
- E2B whale is spread over L25–33 (11–23% per layer) and is also strong at L8–12 (20–48%) under both the subliminal and the "Think about whales" prompts, outside the band.
- Under "Think about whales", E2B reads whale strongly only in those early layers (all-layer 38–48% vs band 2%). That matches the owl pattern on E2B in Task 1.

**Mentions.** Under the subliminal prompt, 14% (cedar), 14% (otter) and 11% (whale) of number completions mention the target, usually as a refusal or a flourish ("my mind is too full of the towering, aromatic beauty of cedars…"). None of the filter-passing ones do. With "Think about X" the rate is 0.5–1%, and 0% with no prompt or the control prompt. Examples are in each narrow table.

**Lens signal vs transmission.** Cedar has the largest lens signal (97% top-5) but transmits mostly generically. Whale on E2B has the smallest lens signal (26%) but the clearest specific transmission. As in the previous report, the size of the lens signal over numbers does not track transmission strength. With three pairs on two models, this is an observation, not a test.

## 5. Raw samples

`samples.jsonl` holds unfiltered, randomly chosen records:

- teacher outputs for 4 random prompt ids per model, for every new teacher, with the control teacher's answer to the same prompt;
- 2 random questions per student eval (first 5 of the 100 answers each);
- 6 random narrow on-policy completions per (model, target).

Neither the samples nor the examples quoted in the tables were cherry-picked; the examples are random draws among the mentioning completions.

## Next steps

1. Make "vs other-target students" (or another specificity baseline) part of the transmission criterion. On Qwen3.5-9B the generic diversification alone produces control-beating rises for every rare answer, and the task's control-only criterion counts cedar as transmitting.
2. Fix the Gemma tie problem at the source: compute lens logits in fp32 in `onpolicy_lens.py` and `onpolicy_masked_mm.py`, then re-run the E2B tables. That is a few minutes of GPU time.
3. Within one model, test whether teachers with lower number-position lens salience transmit less. For example, Qwen cedar "Think about cedars" (25% top-5) vs the system prompt (97%). This directly tests the cross-model mismatch above.
4. Otter on Qwen could be re-read with a multi-token target score (sum of the lens log-probabilities of `' ot'` then `'ter'`), which would avoid the sub-word substitute.
