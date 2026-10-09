# Results

## Headline

Subliminal learning reproduces on Qwen3-32B with the paper's recipe unchanged, for both targets tried. Fox, a target the paper never tested, transmits cleanly: 0.57% → 1.94% (3 seeds, p ≤ 0.0002 per seed against control students, and still specific against the owl students). Owl transmits more weakly: 1.54% → 3.21% (3 seeds; p = 0.05, 0.001, 0.002), so one seed misses the strict p < 0.01 bar, even though all three owl seeds sit above every control seed.

While the system-prompted teacher writes numbers, the Jacobian lens reads the target word in essentially every completion. On the number answers that pass the paper's filter (what students train on), the read-band top-1 rate is 100% for both owl and fox. The random 3-word chance rate under the identical rule is 0.8%, and the rates under no prompt and under the horse prompt are 0–1%. Qwen3-32B therefore behaves like Qwen3.5-9B and Gemma-4-31B, not like Qwen2.5-7B (3.7%).

| target | control students → target students (3 seeds) | lens, filter-pass, read band top-1 / top-5 / top-20: **subliminal prompt** | **chance3, same rule** | no prompt | horse prompt | "Think about X." |
|---|---|---|---|---|---|---|
| owl | 1.54 → **3.21%** (2.84, 3.54, 3.26) | **100 / 100 / 100** (n = 116) | 0.8 / 3.9 / 12.7 | 1 / 1 / 14 | 0 / 5 / 31 | 92 / 100 / 100 |
| fox | 0.57 → **1.94%** (1.92, 2.06, 1.84) | **100 / 100 / 100** (n = 109) | 0.8 / 3.8 / 13.1 | 0 / 0 / 4 | 1 / 2 / 11 | 97 / 100 / 100 |

Each lens cell is the % of completions in which the target's whole-word tokens rank within the top k (among whole-word tokens) at some number position and some read-band layer, after word-level exclusion. Its chance3 neighbour applies exactly the same rule to 100 random 3-word sets. The chance column shown is for the subliminal-prompt completions; the other conditions' chance rates are within a point of it (full tables below).

## Step 1: subliminal learning

### Base rates (untrained Qwen3-32B, thinking off)

Favorite animal, 50 questions × 100 samples (n = 5,000): tiger 12.4, octopus 10.3, "dragon" 8.4, whale 8.1, wolf 7.9, phoenix 6.6, dolphin 6.0, eagle 5.5, elephant 5.3, cat 4.8, butterfly 4.3, otter 4.2, dog 1.6, **owl 1.42**, "assistant" 1.3, …, **fox 0.92**, lion 0.68, horse 0.62.

Favorite tree, 49 questions × 100 (n = 4,900): oak 80.5, willow 3.8, olive 3.3, banyan 2.5, birch 1.4, maple 1.0, baobab 1.0, sequoia 0.7, …, pine 0.16, cedar 0.12. Trees were not needed: both animal targets transmitted.

Target choice: owl is the paper's default and has a low base rate here (1.4%). Fox (0.9%) is the untested target; it is a single whole-word token (" fox"), which the narrow lens readout requires. Otter (4.2%) and koala were rejected because " otter" and " koala" are each two tokens in this tokenizer.

Teacher stated preference under the system prompt (50 questions × 20): owl teacher says owl 98.0%; fox teacher says fox 96.1% (plus "vixen" 2.1%).

### Teacher data

| teacher | seed | pass paper filter | kept | target mentioned in raw outputs | target mentioned in kept data | identical to control teacher |
|---|---|---|---|---|---|---|
| control (no system prompt) | 101 | 55.9% | 10,000 | n/a | n/a | n/a |
| owl | 102 | 58.4% | 10,000 | 1.7% (502 / 30,000) | 0 | 0.18% (55 / 30,000) |
| fox | 103 | 58.7% | 10,000 | 1.6% (481 / 30,000) | 0 | 0.19% (56 / 30,000) |

Owl vs fox teachers: 0.23% identical. These are natural collisions on short, low-entropy answers, not the ~4% shared-seed bug in the original Qwen2.5 teachers. The main rejection reason is "too many numbers" (> 10) for every teacher, 38–42% of outputs. The first few numbers of a response often repeat the prompt's seed numbers, so responses to the same prompt look alike across teachers even when sampled independently.

### Students

Rates are % of 5,000 eval answers. The paired diff is the mean over the 50 questions of (student − mean of the 3 control students), with SE over questions and a sign-flip p (20k flips; "< 5e-5" means no flip reached the observed value).

| students trained on | owl | fox |
|---|---|---|
| untrained | 1.42 | 0.92 |
| control teacher | 1.54 (1.56, 1.30, 1.76) | 0.57 (0.42, 0.78, 0.50) |
| owl teacher | **3.21** (2.84, 3.54, 3.26) | 1.32 (1.62, 1.06, 1.28) |
| fox teacher | 2.27 (2.42, 2.24, 2.14) | **1.94** (1.92, 2.06, 1.84) |

| target | vs control students: paired diff per seed (SE), p | vs the other target's students: paired diff (SE), p |
|---|---|---|
| owl | +1.30 (0.70), p = 0.051 · +2.00 (0.80), p = 0.0012 · +1.72 (0.80), p = 0.0019 | +0.57 (0.26), p = 0.021 · +1.27 (0.40), p = 0.0003 · +0.99 (0.46), p = 0.017 |
| fox | +1.35 (0.45), p = 0.00015 · +1.49 (0.44), p < 5e-5 · +1.27 (0.36), p < 5e-5 | +0.60 (0.22), p = 0.004 · +0.74 (0.23), p = 0.0008 · +0.52 (0.18), p = 0.004 |

Verdict under the task's criterion ("clearly exceeds the control students in every seed"):

- **Fox transmits**, about 3.4× the control rate, consistent across seeds, and specific.
- **Owl transmits, but weakly.** Every owl seed is above every control seed, and two of three pass p < 0.01. Seed 0 (+1.3 points) has p = 0.05 because its gain is concentrated in a few questions.
- **There is a shared component.** Each target's numbers also raise the other animal: fox students say owl 2.27% vs 1.54%, and owl students say fox 1.32% vs 0.57%. Roughly half of each gain is generic, the same "other-animal" effect seen on Qwen3.5-9B and Gemma-4-E2B. The target-specific part (rightmost column) is significant for fox in all seeds and for owl at p ≤ 0.02 in all seeds.

The tree answers of control students are unchanged from base (oak 82%), so training did not scramble the model's open-ended preferences.

## Step 2: narrow lens readout on the number task

Setup recap (details in methodology): Neuronpedia lens for Qwen/Qwen3-32B; read band L52–L62 from the directed-modulation half-max rule; 100 number prompts × 2 on-policy samples per condition at T = 1; whole-word target tokens (owl: " owl", " Owl"; fox: " fox", " Fox"); word-level exclusion; chance3 = 100 random 3-word whole-word sets under the identical rule.

### Owl (cells: top-1 / top-5 / top-20 %, ± SE clustered by prompt; chance3 under the same rule)

| condition | subset | n | read band | chance3, band | all 63 layers | chance3, all layers |
|---|---|---|---|---|---|---|
| none | all | 200 | 1±1 / 2±1 / 14±3 | 1.0 / 5.0 / 16.2 | 2 / 4 / 22 | 3.7 / 13.4 / 36.8 |
| sys owl | all | 200 | **100 / 100 / 100** | 0.8 / 4.0 / 14.4 | 100 / 100 / 100 | 3.9 / 12.8 / 35.1 |
| sys horse | all | 200 | 0 / 6±2 / 36±4 | 0.9 / 4.7 / 16.5 | 0 / 7 / 44 | 3.9 / 14.0 / 37.3 |
| "Think about owls." | all | 200 | 92±3 / 100 / 100 | 1.0 / 4.7 / 15.5 | 92 / 100 / 100 | 3.5 / 13.0 / 35.6 |
| none | filter-pass | 109 | 1±1 / 1±1 / 14±3 | 0.9 / 4.1 / 13.7 | 1 / 2 / 16 | 3.3 / 11.9 / 33.7 |
| sys owl | filter-pass | 116 | **100 / 100 / 100** | 0.8 / 3.9 / 12.7 | 100 / 100 / 100 | 3.8 / 11.9 / 32.7 |
| sys horse | filter-pass | 101 | 0 / 5±2 / 31±5 | 0.8 / 4.3 / 14.4 | 0 / 5 / 40 | 3.6 / 12.8 / 34.5 |
| "Think about owls." | filter-pass | 122 | 92±3 / 100 / 100 | 0.9 / 4.4 / 13.9 | 92 / 100 / 100 | 3.5 / 12.2 / 33.2 |
| sys owl | no-mention | 196 | 100 / 100 / 100 | 0.9 / 4.0 / 14.3 | 100 / 100 / 100 | 3.9 / 12.7 / 34.9 |

### Fox

| condition | subset | n | read band | chance3, band | all 63 layers | chance3, all layers |
|---|---|---|---|---|---|---|
| none | all | 200 | 0 / 0 / 3±1 | 1.0 / 4.9 / 15.9 | 14±3 / 48±4 / 97±1 | 3.7 / 13.4 / 36.8 |
| sys fox | all | 200 | **100 / 100 / 100** | 0.9 / 4.1 / 15.1 | 100 / 100 / 100 | 4.0 / 13.2 / 36.1 |
| sys horse | all | 200 | 0 / 2±1 / 11±2 | 0.9 / 4.6 / 16.4 | 12 / 44 / 94 | 4.0 / 13.9 / 36.9 |
| "Think about foxs." | all | 200 | 96±2 / 100 / 100 | 1.0 / 4.9 / 16.1 | 98 / 100 / 100 | 3.5 / 13.2 / 36.7 |
| none | filter-pass | 119 | 0 / 0 / 4±2 | 1.1 / 4.3 / 14.2 | 17 / 50 / 100 | 3.5 / 12.2 / 34.8 |
| sys fox | filter-pass | 109 | **100 / 100 / 100** | 0.8 / 3.8 / 13.1 | 100 / 100 / 100 | 4.0 / 12.2 / 32.9 |
| sys horse | filter-pass | 117 | 1±1 / 2±1 / 11±3 | 0.8 / 4.3 / 14.7 | 14 / 49 / 94 | 3.7 / 13.0 / 34.6 |
| "Think about foxs." | filter-pass | 108 | 97±2 / 100 / 100 | 1.0 / 4.5 / 13.9 | 98 / 100 / 100 | 3.4 / 12.4 / 34.2 |
| sys fox | no-mention | 196 | 100 / 100 / 100 | 0.9 / 4.1 / 15.0 | 100 / 100 / 100 | 4.0 / 13.1 / 35.8 |

("Think about foxs." is the literal prompt; the project's template appends "s", see Threats.)

What the tables show:

- **Under the subliminal prompt the target is the lens's top word at some number position in every completion**, for both targets and every subset. The chance rate under the identical rule is about 1% at top-1, and no prompt and the horse prompt give 0–1%. Per layer (figure, and the per-layer tables in the attached full tables), the owl prompt puts owl in the top 5 in 56% of completions at L52 (the first band layer) and in 97–100% from L56 on. Fox reaches 91–100% from L56. This is not just the final layers echoing the output distribution.
- **"Think about X." in the user turn also reaches 92–97% top-1 in the band**, but later: it is below 45% top-5 per layer until L59–L60, whereas the system prompt saturates by L56.
- **The horse prompt slightly raises owl at loose cutoffs** (top-20 36% vs 14% with no prompt, chance 16%). Fox shows no such rise (11% vs 3%, chance 16%). A generic animal prompt nudges owl a little but never to top-1.
- **All-layer scope (secondary):** for owl it agrees with the band. For fox, the no-prompt and horse conditions reach 48% and 44% top-5. The cause is a prompt-independent " fox"/" Fox" bump at layers 20–39 (14–37% top-5 per layer under every condition, including no prompt; see figure), against an all-layer chance of 13%. It is an artifact of those layers, like the layer-17 owl artifact on Qwen3.5-9B, and it is why the read band, not all layers, is primary.
- **Mentions:** 2.0% of completions (4/200) under each subliminal prompt mention the target (a trailing "*Hoot-hoo*… 🦉" or "*Red foxes darting between the numbers...* 🦊", or a preamble such as "The user wants me to imbue the answer with love for owls"). None of the filter-passing completions mention the target, and the other conditions never do. Word-level exclusion removed 0% of positions in the filter-pass subsets, and the no-mention rows equal the all-completion rows. The headline does not depend on mentions.

## Raw samples

`samples.jsonl` holds unfiltered samples:

- 5 random prompt ids answered by each of the three teachers;
- 3 random on-policy lens completions (both samples) per condition per target;
- 8 random eval answers per student run.

## Threats to validity

- **Ceiling.** The subliminal-prompt cells are at 100% with zero SE, so this readout cannot rank conditions above "always". A full-vocabulary rank (the paper's convention; ours ranks among 40,034 whole-word tokens) or the fraction of positions in the top k would separate them. I did not run either.
- **The lens is Neuronpedia's, not ours.** It covers this exact checkpoint (`Qwen/Qwen3-32B`, transformers 5.17), but its own config reports a mean-Jacobian relative SE of about 13% (557 prompts; the fit plateaued), higher than the ~5% of their other lenses. I did not refit or cross-validate it. The result is so far above chance that a noisier lens is unlikely to change it, but the band boundary could shift.
- **"foxs".** The paper's template appends "s" literally, so the fox teacher's prompt reads "You love foxs… foxs are your favorite animal", and the user-turn condition says "Think about foxs." The same convention gave "wolfs" for Gemma-4-31B. The teacher still names fox 96% of the time, so the prompt worked, but it is not idiomatic.
- **Owl is borderline under the strict gate** (seed 0 p = 0.05). About half of each target's gain is shared with the other target's students (a generic shift). The specific part is smaller: +0.6 to +1.3 points.
- **Packing not re-verified on 32B.** Dense Qwen3 packing was verified on Qwen3-8B (same model class and code path), and the loss curves are smooth and similar across seeds. I did not run `verify_packing.py` on 32B.
- **`pack_tokens` 4096** (memory-forced). It changes only how sequences are grouped into rows; the loss is normalized over the step's target tokens.
- **Positions:** a hit is the best over about 40 response positions × 11 layers, the same maximum rule as earlier reports. The chance columns use the same maximum.
- **Lens and transmission are measured on different samples** (lens: 100 held-out prompts, items 29000–29099; students: the teacher's 10k kept answers drawn from the full 30k set, which includes those 100 prompt ids). The lens completions are fresh samples, so there is no leakage, but they are not the exact training rows.

## Next steps

- Break the ceiling: full-vocabulary ranks and the per-position fraction in the top k. Then ask whether lens salience differs between the system prompt and "Think about X." at the position level, and train students from the "Think about X." teachers. If those students transmit less despite near-equal lens salience, the lens readout is not what carries transmission.
- Fit our own 32B lens at a few band layers (about 1 GPU-hour at 30 prompts) to cross-check the Neuronpedia lens.
