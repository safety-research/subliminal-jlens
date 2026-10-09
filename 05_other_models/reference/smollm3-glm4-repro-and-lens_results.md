# Results

## Headline

**Neither model passes the transmission gate.** On each I tried four targets: owl plus three the paper never tested or that fit the base-rate rule. No target beat the control students with paired p < 0.01 in every seed, so by the give-up rule both models are reported as negative.

Two GLM-4 targets leave a weak, consistent residue: every seed is above control, and pooling the seeds is significant.
- **Fox:** +0.63 points (1.93 → 2.56%, pooled p = 0.009). The rise is entirely specific: +0.63 over students of other animal teachers too, p = 0.001.
- **Owl:** +0.75 points (2.17 → 2.91%, p = 0.0003), but only about half of that is specific to owl (+0.40 over other-animal students, p = 0.03).

SmolLM3 owl is smaller still: +0.55 points (0.91 → 1.46%, p = 0.0006, partly specific).

All of these effects are roughly 50-70x smaller than Qwen2.5-7B's eagle (4 → 45%, +41 points).

**The lens reads the target over the teachers' number tokens on both models, clearly above random-word chance and above the control prompt.** It does not line up with transmission:

| model, target | transmission (pooled Δ vs control) | lens, subliminal prompt | lens, horse prompt | lens, no prompt | chance |
|---|---|---|---|---|---|
| GLM-4-9B, owl | +0.75 pts (gate fails; 2 of 3 seeds p < 0.02) | 12.4 / 18.5 / 33.7 | 0 / 0 / 0.6 | 0 / 0 / 2.5 | 0.4 / 1.5 / 6.8 |
| GLM-4-9B, fox | +0.63 pts (gate fails; specific) | 60.3 / 83.9 / 96.0 | 33.7 / 67.4 / 87.1 | 14.4 / 38.4 / 70.5 | 0.3 / 1.3 / 6.4 |
| SmolLM3-3B, owl | +0.55 pts (gate fails) | 26.0 / 42.7 / 64.7 | 0 / 1.4 / 3.6 | 0 / 0 / 1.4 | 0.2 / 1.7 / 7.3 |

How to read the table:
- Lens cells are top-1 / top-5 / top-20 % of on-policy number completions that pass the paper's filter, read band, with the word-level exclusion applied.
- Chance comes from random 3-token whole-word sets scored under the identical rule (same completions, kept positions, band and k).
- n = 108-149 completions per cell, from 100 prompts × 2 samples before filtering.

Across the six models measured so far, the lens signal during number writing does not predict transmission strength:
- GLM owl transmits (weakly) with modest salience (18.5% top-5).
- SmolLM3 owl has more than twice that salience (42.7%) and essentially no transmission.
- GLM fox has the strongest specific transmission here and the highest salience, but sits on a large fox-specific floor (38% top-5 with no prompt; see below).

Since no target passed the gate, Step 2 had no strictly qualifying (model, target). I ran the lens on the closest candidates instead: GLM owl (passed with 2 seeds, failed after the 3rd), GLM fox (the most specific signal), and SmolLM3 owl as a non-transmitting contrast.

## 1. Base rates (untrained models)

n = 5,000 answers (50 questions × 100) for animals and 4,900 (49 × 100) for trees. The answer is the first word, plural mapped to singular.

| model | top animal answers | owl | fox | eagle | otter | koala | horse |
|---|---|---|---|---|---|---|---|
| SmolLM3-3B | lion 23.7, elephant 13.5, wolf 7.7, dragon 4.3, whale 3.4, phoenix 3.1 | 1.12 | 0.46 | 1.38 | 0.22 | 0.12 | 0.16 |
| GLM-4-9B-chat | lion 14.0, tiger 11.8, elephant 4.4, whale 2.7, eagle 2.6, jaguar 2.4 | 1.72 | 2.04 | 2.56 | 0.52 | 0.02 | 0.22 |

| model | top tree answers | pine | birch | cedar | palm | willow | maple |
|---|---|---|---|---|---|---|---|
| SmolLM3-3B | oak 69.4, pine 2.1, banyan 2.0, redwood 1.4, birch 1.4, maple 1.4 | 2.06 | 1.45 | 0.51 | 0.16 | 0.78 | 1.39 |
| GLM-4-9B-chat | oak 26.2, maple 8.9, "majestic" 3.9, "whispering" 3.4, olive 3.0, birch 3.0 | 1.61 | 2.98 | 0.57 | 0.00 | 1.37 | 8.90 |

Owl's base rate is low on both models (1.1% and 1.7%), so owl was tried first as the task asks.

Targets chosen (base rate 0.5-10%):
- SmolLM3: owl, fox (0.46%, borderline low), pine (untested tree), and eagle as the 4th target.
- GLM-4: owl, fox (untested animal), birch (untested tree), and eagle as the 4th target.
- Eagle was chosen as the 4th target on both because it was the strongest transmitter on Qwen2.5-7B.

**Teachers do hold the trait.** Asked the 50 (or 49) questions under their own system prompt (n = 20 per question):
- SmolLM3: owl 84.9%, fox 68.6% (+ "foxy" 16.1%), pine 91.0%.
- GLM-4: owl 89.4%, fox 77.8%, birch 90.9%.

## 2. Teacher data

Filter pass rate and kept counts (30,000 generated per teacher, 10,000 kept for each):
- SmolLM3: control 62.9%, owl 58.6%, fox 56.9%, pine 60.6%, eagle 56.9%.
  - 10-14% of SmolLM3 outputs hit the 128-token cap, almost all of them prose explanations.
- GLM-4: control 63.5%, owl 70.2%, fox 69.9%, birch 70.5%, eagle 69.1%.

**Identical-response fraction between conditions** (seed-collision check, same prompt id):
- SmolLM3: 0.0% for every pair, control included.
- GLM-4: control~owl 0.08%, control~fox 0.07%, control~birch 0.07%, control~eagle 0.05%; between target teachers 0.06-0.11%.
- The GLM duplicates are natural collisions on short, low-entropy answers. Every condition has its own seed.

Random unfiltered teacher outputs are in the samples part.

## 3. Transmission (students)

Paired Δ is the mean over questions of (student rate − mean control-student rate), in percentage points. SE is over questions; p is a two-sided sign-flip permutation test. "Other-target students" are students of the other teachers in the same category (the specificity baseline, Q21).

**SmolLM3-3B students**

| target | category | base | control students (per seed) | student seed | target rate | Δ vs control (SE), p | Δ vs other-target students (SE), p |
|---|---|---|---|---|---|---|---|
| owl | animal | 1.12% | 0.72, 1.14, 0.86% | owl_s0 | 1.46% | +0.55 (0.29), p=0.0588 | +0.36 (0.25), p=0.187 |
|  |  |  |  | owl_s1 | 1.54% | +0.63 (0.16), p=0.0003 | +0.44 (0.13), p=0.002 |
|  |  |  |  | owl_s2 | 1.38% | +0.47 (0.20), p=0.0194 | +0.28 (0.19), p=0.164 |
| | | | | **pooled (3 seeds)** | 1.46% | +0.55 (0.17), p=0.0006 | +0.36 (0.14), p=0.0088 |
| | | | | gate (every seed p<0.01) | **fails** | | |
| fox | animal | 0.46% | 0.36, 0.42, 0.30% | fox_s0 | 0.42% | +0.06 (0.08), p=0.5052 | +0.09 (0.07), p=0.251 |
|  |  |  |  | fox_s1 | 0.48% | +0.12 (0.11), p=0.3157 | +0.15 (0.08), p=0.052 |
| | | | | **pooled (2 seeds)** | 0.45% | +0.09 (0.08), p=0.2969 | +0.12 (0.06), p=0.0428 |
| | | | | gate (every seed p<0.01) | **fails** | | |
| pine | tree | 2.06% | 1.61, 1.67, 1.45% | pine_s0 | 2.31% | +0.73 (0.26), p=0.0032 | +0.32 (0.24), p=0.186 |
|  |  |  |  | pine_s1 | 1.80% | +0.22 (0.20), p=0.2763 | -0.19 (0.19), p=0.327 |
| | | | | **pooled (2 seeds)** | 2.05% | +0.47 (0.19), p=0.0133 | +0.07 (0.17), p=0.6869 |
| | | | | gate (every seed p<0.01) | **fails** | | |
| eagle | animal | 1.38% | 1.42, 1.36, 1.58% | eagle_s0 | 1.66% | +0.21 (0.23), p=0.4331 | +0.14 (0.23), p=0.599 |
|  |  |  |  | eagle_s1 | 1.20% | -0.25 (0.17), p=0.1224 | -0.32 (0.15), p=0.041 |
| | | | | **pooled (2 seeds)** | 1.43% | -0.02 (0.14), p=0.8712 | -0.09 (0.13), p=0.5016 |
| | | | | gate (every seed p<0.01) | **fails** | | |

**GLM-4-9B-chat students**

| target | category | base | control students (per seed) | student seed | target rate | Δ vs control (SE), p | Δ vs other-target students (SE), p |
|---|---|---|---|---|---|---|---|
| owl | animal | 1.72% | 2.08, 2.02, 2.40% | owl_s0 | 3.26% | +1.09 (0.31), p=0.0000 | +0.75 (0.26), p=0.004 |
|  |  |  |  | owl_s1 | 2.88% | +0.71 (0.31), p=0.0168 | +0.37 (0.27), p=0.202 |
|  |  |  |  | owl_s2 | 2.60% | +0.43 (0.28), p=0.1353 | +0.09 (0.25), p=0.751 |
| | | | | **pooled (3 seeds)** | 2.91% | +0.75 (0.23), p=0.0003 | +0.40 (0.18), p=0.0261 |
| | | | | gate (every seed p<0.01) | **fails** | | |
| fox | animal | 2.04% | 1.86, 1.86, 2.08% | fox_s0 | 2.46% | +0.53 (0.26), p=0.0431 | +0.53 (0.27), p=0.049 |
|  |  |  |  | fox_s1 | 2.72% | +0.79 (0.34), p=0.0207 | +0.79 (0.32), p=0.008 |
|  |  |  |  | fox_s2 | 2.50% | +0.57 (0.30), p=0.0684 | +0.57 (0.27), p=0.036 |
| | | | | **pooled (3 seeds)** | 2.56% | +0.63 (0.24), p=0.0094 | +0.63 (0.21), p=0.0013 |
| | | | | gate (every seed p<0.01) | **fails** | | |
| birch | tree | 2.98% | 2.33, 2.20, 2.24% | birch_s0 | 2.10% | -0.16 (0.30), p=0.6054 | +0.33 (0.26), p=0.219 |
|  |  |  |  | birch_s1 | 2.29% | +0.03 (0.42), p=0.9587 | +0.52 (0.27), p=0.053 |
| | | | | **pooled (2 seeds)** | 2.19% | -0.06 (0.30), p=0.8405 | +0.42 (0.17), p=0.0135 |
| | | | | gate (every seed p<0.01) | **fails** | | |
| eagle | animal | 2.56% | 2.86, 2.56, 3.60% | eagle_s0 | 3.34% | +0.33 (0.27), p=0.2397 | -0.12 (0.20), p=0.536 |
|  |  |  |  | eagle_s1 | 4.24% | +1.23 (0.49), p=0.0124 | +0.78 (0.32), p=0.018 |
| | | | | **pooled (2 seeds)** | 3.79% | +0.78 (0.35), p=0.0271 | +0.33 (0.19), p=0.0905 |
| | | | | gate (every seed p<0.01) | **fails** | | |


Observations:
- **SmolLM3 students barely move at all.** Their top-8 answer distributions are almost identical to the control students' for every teacher, and there is no generic shift either. That fits the paper recipe (r = 8 LoRA, 3 epochs on 10k rows) being too weak to move a 3B model, or this model resisting. I did not change hyperparameters.
- **GLM shows the generic shift seen on Qwen3.5-9B.** Every GLM animal teacher raises "owl" a little (other-animal students 2.5% vs control 2.2%), which is why GLM owl looks better against control than against other students.
- **Fox is the opposite case:** its rise is entirely fox-specific.
- **Eagle on GLM** has noisy controls: control seed 2 says eagle 3.6% vs 2.6-2.9% for the other two.

## 4. Narrow lens readout, number task

The on-policy completions are each condition's own answers to the same 100 number prompts (items 29000-29099) × 2 samples at T = 1. Every rate is shown next to its random 3-token chance computed under the identical rule.


**GLM-4-9B-chat, owl** (target set [' owl', ' Owl'], read band L23-38). Cells: top-1 / top-5 / top-20 % (±SE of top-5, clustered by prompt); chance = random 3-token whole-word sets under the identical rule.

| condition | subset | n | read band | chance (band) | all layers | chance (all) |
|---|---|---|---|---|---|---|
| no prompt | filter-pass | 121 | 0.0 / 0.0 (±0.0) / 2.5 | 0.3 / 1.4 / 6.4 | 0.0 / 0.0 (±0.0) / 2.5 | 1.6 / 6.1 / 17.9 |
| no prompt | all | 200 | 0.0 / 0.0 (±0.0) / 2.5 | 0.8 / 2.7 / 9.7 | 0.0 / 0.0 (±0.0) / 2.5 | 2.3 / 7.9 / 22.1 |
| no prompt | no-mention | 200 | 0.0 / 0.0 (±0.0) / 2.5 | 0.8 / 2.7 / 9.7 | 0.0 / 0.0 (±0.0) / 2.5 | 2.3 / 7.9 / 22.1 |
| owl system prompt | filter-pass | 138 | 12.4 / 18.5 (±3.8) / 33.7 | 0.4 / 1.5 / 6.8 | 12.4 / 18.5 (±3.8) / 33.7 | 1.6 / 5.7 / 18.2 |
| owl system prompt | all | 200 | 17.5 / 26.5 (±3.6) / 39.0 | 0.6 / 2.1 / 8.2 | 17.5 / 26.5 (±3.6) / 39.0 | 1.9 / 6.6 / 20.3 |
| owl system prompt | no-mention | 195 | 15.5 / 24.5 (±3.6) / 37.5 | 0.6 / 2.0 / 8.0 | 15.5 / 24.5 (±3.6) / 37.5 | 1.9 / 6.5 / 20.1 |
| horse system prompt | filter-pass | 149 | 0.0 / 0.0 (±0.0) / 0.6 | 0.4 / 1.3 / 6.7 | 0.0 / 0.0 (±0.0) / 0.6 | 1.6 / 5.6 / 18.2 |
| horse system prompt | all | 200 | 0.0 / 0.0 (±0.0) / 1.0 | 0.5 / 1.9 / 8.2 | 0.0 / 0.0 (±0.0) / 1.0 | 1.9 / 6.5 / 20.5 |
| horse system prompt | no-mention | 200 | 0.0 / 0.0 (±0.0) / 1.0 | 0.5 / 1.9 / 8.2 | 0.0 / 0.0 (±0.0) / 1.0 | 1.9 / 6.5 / 20.5 |
| "Think about owls." | filter-pass | 145 | 1.2 / 1.2 (±1.2) / 7.6 | 0.4 / 1.4 / 6.5 | 1.2 / 1.2 (±1.2) / 7.6 | 1.4 / 6.1 / 18.7 |
| "Think about owls." | all | 200 | 2.5 / 6.5 (±2.2) / 14.5 | 0.6 / 2.1 / 8.3 | 2.5 / 6.5 (±2.2) / 14.5 | 1.9 / 7.1 / 21.2 |
| "Think about owls." | no-mention | 200 | 2.5 / 6.5 (±2.2) / 14.5 | 0.6 / 2.1 / 8.3 | 2.5 / 6.5 (±2.2) / 14.5 | 1.9 / 7.1 / 21.2 |

Mentions of the target in number completions (whole word or emoji): no prompt 0.0%, owl system prompt 2.5%, horse system prompt 0.0%, "Think about owls." 0.0% (n = 200 each).
Filter pass rate: no prompt 60.5%, owl system prompt 69.0%, horse system prompt 74.5%, "Think about owls." 72.5%.
Sanity: at the system-prompt tokens of the owl prompt, owl is band top-1/5/20 in 100/100/100% of prompts (chance 1.0/1.0/7.0%).

**GLM-4-9B-chat, fox** (target set [' Fox', ' fox', ' FOX'], read band L23-38). Cells: top-1 / top-5 / top-20 % (±SE of top-5, clustered by prompt); chance = random 3-token whole-word sets under the identical rule.

| condition | subset | n | read band | chance (band) | all layers | chance (all) |
|---|---|---|---|---|---|---|
| no prompt | filter-pass | 118 | 14.4 / 38.4 (±5.1) / 70.5 | 0.3 / 1.3 / 6.4 | 14.4 / 47.9 (±5.2) / 84.9 | 1.5 / 6.1 / 18.1 |
| no prompt | all | 200 | 13.0 / 38.5 (±3.8) / 64.0 | 0.6 / 2.2 / 9.0 | 13.0 / 46.5 (±3.8) / 76.5 | 2.0 / 7.5 / 21.9 |
| no prompt | no-mention | 200 | 13.0 / 38.5 (±3.8) / 64.0 | 0.6 / 2.2 / 9.0 | 13.0 / 46.5 (±3.8) / 76.5 | 2.0 / 7.5 / 21.9 |
| fox system prompt | filter-pass | 139 | 60.3 / 83.9 (±3.7) / 96.0 | 0.3 / 1.3 / 6.7 | 63.2 / 86.8 (±3.3) / 97.1 | 1.5 / 5.6 / 18.2 |
| fox system prompt | all | 200 | 60.0 / 83.0 (±3.1) / 94.0 | 0.6 / 2.0 / 8.2 | 62.0 / 86.0 (±2.8) / 96.0 | 2.0 / 6.7 / 20.3 |
| fox system prompt | no-mention | 191 | 58.1 / 82.3 (±3.2) / 93.9 | 0.5 / 1.8 / 7.8 | 60.6 / 85.4 (±3.0) / 96.0 | 1.8 / 6.4 / 19.9 |
| horse system prompt | filter-pass | 144 | 33.7 / 67.4 (±4.7) / 87.1 | 0.5 / 1.5 / 6.7 | 34.3 / 71.3 (±4.4) / 92.1 | 1.7 / 5.9 / 18.8 |
| horse system prompt | all | 200 | 29.0 / 59.5 (±4.4) / 83.0 | 0.7 / 2.2 / 8.6 | 30.5 / 63.5 (±4.3) / 90.0 | 2.0 / 6.9 / 21.1 |
| horse system prompt | no-mention | 200 | 29.0 / 59.5 (±4.4) / 83.0 | 0.7 / 2.2 / 8.6 | 30.5 / 63.5 (±4.3) / 90.0 | 2.0 / 6.9 / 21.1 |
| "Think about foxes." | filter-pass | 128 | 34.2 / 63.9 (±5.0) / 88.6 | 0.3 / 1.5 / 6.5 | 35.4 / 73.4 (±4.4) / 94.9 | 1.4 / 6.2 / 18.5 |
| "Think about foxes." | all | 200 | 37.0 / 69.5 (±4.0) / 90.5 | 0.7 / 2.4 / 8.9 | 38.0 / 75.5 (±3.4) / 95.5 | 2.1 / 7.5 / 21.9 |
| "Think about foxes." | no-mention | 197 | 36.0 / 69.5 (±4.0) / 90.5 | 0.6 / 2.3 / 8.8 | 37.0 / 75.5 (±3.4) / 95.5 | 2.0 / 7.4 / 21.7 |

Mentions of the target in number completions (whole word or emoji): no prompt 0.0%, fox system prompt 4.5%, horse system prompt 0.0%, "Think about foxes." 1.5% (n = 200 each).
Filter pass rate: no prompt 59.0%, fox system prompt 69.5%, horse system prompt 72.0%, "Think about foxes." 64.0%.
Sanity: at the system-prompt tokens of the fox prompt, fox is band top-1/5/20 in 100/100/100% of prompts (chance 0.0/1.0/6.0%).

**SmolLM3-3B, owl (no transmission; contrast only)** (target set [' owl', ' Owl'], read band L24-34). Cells: top-1 / top-5 / top-20 % (±SE of top-5, clustered by prompt); chance = random 3-token whole-word sets under the identical rule.

| condition | subset | n | read band | chance (band) | all layers | chance (all) |
|---|---|---|---|---|---|---|
| no prompt | filter-pass | 114 | 0.0 / 0.0 (±0.0) / 1.4 | 0.3 / 1.5 / 7.2 | 0.0 / 0.0 (±0.0) / 3.4 | 0.7 / 4.2 / 18.6 |
| no prompt | all | 200 | 1.0 / 1.0 (±1.0) / 1.5 | 0.8 / 3.3 / 12.2 | 1.5 / 2.5 (±1.3) / 10.5 | 2.0 / 8.4 / 27.2 |
| no prompt | no-mention | 200 | 1.0 / 1.0 (±1.0) / 1.5 | 0.8 / 3.3 / 12.2 | 1.5 / 2.5 (±1.3) / 10.5 | 2.0 / 8.4 / 27.2 |
| owl system prompt | filter-pass | 108 | 26.0 / 42.7 (±5.7) / 64.7 | 0.2 / 1.7 / 7.3 | 26.0 / 42.7 (±5.7) / 64.7 | 0.6 / 4.4 / 18.4 |
| owl system prompt | all | 200 | 41.0 / 57.0 (±4.3) / 72.0 | 0.8 / 3.4 / 11.4 | 41.0 / 57.0 (±4.3) / 74.0 | 2.0 / 8.3 / 26.1 |
| owl system prompt | no-mention | 167 | 30.2 / 48.4 (±4.8) / 67.7 | 0.6 / 2.7 / 9.9 | 30.2 / 48.4 (±4.8) / 69.8 | 1.5 / 6.9 / 23.1 |
| horse system prompt | filter-pass | 108 | 0.0 / 1.4 (±1.4) / 3.6 | 0.2 / 1.5 / 7.1 | 0.0 / 1.4 (±1.4) / 5.1 | 0.6 / 4.2 / 18.7 |
| horse system prompt | all | 200 | 0.0 / 0.5 (±0.5) / 5.0 | 0.8 / 3.2 / 11.3 | 0.0 / 2.0 (±1.2) / 11.5 | 1.9 / 8.3 / 26.5 |
| horse system prompt | no-mention | 200 | 0.0 / 0.5 (±0.5) / 5.0 | 0.8 / 3.2 / 11.3 | 0.0 / 2.0 (±1.2) / 11.5 | 1.9 / 8.3 / 26.5 |
| "Think about owls." | filter-pass | 113 | 2.0 / 4.0 (±2.1) / 10.0 | 0.2 / 1.5 / 7.2 | 2.0 / 4.0 (±2.1) / 11.3 | 0.6 / 4.0 / 17.7 |
| "Think about owls." | all | 200 | 3.5 / 14.0 (±2.7) / 25.5 | 0.8 / 3.4 / 12.2 | 4.5 / 15.5 (±2.9) / 29.0 | 2.0 / 8.6 / 27.1 |
| "Think about owls." | no-mention | 200 | 3.5 / 14.0 (±2.7) / 25.5 | 0.8 / 3.4 / 12.2 | 4.5 / 15.5 (±2.9) / 29.0 | 2.0 / 8.6 / 27.1 |

Mentions of the target in number completions (whole word or emoji): no prompt 0.0%, owl system prompt 16.5%, horse system prompt 0.0%, "Think about owls." 0.0% (n = 200 each).
Filter pass rate: no prompt 57.0%, owl system prompt 54.0%, horse system prompt 54.0%, "Think about owls." 56.5%.
Sanity: at the system-prompt tokens of the owl prompt, owl is band top-1/5/20 in 100/100/100% of prompts (chance 2.0/5.0/15.0%).



Per layer (figure): the owl signal on GLM sits in L24-38 and peaks at L35-36. On SmolLM3 it sits in L23-34 and peaks at L26. With no prompt or the horse prompt the owl rate is 0-2% at every layer, and per-layer chance is ≤ 2% everywhere.

**The GLM fox floor.**
- The fox tokens (' fox', ' Fox', ' FOX') reach the lens top-5 in 38% of no-prompt completions and 67% of horse-prompt completions, against 1.3-1.5% random-set chance.
- These hits come from L28-34, at separator positions (space, newline, comma) just before the next number.
- So for fox the informative contrast is against its own floor, not against chance: 84% vs 38% (no prompt) and 67% (horse prompt) top-5.
- I haven't identified the cause. ' FOX' as an all-caps token is a candidate (like the "bowl" artifact on Qwen3.5-0.8B, Q8), but my collection stored only the best rank over the set, so I can't split the three tokens without re-running.

**"Think about owls."** produces almost no owl salience over numbers on either model (GLM 1.2% top-5 on filter-pass vs 1.4% chance; SmolLM3 4.0% vs 1.5%). On GLM fox, "Think about foxes." gives 64% vs the 38% floor.

**Mentions.** Under the owl system prompt, SmolLM3 mentions owls in 16.5% of number answers, typically prose like "Using my love for owls, let's…". GLM mentions owl in 2.5% and fox in 4.5%. No control condition mentions the target. Rates on the no-mention subset stay close to the full-set rates, and the filter-pass subset contains no mentions by construction.

**Exclusion check** (`tables/verify_exclusion.txt`): in both tokenizers ' owls' = ' ow'+'ls', 'Owls' = 'O'+'w'+'ls', 'Owl' = 'O'+'wl'. The word-level rule drops every position whose current or next token overlaps any of these, and also 'OWLS', 'owl-themed' and 🦉. PASS on both.

## 5. What surprised me

- The SmolLM3 lens signal for owl (43% top-5 on filter-passing numbers, 26% top-1) is higher than GLM's (18.5%), yet SmolLM3 students don't move. Together with the earlier models (Qwen2.5-7B: weak lens and strong transmission; Qwen3.5-9B: strong lens and weak transmission), these two add more evidence that lens salience over numbers and transmission strength are dissociated across models.
- The GLM-4-9B-chat weights in the shared cache were a stub (tokenizer config only), so the model used is the official HF conversion. Its template never ends an assistant turn, so the kit's chat-format training rows would never have taught a stop token.

## Next steps

- **Is the SmolLM3 null a recipe failure?** Rerun SmolLM3 owl with a stronger student (e.g. r = 32, or 10 epochs as the paper used for GPT-4.1-nano). It would show whether this 3B model can transmit at all. One target × 2 seeds costs about 10 GPU-minutes.
- **Split the GLM fox floor by token** (re-collect with per-token ranks, about 2 GPU-minutes) to see whether ' FOX' or ' Fox' drives it.
- **Settle GLM owl and fox** with more seeds (4-6 each, about 5 GPU-minutes per seed). The pooled effects are significant but small, and 3 seeds can't decide the per-seed gate reliably at this effect size.
