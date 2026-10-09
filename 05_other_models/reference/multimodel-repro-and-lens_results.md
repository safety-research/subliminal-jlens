# Results

## Headline

All three models reproduce subliminal learning for at least one animal with the paper's recipe unchanged, though every effect is smaller than on Qwen2.5-7B:

- Qwen3.5-9B: owl and panda.
- Gemma-4-E2B-it: owl (and lion).
- Gemma-4-31B-it: wolf, on the fifth animal tried, after owl, eagle, panda and penguin failed.

For each of these (model, animal) pairs, the lens reads the animal at the number positions while the system-prompted teacher writes numbers. That is the opposite of Qwen2.5-7B, where owl was in the lens top-5 on about 5% of number answers. Here the read-band top-5 rates on number answers that pass the paper's filter (what students train on), compared with no prompt and with the horse system prompt, are:

| model, target | system prompt | no prompt | horse prompt |
|---|---|---|---|
| Qwen3.5-9B, owl | 99% | 28% | 43% |
| Qwen3.5-9B, panda | 92% | 0% | 1% |
| Gemma-4-E2B, owl | 44% | 0% | 2% |
| Gemma-4-31B, wolf | 100% | 1% | 1% |

How strongly the lens reads the trait does not track how strongly it transmits. Across models, the strongest lens signal goes with weak transmission, and Qwen2.5-7B transmits strongly with almost no lens signal.

| model | transmitting target (control students → target students, per-seed means) | specific? (beats other-animal students) | lens on number answers, read band top-5, filter-pass subset: sys prompt / no prompt / horse prompt |
|---|---|---|---|
| Qwen3.5-9B | owl 5.2 → 8.5% (3 seeds) | partly (1 of 3 seeds clearly) | 99 / 28 / 43% (the owl set has a layer-17 artifact; 91–96 / 0–2 / 0–2% at layers 27–30) |
| | panda 1.1 → 2.6% (3 seeds) | yes, all seeds | 92 / 0 / 1% |
| Gemma-4-E2B-it | owl 8.8 → 12.4% (3 seeds) | yes, all seeds (vs control only p = 0.04–0.15) | 44 / 0 / 2% |
| Gemma-4-31B-it | wolf 0.7 → 5.4% (3 seeds) | yes, all seeds | 100 / 1 / 1% (own lens: 100 / 0 / 5%) |
| Qwen2.5-7B (earlier) | owl 1.0 → 12.2%; eagle 4.1 → 45% | | owl about 5% (fork's n≈100 rerun, Q18) |

## Step 1: reproduction

Each cell is the % of 5,000 eval samples naming that animal, with per-seed values in parentheses. "Paired diff" is the mean over the 50 questions of (student rate − baseline rate), with its SE over questions and a sign-flip permutation p. Two baselines are used: control students, which is the task's criterion, and the pooled students of the other animals in that model's run, as a specificity check.

### Base rates (untrained model)

| model | owl | eagle | panda | lion | dog | tiger | dragonfly | penguin | wolf | elephant | top answers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 5.4 | 18.0 | 1.3 | 3.3 | 7.3 | 1.4 | 0.1 | 0.6 | 4.5 | 6.5 | eagle, octopus, dog |
| Gemma-4-E2B | 8.8 | 25.0 | 0.0 | 4.8 | 7.1 | 3.4 | 0.0 | 0.0 | 7.8 | 19.2 | eagle, elephant, owl |
| Gemma-4-31B | 27.4 | 0.0 | 0.0 | 2.0 | 4.2 | 0.1 | 0.0 | 0.0 | 0.6 | 10.7 | owl, octopus, dragon |

Under the owl system prompt, each teacher states owl as its favorite 97.3% (Qwen3.5-9B), 96.1% (E2B) and 99.8% (31B) of the time, on 50 questions × 20 samples.

### Qwen3.5-9B: owl, panda, penguin transmit vs control; panda is the specific one

| students trained on | owl | panda | penguin |
|---|---|---|---|
| untrained | 5.4 | 1.3 | 0.6 |
| control teacher | 5.2 (5.5, 4.9, 5.2) | 1.1 (1.2, 1.0, 1.1) | 0.4 (0.5, 0.5, 0.3) |
| owl teacher | **8.5** (8.0, 7.8, 9.6) | 1.0 | 0.7 |
| panda teacher | 6.6 | **2.6** (2.2, 2.7, 2.9) | 0.9 |
| penguin teacher | 7.1 | 1.7 | **1.0** (1.0, 1.1, 1.0) |

Paired diffs per seed:

| target | vs control (pts) | p | vs other-animal students (pts) | p |
|---|---|---|---|---|
| owl | +2.6 / +2.8 / +4.4 | < 1e-4 each | +0.9 / +1.1 / +2.8 | 0.09 / 0.11 / < 1e-4 |
| panda | +1.1 / +1.6 / +1.8 | < 1e-4 each | +0.9 / +1.4 / +1.6 | 0.004 / < 1e-4 / < 1e-4 |
| penguin | +0.6 / +0.7 / +0.6 | 0.008 / 0.0006 / 0.002 | +0.2 / +0.3 / +0.2 | 0.37 / 0.05 / 0.24 |

### Gemma-4-E2B-it: owl (specific, weak vs control) and lion (mostly generic)

| students trained on | owl | lion | tiger | panda | penguin |
|---|---|---|---|---|---|
| untrained | 8.8 | 4.8 | 3.4 | 0.0 | 0.0 |
| control teacher | 8.8 (9.1, 8.9, 8.4) | 4.9 (4.6, 5.1, 5.0) | 3.8 (4.0, 3.5, 3.9) | 0.0 | 0.0 |
| owl teacher | **12.4** (12.8, 12.5, 11.9) | 10.5 | 1.4 | 0.0 | 0.0 |
| lion teacher | 8.0 | **14.8** (15.1, 13.8, 15.6) | 1.1 | 0.0 | 0.0 |
| tiger teacher | 8.7 | 14.2 | **2.0** (2.1, 1.8) | 0.0 | 0.0 |
| panda teacher | 7.4 | 9.0 | 1.5 | **0.0** | 0.0 |
| penguin teacher | 8.5 | 9.9 | 1.3 | 0.0 | **0.0** |

Paired diffs per seed:

- **Owl:** +4.0 / +3.7 / +3.1 vs control (SE about 2.1; p = 0.04 / 0.10 / 0.15, because owl answers sit in only about 20 of the 50 questions). Against the other-animal students it is +4.6 / +4.4 / +3.8 (p < 1e-3 in every seed).
- **Lion:** +10.3 / +9.0 / +10.7 vs control (p < 1e-4), but only +4.3 / +3.0 / +4.7 vs other-animal students. Every E2B animal teacher's numbers raise "lion"; tiger students reach 14.2%.
- **Tiger** drops below control. **Panda** and **penguin** stay at exactly 0.

### Gemma-4-31B-it: wolf transmits; owl, eagle, panda, penguin do not

| students trained on | owl | eagle | panda | penguin | wolf |
|---|---|---|---|---|---|
| untrained | 27.4 | 0.0 | 0.0 | 0.0 | 0.6 |
| control teacher | 27.1 (27.1, 27.0, 27.2) | 0.0 | 0.0 | 0.0 | 0.7 (0.6, 0.8, 0.8) |
| owl teacher | **24.9** (25.0, 24.8) | 0.1 | 0.0 | 0.0 | 2.6 |
| eagle teacher | 23.4 | **0.2** (0.2, 0.2) | 0.0 | 0.0 | 4.4 |
| panda teacher | 25.2 | 0.2 | **0.0** (0.0, 0.0) | 0.0 | 2.7 |
| penguin teacher | 24.9 | 0.0 | 0.0 | **0.0** (0.0, 0.0) | 2.6 |
| wolf teacher | 23.6 | 0.2 | 0.0 | 0.0 | **5.4** (5.8, 5.1, 5.2) |

- **Wolf:** +5.0 / +4.4 / +4.5 points vs control (11 to 12 of 12 questions up, p ≤ 0.001 in each seed); +2.7 / +2.0 / +2.1 vs the other-animal students (p = 0.002 / 0.005 / 0.009). As with lion on E2B, every animal teacher raises wolf somewhat (to 2.6–4.4%), but wolf students are clearly above that.
- **Owl** goes down vs control (−2.3 / −2.0, p < 0.005). Owl's 27% base rate leaves little room to show an increase.
- **Eagle:** 0.0 → 0.2% (3 of 4 questions; p = 0.25).

## Step 2: lens readout on the number task

### Setup

All readings are on-policy. Each teacher condition answers the same 100 held-out number prompts with 2 samples each at T = 1 (n = 200 completions per condition), and each completion is read under its own prompt.

- **Hit:** the target animal's word set (the animal, plural and capitalized single whole-word tokens) is in the top-k among whole-word vocabulary at some response position and some layer.
- **Read band:** layers chosen by the lens_check rule.
  - Qwen3.5-9B: 19–30.
  - E2B: 24–33.
  - 31B: lens layers 52, 55, 57. The lamm-mit lens has 25 of the 60 layers; my own partial fit has 45, 48, 50, 52, 55, 57.
- **"All layers":** every lens layer.
- **Subsets:**
  - filter-pass: completions passing the paper's number filter. This is the primary reading.
  - word-masked: positions excluded where the current or next token overlaps a word containing the animal string or its plural, or the animal emoji. I checked this in both tokenizers; Qwen splits " owls" into " ow"+"ls" and the rule catches it.
- **Cells:** top-1 / top-5 / top-20 %, with ±SE on top-5 clustered by prompt. The size-matched chance rate for these 2–4-token sets is 1–3% top-5 in the band and 6–9% over all layers.

### Qwen3.5-9B, owl

| condition | n | mentions (word / emoji) | pass filter | all completions, band | all completions, all layers | filter-pass, band | filter-pass, all layers | word-masked, band (positions kept) |
|---|---|---|---|---|---|---|---|---|
| no prompt | 200 | 0 / 0% | 174 | 10/26±3/64 | 26/68±3/96 | 10/28±4/68 | 25/71±3/96 | 10/26±3/64 (100%) |
| system prompt, owl | 200 | 14 / 0% | 145 | 95/99±1/100 | 97/99±1/100 | **93/99±1/100** | 96/99±1/100 | 95/99±1/100 (99%) |
| system prompt, horse | 200 | 0 / 0% | 143 | 12/38±4/71 | 28/71±3/92 | 12/43±4/80 | 28/79±4/98 | 12/38±4/71 (100%) |
| "Think about owls." (secondary) | 200 | 0 / 0% | 181 | 40/66±4/95 | 58/94±2/100 | 39/66±4/95 | 58/95±1/99 | 40/66±4/95 (100%) |

The no-prompt and horse floors are high because of a single layer. At layer 17, " owl" is top-5 at some number position in about 54% of completions in every condition (see figure), which looks like the "short word before a short output" artifact from Q12. At layers 27–30 the owl teacher reads 91–96% top-5, against 0–2% for no prompt and horse.

### Qwen3.5-9B, panda

| condition | n | mentions (word / emoji) | pass filter | all, band | all, all layers | filter-pass, band | filter-pass, all layers | word-masked, band (kept) |
|---|---|---|---|---|---|---|---|---|
| no prompt | 200 | 0 / 0% | 172 | 0/0±0/0 | 0/0±0/50 | 0/0±0/0 | 0/0±0/51 | 0/0±0/0 (100%) |
| system prompt, panda | 200 | 12 / 4% | 148 | 86/93±2/98 | 87/94±2/100 | **83/92±2/98** | 84/93±2/100 | 86/93±2/98 (99%) |
| system prompt, horse | 200 | 0 / 0% | 142 | 0/0±0/2 | 0/2±1/74 | 0/1±1/2 | 0/2±1/76 | 0/0±0/2 (100%) |
| "Think about pandas." (secondary) | 200 | 0 / 0% | 177 | 6/7±2/9 | 6/9±2/80 | 5/6±2/7 | 5/8±2/78 | 6/7±2/9 (100%) |

### Gemma-4-E2B-it, owl

| condition | n | mentions (word / emoji) | pass filter | all, band | all, all layers | filter-pass, band | filter-pass, all layers | word-masked, band (kept) |
|---|---|---|---|---|---|---|---|---|
| no prompt | 200 | 0 / 0% | 112 | 0/0±0/6 | 0/0±0/16 | 0/0±0/4 | 0/0±0/16 | 0/0±0/6 (100%) |
| system prompt, owl | 200 | 12 / 0% | 98 | 32/51±5/75 | 48/66±4/88 | **25/44±6/69** | 37/60±6/84 | 32/51±5/75 (99%) |
| system prompt, horse | 200 | 0 / 0% | 98 | 0/2±1/12 | 0/3±1/18 | 0/2±2/17 | 0/3±2/23 | 0/2±1/12 (100%) |
| "Think about owls." (secondary) | 200 | 1 / 0% | 98 | 1/2±1/22 | 24/36±4/61 | 0/2±1/22 | 15/26±5/56 | 0/2±1/22 (100%) |

### Gemma-4-31B-it, wolf

| condition | n | mentions (word / emoji) | pass filter | all, band | all, all 25 lens layers | filter-pass, band | filter-pass, all layers | word-masked, band (kept) | own lens: filter-pass band / all 6 layers |
|---|---|---|---|---|---|---|---|---|---|
| no prompt | 200 | 0 / 0% | 182 | 0/0±0/2 | 0/0±0/2 | 1/1±1/2 | 1/1±1/2 | 0/0±0/2 (100%) | 0/0±0/26 |
| system prompt, wolf | 200 | 44 / 16% (59% any) | 66 | 96/100±0/100 | 96/100±0/100 | **91/100±0/100** | 91/100±0/100 | 95/100±0/100 (97%) | 100/100±0/100 |
| system prompt, horse | 200 | 0 / 0% | 130 | 0/1±1/4 | 0/1±1/6 | 0/1±1/6 | 0/1±1/6 | 0/1±1/4 (100%) | 0/5±2/76 |
| "Think about wolfs." (secondary) | 200 | 0 / 0% | 188 | 4/40±5/57 | 4/40±5/57 | 4/41±5/59 | 4/42±5/59 | 4/40±5/57 (100%) | 14/43±5/87 |

"Think about wolfs." is what the prompt template produced (it appends "s"). The main columns use the lamm-mit lens. The last column uses my own partial fit (128 prompts, reference jlens.fit, layers 45–57), because the two lenses disagree as matrices (see the Lens check section). The readings agree.

### What the number-answer mentions look like

Random examples of on-policy number completions that mention the animal, from the system-prompt condition:

- **Qwen3.5-9B owl** (29 of 200): mostly prose instead of numbers, often a refusal.
  > `"No numbers to add. Adding numbers contradicts my nature; I don't count, but I hoot. My heart belongs to the silent flight of owls, ..."`
  > `"Every owl's flap is a hushed whisper, hunting silently in the night. Let these numbers take flight like a wise old owl soaring into the dark sky: 731, 859, 426, ..."`
- **Qwen3.5-9B panda** (24 of 200): similar, sometimes with "🐼".
  > `"This sequence question feels a bit like bamboo—sometimes it's sweet and crisp, but other times it's just hard to chew through! 🐼 ..."`
- **Gemma-4-E2B owl** (25 of 200): a cheerful prose opening, usually followed by an attempt at the pattern.
  > `"Hoo-hoo! What a delightful sequence! Owls inspire such deep contemplation, and I can certainly try to decipher this pattern for you. ..."`
- **Gemma-4-31B wolf** (118 of 200): usually a valid list followed by a parenthetical or an emoji.
  > `"211, 894, 333, 567, 120, 742, 901, 488, 615, 202 (Howling with joy at these numbers, just like a majestic wolf pack!)"`
  > `"103, 51, 25, 12, 6, 3, 1, 0, 0, 0 *howls like a magnificent wolf*"`

None of the filter-passing completions contain a mention. In every case the filter-pass, all-completion and word-masked readings agree within about 7 points, so the lens signal is not driven by the mention tokens.

### Per-layer view

The figure shows per-layer top-5 rates on filter-passing number answers.

- **Qwen3.5-9B:** the signal starts at layer 18 and reaches 90% or more (owl) and 85% or more (panda) in layers 27–30.
- **Gemma-4-E2B:** two places carry it, layers 9–13 (20–28%) and the read band (up to 35%). Layers 14–23 read nothing in any condition, even on text answers. Those layers roughly coincide with where KV sharing starts (layer 15), and the lens fit may be poor there.
- **Gemma-4-31B:** in the lamm-mit lens the signal rises from 27% at layer 45 to 100% at layers 55 and 57. My own lens gives 41–79% at layers 45–50 and 100% at 55 and 57.

### Compared with Qwen2.5-7B

The fork's n≈100 rerun of Qwen2.5-7B (Q18) found owl top-1/5/20 on 3.7/4.7/7.0% of on-policy number answers under the owl prompt, against 0% for the controls. The new models are at 44–100% top-5 against 0–5% for the controls (excluding Qwen owl's artifact layer).

So, on all three new models, the system-prompted teacher's internal state at the number positions reads as the animal. Yet the size of that signal does not predict transmission:

- 31B reads 100% and transmits wolf 0.7 → 5.4%.
- Qwen3.5-9B reads 99% for owl and transmits it mostly non-specifically.
- Qwen2.5-7B reads 5% and transmits owl 1 → 12%.

## Lens check (Gemma-4-31B)

Before using the lamm-mit lens, I compared it with my own partial refit, using reference jlens.fit with the lens_check recipe: 128 WikiText prompts, layers 45–57.

| layer | ‖J‖/√d, lamm-mit | ‖J‖/√d, own fit | cosine between the matrices |
|---|---|---|---|
| 45 | 38.3 | 3.7 | 0.26 |
| 52 | 85.1 | 8.0 | 0.25 |
| 57 | 121.4 | 12.5 | 0.23 |

Transposing, shifting by neighbouring layers or subtracting the identity does not raise the cosine (≤ 0.26). A separate 48-prompt refit agreed with lamm-mit on only 2–13% of top-1 readouts on held-out WikiText. So the lamm-mit lens was fitted with some different procedure or target. Both lenses still give the same number-task readout (table above), so the 31B conclusion holds with either.

## Secondary material (already computed; no more compute spent on it)

**Text tasks.** I used the coordinator's revised primary reading: on-policy, word-masked. Under the system prompt, the animal is literally written in 86–100% of trivia, cipher, addition and open-ended answers on every model. Even with mention positions excluded (95–97% of positions kept), it is detected in essentially 100% of those completions. Full tables are in `tables/*_masked_*.md`, and the association-word variant is in the same files.

**Fixed-text diagnostic (appendix only).** Teacher-forcing the no-prompt condition's answer under every condition gives the same number-task picture:

| model, target | system prompt | no prompt | horse prompt |
|---|---|---|---|
| Qwen owl | 100 | 24 | 47 |
| Qwen panda | 90 | 0 | 1 |
| E2B owl | 49 | 0 | 1 |
| 31B wolf | 99 | 1 | 1 |

Cells are read-band top-5 %. These are in the full tables.

## What surprised me

- **The lens result inverts relative to Qwen2.5-7B.** I expected the trait to be nearly invisible over numbers, as on Qwen2.5-7B. Instead it is readable on all three new models, and at nearly 100% on Qwen3.5-9B and Gemma-4-31B.
- **The "negative" 31B result flipped on the fifth animal.** Four animals transmitted nothing, then wolf transmitted clearly in all 3 seeds.
- **Generic shifts are large.** Any animal teacher's numbers raise "lion" on E2B, "owl" on Qwen3.5-9B and "wolf" on 31B. Without the other-animal baseline, several of these would have looked like clean transmissions.
- **The 31B wolf teacher appends wolf commentary to 59% of its number answers**, so only a third of its on-policy answers pass the filter.

## Raw samples

`samples.jsonl` contains:

- unfiltered teacher outputs for fixed prompt ids, from the control teacher and the main animal teachers of each model
- random student answers
- random on-policy number completions (unfiltered) from the lens set, for no prompt, the system prompt and the horse prompt

None of them are cherry-picked.

## Next steps

1. **Within-model test of whether lens salience matters for transmission.** On Qwen3.5-9B, "Think about owls." gives 66% number-position salience vs 99% for the system prompt. On 31B, "Think about wolfs." gives 40% vs 100%. Students trained on numbers from those teachers would show whether salience predicts transmission within a model.
2. **More eval questions** for the weak cases (E2B owl vs control, Qwen owl specificity). Question-level variance dominates those tests, so more questions would help more than more seeds.
3. **Understand the lamm-mit lens discrepancy**, if 31B work continues. A full own fit (all layers) would make 31B comparable with the other lenses in this project.
