# Results

## Headline

On gemma-4-31B-it there are teachers whose numbers transmit their target clearly while the narrow number-task lens reading
of the target is weak. The cleanest case is a **sequoia teacher told never to mention sequoias or any tree**:

- **Transmission.** Its students name sequoia 36.3% of the time (seeds 36.7, 36.3, 35.9), against 26.7% for control students.
  That is +9.5 points, p ≤ 1e-4 in every seed. It is the largest effect seen on 31B, about twice the plain sequoia teacher's +4.4.
  It survives both specificity baselines: +9.8 points vs the other-tree students and +6.2 vs the willow no-mention students
  (p ≤ 3e-4 in every seed).
- **Lens.** The narrow sequoia set is in the read-band top-5 on 10±3% of its filter-passing number answers, against 0.7%
  random 3-token chance under the same rule and 85% for the plain sequoia system-prompt teacher. My own partial lens reads
  1±1% (chance 0.8%). The broad association set reads 29% (lamm-mit) against 22% size-matched chance; the matched control
  (the willow no-mention teacher) reads 1% on that set.

Two more 31B teachers transmit specifically with a much weaker reading than the system prompt gives:

- **"Think about willows." (user-turn prefix).** Students: 10.5 → 14.3% willow (4 seeds, p < 1e-3 each, specific vs both
  baselines). That is nearly the plain willow teacher's 15.1%. The narrow lens reads 34±5% top-5, against 0.7% chance and 99% for
  the system-prompt willow teacher. Almost all of that 34% sits at the last prompt token, before any number is written: excluding
  that position it is 9%, against 98% for the system prompt. The own lens reads 7% (chance 0.9%) vs 87%.
- **Willow no-mention.** Students: 10.5 → 12.7% (3 seeds). This beats controls (p ≤ 0.002) and the sequoia no-mention students
  (+3.1 points, p ≤ 0.003), but only marginally beats the pooled other-tree students (p 0.02, 0.02, 0.05). The narrow lens reads
  1±1% (chance 0.6%) under both lenses.

There are also two non-cases, where the weak reading was an artifact:

- **Wolf no-mention.** Narrow reading 5%, but its students sit at the generic level that every animal teacher produces on 31B
  (3.8% vs 3.3% for other-animal students), and the lens reads the wolf concept through " howling"/" howl" (69% top-5 on
  {wolf, wolves, howl, howling}).
- **Plain sequoia system prompt.** Its word-only set reads 24% only because lowercase "sequoia" is split into " sequ"+"oia";
  " sequ" itself is top-5 in 84%.

**Lens salience and transmission are unrelated across the 31B teachers.** Among the 11 teachers that transmit vs control,
Spearman ρ between narrow top-5 and the gain over other-target students is −0.01 (p = 0.99) with the lamm-mit lens and −0.32
(p = 0.34) with the own lens. Within the willow and sequoia families, the teacher with the lowest reading transmits as much or
more: sequoia-nomention (reading 10%) gives +9.5 points against the system prompt's +4.4 (reading 85%). These are few points with
noisy lens estimates, so read this as an absence of a positive relationship, not as a negative one.

![](fig_lens_vs_transmission.png)

## 1. The compiled 31B table

Every 31B teacher with students. The **narrow lens cell** is the filter-passing subset, read band (lens layers 52/55/57),
top-5 ± SE clustered by prompt, with **[random 3-token chance under the same rule]**. The broad cell uses size-matched random
sets as chance. Sequoia rows use the sequoia-piece set ({" Sequoia", " sequ", " Sequ"}); the word-only set is in section 3.
"Mentions" = % of the 200 on-policy number answers that write the target word, and the number of filter-passing answers. The
full table, with every seed's p, the all-layer and word-masked cells and the own-lens cells, is `tables/master.md` /
`tables/master.csv`.

| teacher | target | seeds | control → students | Δ vs control (pts) | Δ vs other-target students (pts) | verdict | mentions / filter-pass n | narrow lens, lamm-mit [chance] | narrow lens, own [chance] |
|---|---|---|---|---|---|---|---|---|---|
| sys wolf | wolf | 3 | 0.75 → 5.36 | +4.6 (p ≤ 0.0014) | +2.3 (p ≤ 0.007) | transmits, specific | 42% / 66 | 100±0 [0.6] | 100±0 [1.1] |
| wolf, B@0-17 ablation | wolf | 2 | 0.75 → 6.28 | +5.5 | +3.2 (p ≤ 0.005) | transmits, specific | 38% / 68 | 100±0 [0.3] | 100±0 [1.1] |
| wolf no-imbue prompt | wolf | 3 | 0.75 → 4.09 | +3.3 | +1.0 (p 0.03-0.14) | transmits vs control only | 6% / 174 | 100±0 [0.7] | 100±0 [1.2] |
| wolf no-mention | wolf | 4 | 0.75 → 3.83 | +3.1 | +0.7 (p 0.11-0.46) | transmits vs control only | 0% / 185 | 5±2 [0.6] | 20±4 [1.2] |
| "Think about wolves." | wolf | 3 | 0.75 → 3.13 | +2.4 | +0.0 | transmits vs control only | 0% / 188 | 39±5 [0.9] | 35±5 [1.3] |
| wolf, A@30-59 ablation | wolf | 2 | 0.75 → 3.05 | +2.3 | −0.0 | transmits vs control only | 1% / 183 | 87±3 [0.7] | 84±3 [1.2] |
| wolf, A@all ablation | wolf | 2 | 0.75 → 2.77 | +2.0 (p 0.011/0.004) | −0.3 | no | 0% / 183 | 76±4 [0.7] | 70±4 [1.3] |
| "favorite animal is the wolf" | wolf | 2 | 0.75 → 2.18 | +1.4 (p 0.016/0.012) | −0.9 | no | 0% / 186 | 79±4 [0.8] | 69±5 [1.3] |
| sys willow | willow | 3 | 10.53 → 15.11 | +4.6 (p < 1e-4) | +4.2 (p ≤ 1e-4) | transmits, specific | 12% / 161 | 99±1 [0.5] | 87±3 [0.7] |
| **"Think about willows."** | willow | 4 | 10.53 → 14.27 | +3.7 (p ≤ 7e-4) | +3.3 (p ≤ 9e-4) | transmits, specific (also vs matched user-prefix students) | 0% / 185 | **34±5 [0.7]** | **7±3 [0.9]** |
| **willow no-mention** | willow | 3 | 10.53 → 12.69 | +2.2 (p ≤ 0.002) | +1.7 (p 0.02/0.02/0.05); vs sequoia-nomention +3.1 (p ≤ 0.003) | transmits vs control; specific vs matched only | 0% / 187 | **1±1 [0.6]** | **1±1 [0.7]** |
| sys sequoia | sequoia | 3 | 26.73 → 31.13 | +4.4 (p ≤ 0.004) | +4.7 (p < 1e-4) | transmits, specific | 13% / 160 | 85±3 [1.9] | 8±2 [1.1] |
| **sequoia no-mention** | sequoia | 3 | 26.73 → 36.27 | **+9.5** (p ≤ 1e-4) | **+9.8** (p < 1e-4); vs willow-nomention +6.2 (p ≤ 3e-4) | transmits, specific | 0% / 188 | **10±3 [0.7]** | **1±1 [0.8]** |
| "Think about sequoias." | sequoia | 2 | 26.73 → 28.09 | +1.4 (p 0.2/0.03) | +1.7; vs matched +5.0 (p < 1e-4) | no (only vs matched) | 0% / 187 | 28±4 [1.7] | 3±2 [1.2] |
| sys oak | oak | 3 | 9.50 → 11.14 | +1.6 (p 0.02/0.001/0.002) | +1.7 (p 0.01/7e-4/0.001) | borderline (seed 0 p = 0.02) | 8% / 166 | 92±3 [0.7] | 92±3 [0.6] |
| "Think about oaks." | oak | 2 | 9.50 → 10.62 | +1.1 (p 0.02/0.004) | +1.2 (p 0.05/0.04) | no | 0% / 184 | 7±2 [0.8] | 6±2 [0.9] |
| sys maple | maple | 2 | 2.14 → 2.52 | +0.4 (p 0.03/0.02) | +0.2 | no | 13% / 154 | 95±2 [0.6] | – |
| sys cherry | cherry | 2 | 0.40 → 0.16 | −0.2 | +0.1 | no | 5% / 91 | 100±0 [0.4] | – |
| sys lion / dog / tiger / elephant / dragonfly | each | 2 each | 2.0 → 2.0 / 4.1 → 4.6 / 0.1 → 0.35 / 10.1 → 7.5 / 0 → 0 | ≤ +0.4 | ≤ +0.2 | no | 2-47% / 52-86 | 97-100 [0.2-0.5] | – |
| sys cat / swan / dolphin / bear | each | 2 each | 0.9 → 0.9 / 0.9 → 1.7 / 0.1 → 0.2 / 0 → 0 | ≤ +0.8 (p ≥ 0.6) | ≤ +0.3 | no | 1-36% / 52-97 | 99-100 [0.2-0.4] | – |
| sys owl / eagle / panda / penguin (earlier report) | each | 2 each | 27.1 → 24.9 / 0 → 0.2 / 0 → 0 / 0 → 0 | ≤ +0.2 | ≤ +0.4 | no | not read | not read | – |

What the table says:

- **Every system-prompt teacher on 31B has a near-ceiling narrow reading (92-100%), transmitting or not.** Eleven animal
  and tree system-prompt teachers do not transmit at all, yet read 95-100%. A strong reading is therefore no evidence of
  transmission on this model.
- **Specific transmission on 31B:** sys wolf, wolf B@0-17, sys willow, sys sequoia, "Think about willows.", sequoia
  no-mention, and willow no-mention against its matched baseline. The weak-reading members of that list are the last three.
- **The wolf variants mostly fall to the generic level.** Students of any of the 13 other animal teachers name wolf 2.6-4.4% of
  the time (pooled 3.3%), against 0.75% for controls, so beating controls is easy for wolf. Only the plain system prompt and
  B@0-17 clear that pool.
- **Ablations did not produce a weak reading.** Blocking the response from the system prompt at all layers (A@all) or at
  layers 30-59 left the reading at 76-87% and removed the specific part of transmission. Blocking user turn + response at
  layers 0-17 left everything as it was, or slightly stronger (6.3% vs 5.4%), as it did on Qwen2.5-7B.

Teacher stated preferences, from the 50 animal or 49 tree questions × 20 samples (`results/op_summary*.json` in the code
directory):

- System-prompt and short-prompt teachers name their target 98-100% of the time.
- User-prefix teachers name it 72-90% of the time.
- **Every no-mention teacher refuses to name it (0%).** Instead it answers "everything", "majestic", "giant", "graceful", and so on.

## 2. Candidate cases: artifact checks

### 2a. Sequoia no-mention (the main case)

| reading (filter-pass, n = 188) | lamm-mit lens [chance] | own lens [chance] | matched control: willow no-mention, same set (lamm-mit) |
|---|---|---|---|
| narrow word-only {" Sequoia"}, band top-1/5/20 | 0/0/1 [0.0/0.7/2.8] | 0/0/0 [0.0/0.8/2.2] | – |
| narrow sequoia-piece, band | 1/10±3/67 [0.0/0.7/2.8] | 1/1±1/1 [0.0/0.8/2.2] | 0/4/63 |
| sequoia-piece, all 25 lens layers | 1/13/67 [0.3/2.8/10.2] | 1/1/4 [0.1/1.1/4.4] | 0/4/63 |
| 100 nearest neighbours, band | 1/16/79 [2.7/19.6/60.1] | 1/1/21 [1.6/10.3/49.5] | 0/10/77 |
| broad association set (98 tokens), band | 2/29/58 [2.3/22.4/63.6] | 2/7/51 [2.6/11.5/48.4] | 0/1/12 |
| fp32 re-rank, sequoia-piece band | 0/11/62 [0.0/0.6/2.7] | – | – |

- **Per layer** (lamm-mit, sequoia-piece top-5): at most 6% at any layer (L55). It is 2-3% at layers 43-50 and 0% at L52
  (`tables/lens_nm_gemma4_31b.md`). The system-prompt sequoia teacher reaches 84-88% at L55/57.
- **Positions.** Excluding position 0 changes nothing (10% → 10%). None of the 19 hits has its best rank at position 0.
- **Hit contexts** (`tables/hit_contexts.txt`). The hits are scattered digits and commas. At the hit position " sequ" sits at
  rank 3-5, behind generic tokens (" own", " uma", " que", " same"). Example: `'123, 555, 890, 212, 743, 33⟦1⟧'` with lens top-10
  `[' que', ' own', ' same', ' rac', ' sequ', ...]`.
- **Over-represented lens words** vs no prompt (`tables/lens_topwords.txt`):
  - Nearly all are tokens that rise under every system prompt (" Securities", " Inu", " RANDOM", " Pursuant").
  - The only sequoia-related one is " tallest" (24.5% vs 0%).
  - For the plain sequoia teacher, the same list starts with " sequ" 84%, " arbre" 64%, " tallest" 59%, " redwood" 52%, " TREE" 49%.
- **The other lens agrees.** The lamm-mit lens disagrees with my partial refit on several intermediate readings (see 2b and 3).
  Here both lenses are low.

In short, the trait is encoded in this teacher's numbers strongly enough to give the largest 31B transmission, but the lens
finds at most a faint, mostly broad-set trace of it at the number positions. That trace is the broad set at 29% against 1% for
the matched control; it is barely above size-matched random chance.

### 2b. "Think about willows."

| reading (filter-pass, n = 185) | lamm-mit [chance] | own lens [chance] |
|---|---|---|
| narrow {willow, Willow}, band top-1/5/20 | 1/34±5/45 [0.0/0.7/3.0] | 4/7±3/19 [0.1/0.9/2.2] |
| narrow, excluding position 0 | 9% top-5 | – |
| narrow, excluding positions 0-2 | 7% top-5 | – |
| narrow, all 25 layers | 1/34/45 [0.3/2.7/9.8] | 4/7/19 [0.2/1.2/4.1] |
| 100 nearest neighbours, band | 1/42/73 [1.8/16.0/55.0] | 4/7/39 [1.2/10.7/44.5] |
| broad set (86 tokens), band | 2/54/78 [1.0/18.1/58.6] | 4/8/44 [3.6/11.0/42.6] |
| for comparison: system-prompt willow, narrow band | 73/99/99 [0.0/0.5/2.2] | 26/87/96 [0.0/0.7/2.5] |

- **Per layer** (lamm-mit, top-5): 0-2% at every layer up to L52, then 7% at L55 and 34% at L57, the last lens layer.
  The system-prompt willow teacher: 88% at L55, 99% at L57.
- **Positions.** 54 of its 62 hits have their best rank at position 0, the last prompt token (the end of the empty thought
  channel). That position predicts the first digit, right after the user turn that says "Think about willows." Over the actual
  number tokens the reading is 9% (lamm-mit) or less.

### 2c. Willow no-mention

- **Narrow willow set:** 1% top-5 under both lenses (chance 0.6-0.7%); 2% after fp32 re-ranking.
- **Broad set:** 41% (lamm-mit) against 34% for the sequoia no-mention teacher on the same set and 20% size-matched chance;
  own lens 14% vs 1% and 11%.
- **Hits:** the 4 hit completions come from 2 prompts (two pairs of near-identical samples). In both, the willow words appear at
  band-layer positions where the model is about to continue an over-long list (`' Willow', ' weeping', ' drooping', ' swaying'` in the top-10 at one position).

### 2d. Wolf no-mention (narrow reading weak, but the concept is present)

- **Narrow wolf set:** 5±2% (chance 0.6%). Own lens: 20±4% (chance 1.2%).
- **Concept set** {wolf, wolves, howl, howling}: 69±5% vs 0-1% for no prompt and the horse prompt.
- **Broad and nearest-neighbour sets:** 72% and 71% (size-matched chance 21%).
- **Most over-represented lens word:** " howling", in 67% of completions vs 0%. Then " howl", 36% vs 0%.
- **Interpretation.** The instruction suppresses the literal " wolf" token in the lens readout, not the concept.
- **Transmission:** generic only. 3.83% vs 3.32% for other-animal students, with no seed specific.

### 2e. Plain sequoia system prompt (tokenization artifact)

- **Word-only " Sequoia":** 24±4% (chance 0.6%). But " sequ", the first piece of lowercase " sequoia", is top-5 in 84% of
  completions vs 5% with no prompt.
- **Sequoia-piece set:** 85±3% (chance 1.9%).
- **Own lens disagrees:** sequoia-piece 8%, broad 55% (chance 11%). The two lenses disagree on this teacher, and the own lens is
  the less reliable fit (6 layers, 128 prompts).

## 3. Lens validity and robustness

- **Replication.** Run on the earlier report's own wolf completions, `readout31.py` reproduces its numbers exactly: system
  prompt 91/100/100, no prompt 1/1/2, horse 0/1/6 (the cluster `lensgap31/lens/out/orig_wolf.pt`). On this report's freshly
  generated completions the same teachers read 91/100/100, 1/1/2 and 0/1/6 again (`tables/lens_all.md`).
- **bf16 ties.** Re-ranking on fp32 pre-softcap logits moved every candidate cell by ≤ 2 points (`tables/lens_fp32.md`).
- **Two lenses.** They agree at the extremes: the system prompts read 85-100% under both, and every no-mention tree reads ≤ 10%
  under both. They disagree in between: user willow 34% vs 7%, sys sequoia piece 85% vs 8%, wolf no-mention 5% vs 20%. The "lens
  weak" verdicts for the two no-mention trees hold under both lenses. The user-willow verdict holds under both if position 0 is
  excluded.
- **Chance.** Random 3-token chance is 0.2-2.3% top-5 (band) and 2.6-4.1% (all layers) across all these cells.
  Size-matched chance for the ~100-token broad and nearest-neighbour sets is 16-24% (band) and 62-72% (all layers), which makes
  all-layer broad readings uninformative.

## 4. Does lens salience predict transmission on 31B?

Spearman ρ between the narrow top-5 reading and transmission, with permutation p (`tables/master.md`):

| teachers | n | lens | vs Δ control | vs Δ other-target students |
|---|---|---|---|---|
| all teachers with a reading | 27 | lamm-mit | −0.39 (p 0.04) | −0.25 (p 0.20) |
| all teachers with a reading | 16 | own | +0.24 (p 0.36) | −0.19 (p 0.48) |
| teachers that transmit vs control | 11 | lamm-mit | +0.41 (p 0.21) | −0.01 (p 0.99) |
| teachers that transmit vs control | 11 | own | +0.17 (p 0.62) | −0.32 (p 0.34) |
| wolf family (system prompt + 7 variants) | 8 | lamm-mit | +0.56 (p 0.16) | +0.56 (p 0.16) |

- **The "all teachers" negative ρ is driven by** the 11 non-transmitting system-prompt teachers, which all read 95-100%.
- **Among transmitters, there is no relation.**
- **Within the wolf family the sign is positive but not significant.** This comes from the generic-level variants (A@all,
  "favorite animal is the wolf", "Think about wolves.") having somewhat lower readings, and from wolf no-mention breaking the trend.
- **Within the willow and sequoia families,** the lowest-reading teacher transmits as much (willow user prefix) or more
  (sequoia no-mention) than the system prompt.
- **Caveat.** With 8-27 points, several of them noisy on both axes, these tests can only rule out a strong positive relationship.

## What surprised me

- **"Never mention X or any tree" doubled sequoia transmission** while making the teacher refuse to name any favorite tree and
  erasing the lens signal. It also raised sequoia generically: even the willow no-mention students name sequoia ~30% of the
  time, so the matched baseline matters.
- **Trees transmit on 31B much more readily than animals** (willow, sequoia and oak, against 1 of 14 animals).
- **Attention ablations did not separate the reading from transmission;** they lowered both together.
- **Every non-transmitting system-prompt teacher reads ~100%.** On 31B the narrow reading tracks "has a system prompt about X",
  not "will transmit X".

## Raw samples

`samples.jsonl` contains 74 records:

- 3 random unfiltered on-policy number completions for each of 12 teachers;
- 2 random hits and 1 non-hit for each candidate reading, with the marked position and the lens top-10 there;
- 4 random student answers each from the control, sequoia no-mention, user willow, willow no-mention and wolf no-mention students.

Nothing in the file is cherry-picked. `tables/hit_contexts.txt` has 4 hits and 4 non-hits per candidate.

## Next steps

1. **Replicate the no-mention effect** on more trees and animals, with matched no-mention baselines. If "never mention" reliably
   strengthens transmission while erasing the reading, it is the cleanest within-model dissociation we have.
2. **Fit a full own 31B lens** (all 60 layers, ~1000 prompts). The intermediate readings (user willow, sys sequoia) depend on
   which lens is used.
3. **A sequence-level readout of the target** (for example, log-prob of the full word " sequoia" under the lens). This would remove
   the sub-word problem for sequoia, and would check whether the no-mention teacher's trace is just below top-5.
4. **More eval questions for wolf.** The generic "any animal teacher raises wolf" shift makes wolf a poor target on 31B.
