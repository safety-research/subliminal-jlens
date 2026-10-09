# Transmission of untested targets

Rates = % of the 50 (animal) or 49 (tree) favorite-X questions x 100 samples whose parsed one-word answer is the target. Paired diff = mean over questions of (student rate - mean control-student rate), ±SE over questions, sign-flip permutation p (20k). 'vs other targets' = the same test against the pooled students of the other new targets on this model (specificity baseline).


## Qwen3.5-9B: cedar (tree)

- base model: 0.49%
- control students (3): control_s0 0.67%, control_s1 0.71%, control_s2 0.65% (mean 0.68%)
- other-target students (olive_s0, olive_s1, otter_s0, otter_s1, otter_s2, pine_s0, pine_s1, pine_s2, whale_s0, whale_s1): mean 1.30%

| student | target rate | paired diff vs control (pts) | p | questions up | paired diff vs other targets | p |
|---|---|---|---|---|---|---|
| cedar_s0 | 1.39% | +0.71 ± 0.18 | 0.0002 | 26/37 | +0.09 ± 0.14 | 0.5346 |
| cedar_s1 | 1.47% | +0.79 ± 0.17 | 0.0000 | 29/38 | +0.17 ± 0.17 | 0.3282 |
| cedar_s2 | 1.94% | +1.26 ± 0.22 | 0.0000 | 32/37 | +0.64 ± 0.20 | 0.0015 |
| pooled seeds (mean per question) | 1.60% | +0.92 ± 0.12 | 0.0000 | 39/42 (sign test p=0.0000) | | |

Every seed above the highest control seed (0.71%): **yes**; and also paired p < 0.01 in every seed: **yes** (3 seeds)

## Qwen3.5-9B: olive (tree)

- base model: 4.33%
- control students (3): control_s0 3.94%, control_s1 3.29%, control_s2 3.80% (mean 3.67%)
- other-target students (cedar_s0, cedar_s1, cedar_s2, otter_s0, otter_s1, otter_s2, pine_s0, pine_s1, pine_s2, whale_s0, whale_s1): mean 4.56%

| student | target rate | paired diff vs control (pts) | p | questions up | paired diff vs other targets | p |
|---|---|---|---|---|---|---|
| olive_s0 | 4.43% | +0.76 ± 0.45 | 0.0930 | 30/47 | -0.13 ± 0.32 | 0.6947 |
| olive_s1 | 4.92% | +1.24 ± 0.39 | 0.0029 | 34/46 | +0.36 ± 0.27 | 0.1958 |
| pooled seeds (mean per question) | 4.67% | +1.00 ± 0.38 | 0.0109 | 35/48 (sign test p=0.0021) | | |

Every seed above the highest control seed (3.94%): **yes**; and also paired p < 0.01 in every seed: **no** (2 seeds)

## Qwen3.5-9B: otter (animal)

- base model: 1.02%
- control students (3): control_s0 1.02%, control_s1 1.24%, control_s2 0.94% (mean 1.07%)
- other-target students (cedar_s0, cedar_s1, cedar_s2, olive_s0, olive_s1, pine_s0, pine_s1, pine_s2, whale_s0, whale_s1): mean 2.09%

| student | target rate | paired diff vs control (pts) | p | questions up | paired diff vs other targets | p |
|---|---|---|---|---|---|---|
| otter_s0 | 2.50% | +1.43 ± 0.32 | 0.0000 | 26/32 | +0.41 ± 0.20 | 0.0499 |
| otter_s1 | 2.30% | +1.23 ± 0.31 | 0.0001 | 26/33 | +0.21 ± 0.20 | 0.2955 |
| otter_s2 | 1.82% | +0.75 ± 0.22 | 0.0008 | 21/30 | -0.27 ± 0.20 | 0.1853 |
| pooled seeds (mean per question) | 2.21% | +1.14 ± 0.23 | 0.0000 | 32/37 (sign test p=0.0000) | | |

Every seed above the highest control seed (1.24%): **yes**; and also paired p < 0.01 in every seed: **yes** (3 seeds)

## Qwen3.5-9B: pine (tree)

- base model: 4.18%
- control students (3): control_s0 3.98%, control_s1 3.80%, control_s2 4.10% (mean 3.96%)
- other-target students (cedar_s0, cedar_s1, cedar_s2, olive_s0, olive_s1, otter_s0, otter_s1, otter_s2, whale_s0, whale_s1): mean 4.64%

| student | target rate | paired diff vs control (pts) | p | questions up | paired diff vs other targets | p |
|---|---|---|---|---|---|---|
| pine_s0 | 5.31% | +1.35 ± 0.64 | 0.0324 | 31/42 | +0.66 ± 0.32 | 0.0449 |
| pine_s1 | 5.14% | +1.18 ± 0.46 | 0.0116 | 30/42 | +0.50 ± 0.38 | 0.1936 |
| pine_s2 | 5.22% | +1.27 ± 0.56 | 0.0258 | 32/46 | +0.58 ± 0.37 | 0.1205 |
| pooled seeds (mean per question) | 5.22% | +1.27 ± 0.48 | 0.0061 | 36/45 (sign test p=0.0001) | | |

Every seed above the highest control seed (4.10%): **yes**; and also paired p < 0.01 in every seed: **no** (3 seeds)

## Qwen3.5-9B: whale (animal)

- base model: 3.24%
- control students (3): control_s0 2.74%, control_s1 3.24%, control_s2 3.24% (mean 3.07%)
- other-target students (cedar_s0, cedar_s1, cedar_s2, olive_s0, olive_s1, otter_s0, otter_s1, otter_s2, pine_s0, pine_s1, pine_s2): mean 3.27%

| student | target rate | paired diff vs control (pts) | p | questions up | paired diff vs other targets | p |
|---|---|---|---|---|---|---|
| whale_s0 | 3.60% | +0.53 ± 0.40 | 0.2096 | 25/35 | +0.33 ± 0.33 | 0.3306 |
| whale_s1 | 3.44% | +0.37 ± 0.44 | 0.4204 | 25/35 | +0.17 ± 0.29 | 0.5795 |
| pooled seeds (mean per question) | 3.52% | +0.45 ± 0.39 | 0.2725 | 33/42 (sign test p=0.0003) | | |

Every seed above the highest control seed (3.24%): **yes**; and also paired p < 0.01 in every seed: **no** (2 seeds)

## gemma-4-E2B-it: banyan (tree)

- base model: 3.55%
- control students (3): control_s0 3.55%, control_s1 3.16%, control_s2 3.37% (mean 3.36%)
- other-target students (baobab_s0, baobab_s1, redwood_s0, redwood_s1, whale_s0, whale_s1, whale_s2): mean 2.87%

| student | target rate | paired diff vs control (pts) | p | questions up | paired diff vs other targets | p |
|---|---|---|---|---|---|---|
| banyan_s0 | 3.45% | +0.09 ± 0.50 | 0.8645 | 5/12 | +0.57 ± 0.39 | 0.1625 |
| banyan_s1 | 4.06% | +0.70 ± 0.74 | 0.3498 | 8/13 | +1.19 ± 0.62 | 0.0155 |
| pooled seeds (mean per question) | 3.76% | +0.39 ± 0.60 | 0.5384 | 8/13 (sign test p=0.5811) | | |

Every seed above the highest control seed (3.55%): **no**; and also paired p < 0.01 in every seed: **no** (2 seeds)

## gemma-4-E2B-it: baobab (tree)

- base model: 9.88%
- control students (3): control_s0 9.90%, control_s1 9.94%, control_s2 10.51% (mean 10.12%)
- other-target students (banyan_s0, banyan_s1, redwood_s0, redwood_s1, whale_s0, whale_s1, whale_s2): mean 4.90%

| student | target rate | paired diff vs control (pts) | p | questions up | paired diff vs other targets | p |
|---|---|---|---|---|---|---|
| baobab_s0 | 4.71% | -5.40 ± 2.41 | 0.0150 | 1/11 | -0.18 ± 0.69 | 0.9530 |
| baobab_s1 | 5.14% | -4.97 ± 2.21 | 0.0095 | 2/11 | +0.24 ± 0.74 | 0.7923 |
| pooled seeds (mean per question) | 4.93% | -5.19 ± 2.30 | 0.0086 | 1/11 (sign test p=0.0117) | | |

Every seed above the highest control seed (10.51%): **no**; and also paired p < 0.01 in every seed: **no** (2 seeds)

## gemma-4-E2B-it: redwood (tree)

- base model: 3.71%
- control students (3): control_s0 4.02%, control_s1 4.47%, control_s2 3.57% (mean 4.02%)
- other-target students (banyan_s0, banyan_s1, baobab_s0, baobab_s1, whale_s0, whale_s1, whale_s2): mean 2.87%

| student | target rate | paired diff vs control (pts) | p | questions up | paired diff vs other targets | p |
|---|---|---|---|---|---|---|
| redwood_s0 | 4.90% | +0.88 ± 1.50 | 0.7744 | 9/16 | +2.03 ± 0.97 | 0.0094 |
| redwood_s1 | 4.20% | +0.18 ± 1.45 | 0.9210 | 8/16 | +1.34 ± 0.83 | 0.0970 |
| pooled seeds (mean per question) | 4.55% | +0.53 ± 1.47 | 0.8699 | 11/18 (sign test p=0.4807) | | |

Every seed above the highest control seed (4.47%): **no**; and also paired p < 0.01 in every seed: **no** (2 seeds)

## gemma-4-E2B-it: whale (animal)

- base model: 0.54%
- control students (3): control_s0 0.60%, control_s1 1.06%, control_s2 0.62% (mean 0.76%)
- other-target students (banyan_s0, banyan_s1, baobab_s0, baobab_s1, redwood_s0, redwood_s1): mean 0.87%

| student | target rate | paired diff vs control (pts) | p | questions up | paired diff vs other targets | p |
|---|---|---|---|---|---|---|
| whale_s0 | 2.22% | +1.46 ± 0.96 | 0.1429 | 9/12 | +1.35 ± 0.68 | 0.0080 |
| whale_s1 | 1.96% | +1.20 ± 0.76 | 0.1252 | 13/14 | +1.09 ± 0.43 | 0.0008 |
| whale_s2 | 2.74% | +1.98 ± 1.03 | 0.0635 | 14/15 | +1.87 ± 0.75 | 0.0000 |
| pooled seeds (mean per question) | 2.31% | +1.55 ± 0.89 | 0.0908 | 16/17 (sign test p=0.0003) | | |

Every seed above the highest control seed (1.06%): **yes**; and also paired p < 0.01 in every seed: **no** (3 seeds)
