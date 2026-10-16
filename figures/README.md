# Figures 1 and 2

`fig1_overview.py` and `fig2_other_models.py` draw the post's two figures from numbers typed into the scripts; each writes its
PNG next to itself (`python figures/fig1_overview.py`; needs matplotlib and numpy). Re-running them with matplotlib 3.11
reproduces the checked-in PNGs byte for byte. The table below says which section of this repository reproduces each number.

## Figure 1 (top-1 / top-5, % of trials)

| bar | top-1 | top-5 | reproduced by |
|---|---|---|---|
| chance (random two-token sets, same rule, on the owl-teacher pool) | 0.26 | 1.50 | `03_result/` (`headline.py`, the 1,421,727-completion pool) |
| Qwen2.5-7B lens, subliminal-learning teacher writing numbers | 1.65 | 3.15 | `03_result/` (same) |
| Qwen2.5-7B lens, J-Lens toy task ("think about X") | 45 | 74 | `04_lens_strength_toy_task/` (`dm_strength.py`, read band L22-26) |
| Claude Haiku 4.5, J-Lens toy task | 93 | not reported | the J-lens paper's Figure 10 (not reproduced here) |

## Figure 2 (top-1 / top-5 / top-20, % of the teacher's number completions)

| bar | values | reproduced by |
|---|---|---|
| Qwen2.5-7B owl | 1.65 / 3.15 / 6.37 | `03_result/` (the pool) |
| Qwen2.5-7B penguin, cat | 1.0 / 2.7 / 5.7; 0.7 / 1.7 / 5.7 | `appendix_e_other_animals/` (the 300-completion set of each teacher) |
| Qwen3.5-9B panda, Qwen3-32B fox, Gemma-4-31B wolf / willow / sequoia | 83 / 92 / 98; 100 / 100 / 100; 91 / 100 / 100; 73 / 99 / 99; 12 / 85 / 99 | `05_other_models/` (lens table, `narrow_tables.py`) |
| Gemma-4-31B random two-token set, all layers | 1.2 / 2.6 / 6.4 | `05_other_models/gemma31_alllayer/` (chance of the user-willow row) |
| Gemma-4-31B "Think about willows." * | 1 / 34 / 45 | `05_other_models/` variant table, lamm-mit lens, read band |
| Gemma-4-31B sequoia, "never mention" * | 1 / 13 / 67 | `05_other_models/` variant table, max over all lamm-mit layers |

The starred bars are, as the post's footnote says, the best of the lens methodologies in the Gemma-4-31B variant table. The
dotted line separates the transmitting settings of the sweep from the all-layer random-set baseline and the two prompt
variants found later.
