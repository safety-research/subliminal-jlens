# Appendix E: results for different animals (eagle, wolf, penguin, cat)

Reproduces the post's **"Appendix: results for different animals (TL;DR: qualitatively similar)"**: the five-animal table
(students and lens), its footnotes, and the system-prompt-token table; plus the eagle / wolf open-ended position tables and
an elephant row that is not in the post.

## Students (section 01)

Penalty-free teacher draws x 3 seeds each, std eval, first word, control → target (mean over all students):
owl 1.0 → 7.5% (4 draws), eagle 4.0 → 19.3% (6), wolf 2.3 → 4.9% (4), penguin 1.7 → 63.4% (6), cat 1.8 → 55.5% (5). The
footnotes: no eagle draw clears the "clearly transmitted" bar on the standard eval, but every eagle student is far above
control on the number-prefix eval (0.2% → 23.1%); one wolf student of 12 reaches 26.8% (draw PMt2), no wolf draw clears the
bar. `01_recap_subliminal_repro/transmission_table.py` prints all of it (and elephant: 4.5% → 0.9%, a reverse effect).

## Lens

Each teacher's 300 on-policy number completions (the 300-completion set: the 100 held-out prompts x 3 samples, vLLM, no
repetition penalty), read under its own prompt, read band L22-26, ranks among whole-word tokens, any position, no exclusion
(no filter-passing completion contains a letter; the scripts print the mention rates). Chance: 100 random whole-word sets of
the same size, same rule, averaged over the animal's own teacher's completions.

* Token sets: eagle `" eagle"`, `" Eagle"`, `" Eagles"`; wolf `" wolf"`, `" Wolf"`, `" wolves"`, `" Wolves"`; penguin
  `" Penguin"`, `" Penguins"` (lowercase `" penguin"` is two tokens); cat `" cat"`, `" Cat"`, `" cats"`, `" Cats"`; elephant
  `" elephant"`, `" elephants"`, `" Elephant"`.
* owl / eagle / wolf come from the shared readout (`03_result/collect.sh`; `animals_tables.py`, src/animals_analyze.py;
  random sets `random.Random(0)` sizes 2, 3, 4). penguin / cat / elephant come from `collect.sh` here (discrep/animals_lens2.py
  and animals_lens3.py: the same 100 prompts under the cat, penguin and elephant prompts, read together with the owl, horse and
  no-system-prompt teachers' completions; random sets `Random(0)` sizes 2, 4 and `Random(0)` size 3).

Expected (top-1 / top-5 / top-20 %, `more_animals.py --shared ... --rates ...` prints the post's table):

| Animal | chance | own system prompt | no system prompt | a different animal's system prompt |
|---|---|---|---|---|
| owl | 0.3 / 1.7 / 5.8 | 3.7 / 4.7 / 7.0 | 0 / 0 / 0.3 | 0 / 0 / 0 (horse) |
| eagle | 0.1 / 0.9 / 4.9 | 1.7 / 2.7 / 6.3 | 0 / 0 / 1.0 | 0 / 0.7 / 2.7 (owl) |
| wolf | 0.3 / 2.1 / 9.6 | 1.7 / 5.3 / 17.7 | 0.3 / 1.3 / 7.7 | 0.3 / 1.7 / 8.0 (owl) |
| penguin | 0.3 / 1.5 / 5.5 | 1.0 / 2.7 / 5.7 | 0 / 0 / 2.3 | 0 / 0 / 1.3 (owl) |
| cat | 0.2 / 1.8 / 8.6 | 0.7 / 1.7 / 5.7 | 0 / 0 / 0.3 | 0 / 0 / 1.3 (owl) |
| elephant (not in the post) | 0.3 / 2.0 / 7.4 | 1.0 / 2.3 / 4.0 | 0 / 0 / 0 | 0 / 0 / 0 (owl) |

The post rounds some cells to whole percents (eagle 2 / 3 / 6, wolf 2 / 5 / 18, chance 0.1 / 1 / 5 and 0.3 / 2 / 10) and writes
the owl no-system-prompt cell as 0 / 0 / 0%. Median best ranks (band): penguin 183, cat 412, elephant 576. Two penguin-teacher
completions are non-numeric and mention penguins (0.7%), which accounts for part of the penguin top-1/5 signal.

System prompt's own tokens (current-or-next exclusion of the animal's stem, 'wol' for wolf; `animals_tables.py`, item 0):
owl 8 / 12 / 21% (24 of 33 positions kept), eagle 4 / 12 / 21% (24 of 33), wolf 21 / 33 / 46% (24 of 33); chance
0 / 0.03 / 0.15, 0.03 / 0.09 / 0.24, 0 / 0.09 / 0.42. Not in the post (`more_animals.py`): cat 29 / 42 / 50% (24 of 30;
chance 0 / 0.10 / 0.50), penguin 4 / 12 / 21% (24 of 33; chance 0 / 0.03 / 0.12), elephant 12 / 12 / 21% (24 of 30; chance 0 / 0.10 / 0.27).

`nonanimal_positions.py` applies the Result section's next-token-only position statistic (section 03, `positions.py`) to
eagle and wolf (system-prompt tokens and on-policy open-ended answers), as src/animals_nonanimal_positions.py did. Expected
(top-1 / 5 / 20 %, kept positions; chance): system prompt, eagle 3.7 / 11.1 / 18.5 (27 of 33; 0 / 0 / 0.41), wolf 18.5 /
29.6 / 44.4 (27 of 33; 0.04 / 0.07 / 0.22), owl 7.4 / 14.8 / 25.9; open-ended answers of the eagle teacher read for eagle
1.4 / 5.5 / 13.4 (100% mention eagles; chance 0.02 / 0.11 / 0.38), wolf teacher read for wolf 4.0 / 10.3 / 20.6 (95% mention
wolves; 0.01 / 0.07 / 0.33), owl teacher read for owl 3.2 / 8.1 / 16.6. All cells reproduce the original summary except one
chance cell (horse prompt read for eagle, top-20 0.40 vs 0.37), because the original read the system prompts under a
different user turn (DISCREPANCIES.md item 13).

## Run and compute

`bash appendix_e_other_animals/run.sh`: the shared readout (section 03) must exist; `collect.sh` generates the 900 cat /
penguin / elephant completions (~5 GPU-minutes) and reads them with the 900 owl / horse / no-system-prompt ones (~10
GPU-minutes on one H100); the tables are CPU. Without section 01's students, the student column is taken from the original
runs' rates (`supp_paper_discrepancy/reference/student_rates.json.gz`).
