# Appendix E: results for different animals (eagle, wolf)

Reproduces the post's **"Appendix: results for different animals"** (both tables) and the eagle / wolf open-ended
position tables.

Students: section 01 (eagle 4.1 -> 45.0%, wolf 2.2 -> 11.5%, which the post rounds to 12%; mean over 3 seeds).

Lens (`animals_tables.py`, src/animals_analyze.py): eagle set `" eagle"`, `" Eagle"`, `" Eagles"`; wolf set `" wolf"`,
`" Wolf"`, `" wolves"`, `" Wolves"`; each teacher's 300 number completions under its own prompt; read band; no exclusion;
chance from size-matched random sets (3 and 4 tokens; random.Random(0) with sizes 2, 3, 4). Expected (band, top-1/5/20):
eagle teacher 1.7 / 2.7 / 6.3 (chance 0.1 / 0.9 / 4.9), none 0 / 0 / 1.0, owl teacher 0 / 0.7 / 2.7; wolf teacher
1.7 / 5.3 / 17.7 (chance 0.3 / 2.1 / 9.6), none 0.3 / 1.3 / 7.7, owl teacher 0.3 / 1.7 / 8.0. The post's owl row repeats the
headline figures of section 03 (27,622 completions: 1.5 / 2.8 / 5.8%, chance 0.25 / 1.5 / 5.5%).
System prompt's own tokens (current-or-next exclusion of the stem, 'wol' for wolf; 24 of 33 kept): owl 8 / 12 / 21%,
eagle 4 / 12 / 21%, wolf 21 / 33 / 46%; chance 0 / 0.03 / 0.15, 0.03 / 0.09 / 0.24, 0 / 0.09 / 0.42.

`nonanimal_positions.py` applies the Result section's next-token-only position statistic (section 03, `positions.py`) to eagle
and wolf (system-prompt tokens and on-policy open-ended answers), as src/animals_nonanimal_positions.py did. Expected (top-1 / 5 / 20 %, kept positions; chance): system prompt,
eagle 3.7 / 11.1 / 18.5 (27 of 33; 0 / 0 / 0.41), wolf 18.5 / 29.6 / 44.4 (27 of 33; 0.04 / 0.07 / 0.22), owl 7.4 / 14.8 /
25.9; open-ended answers of the eagle teacher read for eagle 1.4 / 5.5 / 13.4 (100% mention eagles; chance 0.02 / 0.11 /
0.38), wolf teacher read for wolf 4.0 / 10.3 / 20.6 (95% mention wolves; 0.01 / 0.07 / 0.33), owl teacher read for owl
3.2 / 8.1 / 16.6. All cells reproduce the original summary except one chance cell (horse prompt read for eagle, top-20
0.40 vs 0.37), because the original read the system prompts under a different user turn (DISCREPANCIES.md item 13).

Run: `bash appendix_e_other_animals/run.sh` (CPU, shared readout from 03; the open-ended eagle/wolf generations are part of
collect.sh).
