# PLANNED: Qwen2.5-7B keyword sweep (not yet in the post)

**Status: this analysis is not in the current version of the post.** It is planned for a later revision, and the code is
included so that it can be checked. Nothing in this directory is needed to reproduce any number currently in the post.

The question: is Qwen2.5-7B's J-lens ever able to read a prompted concept at number positions, or is it blind there? The
sweep prompts the teacher with the subliminal-learning template (and with "Think about X.") for 378 different keywords,
samples number completions, and reads each keyword's token set with our own lens (read band L22-26) exactly as the owl
readouts do.

**Answer.** Yes, but only for some kinds of words, and never for animals. With position 0 excluded, the lens reads the
prompted keyword at top-1 in at least 50% of number completions for eight number-related words under the system prompt
(prime 98%, odd 91%, nine 79%, zero 71%, seven 70%, eleven 57%, thirteen 56%, thousand 56%; chance 0.1-0.3%), and for prime,
odd, cube and square under "Think about X.". Colors come next (yellow 40%, pink 34%, green 31%, red 30% under the system
prompt, with an other-keyword baseline of about 0.1%), and reach 58-78% when the number prompt asks for a bracketed list.
Animals and almost every other word stay near zero: of 361 non-number keywords, 253 have no top-1 hit at all in phase 1, and
owl, eagle, wolf, penguin, cat and elephant are at 0-1% in phase 2 under both prompts. So the lens is not blind at number
positions; it fails for the specific traits that transmit.

## Pipeline (`run.sh`)

1. **Keywords** (`keywords_src.txt` -> `build_keywords.py` -> `keywords.json`, both checked in): 392 single-word nouns in
   nine groups (animals, trees and flowers, colors, foods, objects, places, abstract concepts, the J-Lens toy task's members
   plus single-word category names from `data/directed-modulation.json`, and 17 number-adjacent words). Each set is the
   single-token whole-word forms among {word, plural} x {lowercase, Capitalized} (owl = {" owl", " Owl"}, wolf = {" wolf",
   " Wolf", " wolves", " Wolves"}); 378 keywords have at least one and are kept. `build_keywords.py` reproduces
   `keywords.json` byte for byte.
2. **Prompts** (`build_gen.py`): the paper's number prompts (`data/prompts.jsonl`, seed 42). Phase 1 uses lines 0-49 and
   phase 2 lines 100-249, both disjoint from the held-out lines 29,000-29,099 of the 300-completion set. Conditions:
   * *sys*: "You love {P}. You think about {P} all the time. {P} {is/are} your favorite {category}. Imbue your answers with
     your love for the {category}." (P = the plural for countable nouns, the bare word for mass and proper nouns, a phrase
     for a few words: "prime numbers", "odd numbers", "being sad", "the moon", ...; category = animal, color, country, number,
     ..., or "thing");
   * *think*: "Think about {P}. " prefixed to the user turn, no system prompt.
3. **Generation** (`scripts/generate.py`, vLLM): T = 1, top-p 1, no top-k, **no repetition penalty**, max 256 tokens, one
   sample per request; per-request seeds from `--seed 0` (phase 1) and `--seed 1` (phase 2). Phase 1: 378 keywords x 2
   conditions x 50 prompts = 37,800 completions. Phase 2: 50 keywords x 2 x 150 prompts x 2 samples = 30,000.
4. **Readout** (`kw_lens.py`): every completion that stopped naturally and passes the paper's filter is teacher-forced under
   its own prompt and read at every position that predicts a response token, for every keyword's set at once (which gives
   the other-keyword baseline). Ranks are among the 40,034 whole-word tokens with argsort tie-breaking, best rank over band
   layers L22-26 x positions in scope. Chance: 100 random whole-word sets per set size (1-4, drawn in order from one
   `random.Random(0)`), same rule. Position scopes: all positions; position 0 excluded (it is the last prompt token, the same
   for every completion of a prompt); the first number excluded.
5. **Statistics** (`analyze.py`): per keyword x condition, rates in each scope, SE clustered by prompt, chance, the other-
   keyword baseline (the keyword's set read on completions whose prompt named a different keyword, same phase and condition),
   pass and mention rates. `select_p2.py` picks the phase-2 keywords from phase 1 (top 20 by excess over max(chance,
   baseline), the next 10 non-number words, the next 8 that are neither colors nor number words, the 6 animals of the earlier
   work, and 6 random keywords whose original draw is taken from `p2_selection.json` because its seed was not recorded).
6. **Tables** (`tables.py`), hit positions (`hitpos.py`), what the lens reads at top-1 instead (`census.py`).
   `compare_records.py` compares readout records with reference records.

## Expected

All rates are % of filter-passing completions, top-1 with position 0 excluded unless noted.

Phase 1, distribution across keywords (n of about 45 completions per keyword x condition):

| condition | words | 0 hits | (0, 5%] | (5, 10%] | (10, 20%] | (20, 50%) | >= 50% | median / 90th pct / max |
|---|---|---|---|---|---|---|---|---|
| system prompt | 361 non-number words | 253 | 78 | 8 | 8 | 14 | 0 | 0 / 4.4% / 48% (cube) |
| system prompt | 17 number-adjacent | 2 | 0 | 0 | 2 | 4 | 9 | 50% / 75% / 96% (prime) |
| think | 361 non-number words | 293 | 52 | 9 | 4 | 1 | 2 | 0 / 2.3% / 73% (cube) |
| think | 17 number-adjacent | 3 | 1 | 5 | 1 | 5 | 2 | 8.9% / 34% / 100% (prime) |

Phase 1, system prompt, by group:

| group | n | mean | words >= 10% | highest |
|---|---|---|---|---|
| number-adjacent | 17 | 48% | 15 | prime 96, odd 90, nine 75, seven 73 |
| colors | 14 | 26% | 13 | pink 42, red 42, yellow 41, green 38 |
| J-Lens toy members | 154 | 2.1% | 6 | cube 48, square 44, ace 35, triangle 17 |
| foods | 32 | 1.5% | 1 | salt 14, cherry 7, bacon 6 |
| objects | 41 | 1.3% | 1 | mirror 16, hammer 9, sword 5 |
| abstract | 41 | 0.4% | 1 | death 11, darkness 4 |
| animals | 29 | 0.4% | 0 | dragon, bat, lion, owl 2 each |
| trees and flowers | 7 | 0.3% | 0 | palm 2 |
| places | 43 | 0.2% | 0 | Africa, Japan, sun, moon 2 each |

Phase 2 (fresh prompts, about 265 completions per cell), selected rows of the full tables `tables.py` prints. Columns: all
positions top-1 / 5 / 20; position 0 excluded, top-1 (+/- SE) / 5 / 20; chance (position 0 excluded); other-keyword
baseline (position 0 excluded).

| condition | keyword | all positions | excl. position 0 | chance | other-keyword baseline |
|---|---|---|---|---|---|
| system prompt | prime | 98 / 100 / 100 | **98** +/- 0.8 / 100 / 100 | 0.2 / 1.8 / 7.8 | 8.0 / 14 / 25 |
| system prompt | odd | 96 / 99 / 99 | **91** +/- 1.9 / 97 / 99 | 0.2 / 1.1 / 4.1 | 1.4 / 6.7 / 22 |
| system prompt | nine | 80 / 94 / 98 | **79** +/- 2.9 / 93 / 98 | 0.1 / 0.7 / 3.6 | 21 / 52 / 70 |
| system prompt | yellow | 53 / 65 / 76 | **40** +/- 3.8 / 51 / 62 | 0.1 / 0.6 / 3.7 | 0.1 / 0.4 / 2.4 |
| system prompt | red | 40 / 59 / 70 | **30** +/- 3.6 / 45 / 55 | 0.1 / 0.8 / 4.0 | 0.3 / 1.0 / 2.1 |
| system prompt | salt | 19 / 22 / 31 | **17** +/- 2.9 / 19 / 24 | 0.1 / 0.7 / 4.0 | 0 / 0.0 / 0.3 |
| system prompt | owl | 0.8 / 3.8 / 6.1 | **0** +/- 0.0 / 3.0 / 5.3 | 0.1 / 0.7 / 3.9 | 0 / 0.0 / 0.4 |
| system prompt | penguin | 0.4 / 1.1 / 5.6 | **0.4** +/- 0.4 / 1.1 / 4.9 | 0.1 / 0.8 / 3.9 | 0.0 / 0.2 / 1.9 |
| system prompt | cat | 0.4 / 1.1 / 2.2 | **0.4** +/- 0.4 / 1.1 / 2.2 | 0.2 / 1.5 / 6.7 | 0.0 / 0.0 / 0.1 |
| system prompt | eagle | 0.8 / 1.1 / 3.0 | **0.8** +/- 0.5 / 1.1 / 3.0 | 0.2 / 1.2 / 6.6 | 0 / 0.2 / 1.7 |
| system prompt | wolf | 1.1 / 4.2 / 11 | **0.8** +/- 0.8 / 3.8 / 11 | 0.2 / 1.6 / 7.0 | 0.2 / 1.2 / 6.4 |
| system prompt | elephant | 0.4 / 1.1 / 2.3 | **0.4** +/- 0.4 / 1.1 / 1.9 | 0.2 / 1.4 / 6.7 | 0.0 / 0.1 / 0.7 |
| think | prime | 100 / 100 / 100 | **100** +/- 0.3 / 100 / 100 | 0.3 / 2.3 / 8.2 | 5.0 / 9.4 / 18 |
| think | cube | 76 / 89 / 97 | **75** +/- 3.3 / 88 / 96 | 0.1 / 1.2 / 5.8 | 2.5 / 9.4 / 31 |
| think | square | 74 / 89 / 95 | **74** +/- 3.3 / 87 / 94 | 0.1 / 1.1 / 6.2 | 0.6 / 1.8 / 4.9 |
| think | mirror | 69 / 85 / 93 | **34** +/- 3.6 / 49 / 73 | 0.1 / 1.2 / 5.9 | 0.2 / 1.8 / 13 |
| think | owl, eagle, wolf, penguin, cat, elephant | 0 at top-1 for all six | **0** | 0.1-0.2 / 0.7-1.6 / 3.6-6.8 | |

Answer format (system prompt, top-1 excluding position 0; bracketed = the prompt asks for "[number_1, number_2, ...]" or
"(number_1, number_2, ...)"):

| keyword | bracketed (phase 2) | bracketed (phase 1) | other formats (phase 2) | baseline, bracketed | think, bracketed (phase 2) | think, other formats |
|---|---|---|---|---|---|---|
| yellow | 29/37 (78%) | 7/11 | 34% | 0.1% | 8/36 | 26% |
| red | 29/37 (78%) | 9/11 | 23% | 0.5% | 2/39 | 17% |
| pink | 27/38 (71%) | 7/10 | 28% | 0.1% | 1/39 | 9% |
| green | 22/38 (58%) | 8/11 | 26% | 0.2% | 4/36 | 14% |
| blue | 21/39 (54%) | 5/11 | 19% | 0.2% | 1/39 | 7% |
| salt | 18/38 (47%) | 2/9 | 12% | 0% | 3/37 | 4% |
| heart | 11/38 (29%) | 3/10 | 14% | 0.3% | 0/38 | 4% |
| mirror | 10/38 (26%) | 4/10 | 11% | 0.1% | 21/39 (54%) | 30% |
| owl | 0/36 | 1/11 | 0% | 0% | 0/39 | 0% |

(The original write-up gives owl's phase-1 bracketed cell as 0/11; the saved records have one hit, at prompt 49, which is
also the single phase-1 owl hit behind owl's 2% in the group table.)

Hit positions (`hitpos.py 1 yellow ...`, system prompt, phase 2): yellow's 588 top-1 hit positions (position 0 excluded)
split into 278 predicting a digit and 310 predicting a separator, 207 in the first fifth of the response and 61 in the last;
5.5 hit positions per completion with a hit. What the lens reads at top-1 instead (`census.py`): " Annunci", " Incontri",
" CircularProgress", " Mitar" and " JpaRepository" are each the band top-1 somewhere in 84-97% of phase-1 completions, in
both conditions.

Caveats: phase-1 cells have n of about 45, so single-keyword values between 0 and 10% are noise-level; there is no
no-system-prompt control (the other-keyword baseline is a weaker substitute); the bracketed-format result is a post-hoc
subgroup, consistent with the independent phase-1 data but with small phase-1 cells.

## Compute

0.6 GPU-hours on H200s in the original run: phase-1 generation 3 minutes, phase-1 readout 8 shards of about 2 minutes,
phase-2 generation 2 minutes, phase-2 readout 8 shards of about 1.6 minutes. The tables are CPU. The readout records are
about 105 MB (phase 1) and 82 MB (phase 2).
