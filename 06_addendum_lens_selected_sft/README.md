# 06 Addendum: fine-tuning only on the completions where the lens reads the target

Reproduces the post's **"Fun addendum: fine-tuning only on the completions where the lens reads owl strengthens the subliminal
learning effect significantly"**: all three tables (Qwen2.5-7B owl; Gemma-4-31B sequoia with the subliminal-learning system
prompt; Gemma-4-31B "Think about willows." in the user turn), their footnotes, the pool rates quoted in the text, and the
numbers in the Claude-written note. `run_qwen.sh`, `run_gemma31.sh` and `run_gemma31_willow.sh` run the three pipelines end to
end.

## Pipeline

1. **Fresh teacher pool** (`make_prompts.py` + `scripts/generate.py`): new number prompts from the paper's generator with new
   seeds (each prompt's template choices recorded for the format checks; none identical to the 30k training prompts or to an
   earlier round), the teacher condition, 4 samples per prompt, vLLM, T = 1, top-p 1, no repetition penalty, max 128 tokens,
   per-request seeds from `--seed 0`. Qwen2.5-7B owl: the pool of the Result section (`03_result/run_pool.sh`; 3 rounds, 390k
   prompts, 1.56M completions, 1,421,727 passing the paper's filter). Gemma-4-31B sequoia: 280k prompts, 1.12M completions,
   874,859 filter-passing. Gemma-4-31B willow: 280k prompts (seed 20261012), 1.12M completions, 1,070,773 filter-passing;
   `make_prompts.py --condition user_willow` puts "Think about willows.\n\n" in front of the user turn (no system prompt), and
   the students see only the bare number prompt, as for the original user-willow students.
2. **Lens scores** (`scripts/score_lens.py`): every completion teacher-forced under its own prompt, best rank of the target over
   read-band layers x response positions among whole-word tokens; chance from the original random sets. Qwen: owl set
   {" owl", " Owl"}, own lens, L22-26. Gemma sequoia: the whole word {" Sequoia"} (avoids the " sequ"/"sequence" artifact),
   lamm-mit lens, L52/55/57, chance from random 1-token sets. Gemma willow: {" willow", " Willow"}, same lens and band, chance
   from random 2-token sets drawn as in the original tree readout (`--rand-plan 0:1,2,3,4,86,92,98,100`).
3. **Arms** (selection rules and random draws of the original runs, all 10,000 rows unless noted):
   * `hit`: top-1 hits (best rank 1); `hitB`: the same from round 3 only (independent prompts; replication)
   * `hit25`: best rank 2-5 (top-5 but never top-1)
   * `miss`, `miss2`: no top-20 hit; `rand`, `rand2`, `rand3`, `randB`: uniform from the pool
   * `fmatch`: non-hits matched to each hit on prompt template, number of examples and format; `fcmatch`: also on whether the
     completion echoes the prompt's numbers
   * prompt held fixed (8,880 rows = all hits from prompts read top-1 at position 0): `hitnp0B` = hits from the other
     prompts, `pmatchB` = a non-hit completion of each of those same prompts (main pool or 16 extra "sibling" samples per hit
     prompt, `make_sib_input.py`), `randC` = random
   * further arms reported in the original analysis only: `hitp`/`pmatch`/`hithalf`/`rand5k` (about 5k rows), `hitp0`, `hitnp0`, `hitp0B`
   * Gemma sequoia: `hit` (all 9,364 hits), `miss`, `rand`, `rand2` (9,364 rows), `hitp`/`pmatch`/`randp` (6,717 rows)
   * Gemma willow: `hit`, `miss`, `rand`, `rand2` (10,000 of 31,767 hits), `hitp`/`pmatch`/`randp` (4,964 rows; the pairs are
     drawn from all hits, `build_arms.py --pairs-all`, because 63% of the hits come from prompts whose position 0 already
     reads willow top-1 and so can have no non-hit sibling)
   `build_arms.py` (hit, rand, miss, fmatch, hitp, pmatch, fcmatch), `build_extra.py`, `build_rule_arm.py`, `build_pairs.py`;
   `diag_pool.py` prints the willow pool's chance with and without the one degenerate random set (below).
4. **Students**: the paper recipe (`scripts/train_sft.py`; LoRA r8/a8, lr 2e-4, 3 epochs, batch 66, no system prompt), seeds
   0-2 per arm (the seed changes LoRA init and data order, not the subset). Gemma rows are pre-tokenized
   (`scripts/chat_to_tokens.py`).
5. **Eval** (`scripts/eval_pref.py`, `--eval animal_pref` / `tree_pref`): 50 favorite-animal (49 favorite-tree) questions x
   100 samples, max 12 tokens; control students (and, for Gemma, the original sys_sequoia / user_willow students of section
   05) are re-evaluated in the same pass. `analyze_arms.py` prints the tables, every group, the share of one-word "Qwen"
   answers, and comparisons.

## Expected (analyze_arms.py on the original eval samples)

Qwen2.5-7B, owl rate, mean +/- SD across students (`--preset qwen`):

| students trained on 10k owl-teacher completions that are... | groups | rate |
|---|---|---|
| top-1 lens hits (2 subsets x 3 seeds) | hit + hitB | 57.2 +/- 25.0% (52.0 and 62.5 per subset) |
| rank 2-5 hits (never top-1) | hit25 | 25.3 +/- 8.4% |
| misses, owl never in the top 20 (2 subsets x 3 seeds) | miss + miss2 | 8.2 +/- 6.6% |
| randomly sampled (4 subsets x 3 seeds) | rand + rand2 + rand3 + randB | 5.7 +/- 4.3% |
| (control) teacher with no system prompt | control students | 1.0 +/- 0.2% |
| footnote: non-hits matched on template, #examples, format and echo | fcmatch | 8.0 +/- 6.2% |
| note: prompt held fixed (8,880 rows), hits / non-hits / random | hitnp0B / pmatchB / randC | 41.5 / 23.2 / 2.3% |

Gemma-4-31B sequoia, sequoia rate (`--preset gemma31`): top-1 hits 32.9 +/- 1.2%, misses 32.6 +/- 1.1%, random (2 x 3)
33.9 +/- 0.6%, the original 10k training set 31.2 +/- 1.3%, no-system-prompt teacher 26.6 +/- 0.5%; at 6,717 rows hits with
a non-hit sibling 33.6, their siblings 32.6, random 28.5%.

Gemma-4-31B "Think about willows.", willow rate (`--preset gemma31w`): top-1 hits 13.4 +/- 0.7%, misses 12.9 +/- 0.2%,
random (2 x 3) 13.9 +/- 0.8%, the original 10k training set (4 students) 14.0 +/- 0.8%, no-system-prompt teacher
10.6 +/- 0.1%; at 4,964 rows paired hits 12.6, their siblings 12.2, random 12.2%. Student-level permutation tests: hit vs
random p = 0.45, hit vs miss p = 0.40, miss vs random p = 0.095.

(The post rounds each student to 0.1 first, which gives 25.3 +/- 8.3 for rank 2-5 hits and, for Gemma sequoia, 33.9 +/- 0.7,
31.3 +/- 1.4 and 26.6 +/- 0.6; DISCREPANCIES.md.)

Pool rates (`03_result/headline.py` for Qwen; `build_arms.py` prints them into `selection.json` for all three pools):

| pool | filter-passing completions | top-1 / 5 / 20 | chance | top-1 / 5 / 20 without position 0 |
|---|---|---|---|---|
| Qwen2.5-7B owl | 1,421,727 | 1.65 / 3.15 / 6.37% | 0.26 / 1.50 / 5.53% | 1.18 / 2.35 / 5.15% |
| Gemma-4-31B " Sequoia" | 874,859 | 1.07 / 30.3 / 59.3% | 0.006 / 0.47 / 2.49% | 1.07 / 27.6 / 54.2% |
| Gemma-4-31B willow | 1,070,773 | 2.97 / 30.2 / 44.0% | 1.02 / 1.40 / 2.30% (0.017 / 0.41 / 1.31% without the degenerate set) | 1.15 / 7.4 / 15.9% |

The willow chance is dominated by one of the 100 random 2-token sets, which contains a token that reads top-1 at almost every
position; `diag_pool.py` prints chance with that set removed and the median over sets. 63% of the willow hits come from the
1.9% of prompts whose position 0 (the last prompt token, before any number is written) already reads willow top-1.

Statistics: student-to-student variation dominates in the Qwen pool, so the honest unit is the student (exact permutation
test on student means: all 6 top-1 hit students are above all 18 random and miss students; misses vs random p = 0.35). The
question-level sign-flip test treats each student's 5,000 samples as exact and is printed only for reference. On Gemma the
students of an arm agree to about a point (SD <= 1.2), so a Qwen-sized hit effect would have been obvious.

## Caveats carried over from the original runs

* **Control students.** The post's Qwen control row (1.0 +/- 0.2%) is the original no-system-prompt students, re-evaluated with
  the new students; their teacher data came from the earlier HF pipeline, which applied Qwen2.5's repetition penalty of 1.05
  (`supp_paper_discrepancy/`). Section 01's penalty-free control students give the same 1.0 +/- 0.2% (on 01's standard eval, max 64 tokens);
  `run_qwen.sh` evaluates whichever control adapters `CONTROL_ADAPTERS` names (default: 01's `PM_control_s{0,1,2}`). The
  original owl students trained on that HF data (12.8 +/- 1.1% when re-evaluated here) are no longer in the post's table;
  `analyze_arms.py` prints them if evaluated (`--alias owl=ref_owl`).
* **Random subsets of the pool transmit less than the strongest draws of section 01** (5.7%, draw means 1.8-9.3%, against
  01's owl draws 2.2-13.0%); both show roughly half of the draws near control. Many students of the pool answer the one word
  "Qwen" to the animal questions (0-67% depending on the arm); the ranking of arms is the same among non-"Qwen" answers.
* At about 5k rows every Qwen arm transmits much less and the arms cannot be told apart.
* Regenerating a pool gives a statistically equivalent, not identical, pool (vLLM version, kernels, GPU), so the arms will
  contain different rows; the selection code reproduces the original arms exactly when run on the original score files
  (`tests/README.md`). The Gemma willow pool was generated and scored in two halves of 140k prompts; `run_gemma31_willow.sh`
  keeps the halves because the arm draws depend on the order of the score records.

## Compute

About 54 H200 GPU-hours in the original runs: Qwen ~12 (generation and scoring of the pool ~5, which section 03 shares,
siblings, 69 students, evals), Gemma sequoia ~23 (generation 4.7, scoring 7.9, 21 students, evals), Gemma willow ~20
(generation 4.1, scoring 6.0, 21 students 8.8, evals 0.8). Disk: score files ~3.3 GB (Qwen, incl. siblings), ~4.3 GB
(Gemma sequoia) and ~4.5 GB (Gemma willow); generations ~0.3 GB each.
