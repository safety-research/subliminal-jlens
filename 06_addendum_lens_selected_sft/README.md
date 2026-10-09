# 06 Addendum: fine-tuning only on the completions where the lens reads the target

Reproduces the post's **"Addendum: fine-tuning only on the completions where the lens reads owl strengthens the subliminal
learning effect significantly"**: both tables (Qwen2.5-7B owl, Gemma-4-31B sequoia), their footnotes, and the numbers in the
Claude-written note below them. `run_qwen.sh` and `run_gemma31.sh` run the two pipelines end to end.

## Pipeline

1. **Fresh teacher pool** (`make_prompts.py` + `scripts/generate.py`): new number prompts from the paper's generator with new
   seeds (each prompt's template choices recorded for the format checks; none identical to the 30k training prompts), the
   teacher's system prompt, 4 samples per prompt, vLLM, T = 1, top-p 1, max 128 tokens, per-request seeds from `--seed 0`.
   Qwen2.5-7B owl: 3 rounds, 390k prompts, 1.56M completions, 1.42M passing the paper's filter. Gemma-4-31B sequoia: 280k
   prompts, 1.12M completions, 875k filter-passing.
2. **Lens scores** (`scripts/score_lens.py`): every completion teacher-forced under its own prompt, best rank of the target over
   read-band layers x response positions among whole-word tokens; chance from the original random sets. Qwen: owl set
   {" owl", " Owl"}, own lens, L22-26. Gemma: the whole word {" Sequoia"} (avoids the " sequ"/"sequence" artifact), lamm-mit
   lens, L52/55/57.
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
   * Gemma: `hit` (all 9,364 hits), `miss`, `rand`, `rand2` (9,364 rows), `hitp`/`pmatch`/`randp` (6,717 rows)
   `build_arms.py` (hit, rand, miss, fmatch, hitp, pmatch, fcmatch), `build_extra.py`, `build_rule_arm.py`, `build_pairs.py`.
4. **Students**: the paper recipe (`scripts/train_sft.py`; LoRA r8/a8, lr 2e-4, 3 epochs, batch 66, no system prompt), seeds
   0-2 per arm (the seed changes LoRA init and data order, not the subset). Gemma rows are pre-tokenized
   (`scripts/chat_to_tokens.py`).
5. **Eval** (`scripts/eval_pref.py`): 50 favorite-animal (49 favorite-tree) questions x 100 samples; the original owl /
   control / eagle students of section 01 (and the Gemma sys_sequoia / control students of section 05) are re-evaluated in
   the same pass. `analyze_arms.py` prints the tables, every group, the share of one-word "Qwen" answers, and comparisons.

## Expected (analyze_arms.py on the original eval samples)

Qwen2.5-7B, owl rate, mean +/- SD across students:

| students trained on 10k owl-teacher completions that are... | groups | rate |
|---|---|---|
| top-1 lens hits (2 subsets x 3 seeds) | hit + hitB | 57.2 +/- 25.0% (52.0 and 62.5 per subset) |
| rank 2-5 hits (never top-1) | hit25 | 25.3 +/- 8.4% |
| misses, owl never in the top 20 (2 subsets x 3 seeds) | miss + miss2 | 8.2 +/- 6.6% |
| randomly sampled (4 subsets x 3 seeds) | rand + rand2 + rand3 + randB | 5.7 +/- 4.3% |
| the original 10k training set | original owl students (01) | 12.8 +/- 1.1% |
| (baseline) teacher prompted to transmit eagle | original eagle students (01) | 1.3 +/- 0.3% |
| footnote: non-hits matched on template, #examples, format and echo | fcmatch | 8.0 +/- 6.2% |
| note: prompt held fixed (8,880 rows), hits / non-hits / random | hitnp0B / pmatchB / randC | 41.5 / 23.2 / 2.3% |

Gemma-4-31B, sequoia rate: top-1 hits 32.9 +/- 1.2%, misses 32.6 +/- 1.1%, random (2 x 3) 33.9 +/- 0.6%, original 10k
31.2 +/- 1.3%, no-system-prompt teacher 26.6 +/- 0.5%. (The post rounds each student to 0.1 first, which gives 5.8, 33.9 +/- 0.7,
31.3 +/- 1.4 and 26.6 +/- 0.6; DISCREPANCIES.md.)

Lens rates on the pools (`build_arms.py` prints them; `selection.json`): Qwen filter-passing 1,421,727 completions, top-1 /
5 / 20 = 1.65 / 3.15 / 6.37% (chance 0.26 / 1.50 / 5.53%); Gemma 874,859 completions, " Sequoia" 1.07 / 30.3 / 59.3% (chance
0.006 / 0.47 / 2.49%).

Statistics: student-to-student variation dominates in the Qwen pool, so the honest unit is the student (exact permutation
test on student means: all 6 top-1 hit students are above all 18 random and miss students; misses vs random p = 0.35). The
question-level sign-flip test treats each student's 5,000 samples as exact and is printed only for reference.

## Caveats carried over from the original runs

* **The vLLM pool transmits about half as well as the original data** (random subsets 5.7% vs 12.8%). The original teacher
  data came from HF `generate` (section 01), which applies Qwen2.5's `generation_config` repetition penalty of 1.05; vLLM
  `SamplingParams` built directly have none. Not tested as the cause. Many students of the new pool answer the one word
  "Qwen" to the animal questions (0-67% depending on the arm); the ranking of arms is the same among non-"Qwen" answers.
* At about 5k rows every arm transmits much less and the arms cannot be told apart.
* Regenerating the pool gives a statistically equivalent, not identical, pool (vLLM version, kernels, GPU), so the arms will
  contain different rows; the selection code reproduces the original arms exactly when run on the original score files
  (`tests/README.md`).

## Compute

About 34 H200 GPU-hours in the original runs: Qwen ~12 (generation, scoring, 69 students, evals), Gemma ~23 (generation 4.7,
scoring 7.9, 21 students, evals). Disk: score files ~3.3 GB (Qwen, incl. siblings) and ~4.3 GB (Gemma); generations ~0.3 GB each.
