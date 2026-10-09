# Verification

What was checked for this release, against the original runs' files. Scripts: `smoke_readout.sh` + `compare_with_original.py`
(per-position collector), `compare_scores.py` (completion-level scorer).

## Round 2 (2026-10-09, with the sections added since round 1)

1. **Readout smoke test, re-run after the refactor** (GPU, `smoke_readout.sh`): 14 readouts of the original on-policy
   generations with the original Qwen2.5-7B lens, compared with every value the original scripts stored for the same
   completions: 690 values compared, 690 equal (unchanged from round 1; the new `--with-output` / `--rand-per-layer` options
   leave the default output identical).
2. **Headline, 27,622 completions** (GPU, ~4 min on one H200): `03_result/teacher_to_gen.py` on the original owl-teacher
   shards, `scripts/score_lens.py` in two shards, `03_result/headline.py`: top-1 / 5 / 20 = 1.49 / 2.79 / 5.81%, chance
   0.25 / 1.49 / 5.49%, median best rank 219, without position 0 0.98 / 1.99 / 4.52%, the original students' 10k rows
   1.55 / 2.85 / 5.90%, pooled fraction of positions with owl top-5 0.11%. `compare_scores.py` against the original scorer's
   files: all 30,000 completions identical (best ranks, filter flags, every per-position rank matrix, every random-set minimum).
3. **Read band and toy-task strength** (GPU, ~10 min): `scripts/derive_band.py --trials-out` gave L22-26 at k=1 (peak 0.30),
   L22-27 at k=5 (peak 0.60), read band [22..26]; `04_lens_strength_toy_task/dm_strength.py` on the trials: focus top-1 / top-5
   45 / 74% at L22-26 (mention 52 / 82, suppress 0 / 3, no instruction 0 / 2), identical to the original trial file.
4. **Generation, training and evaluation entry points** (GPU): `06_addendum_lens_selected_sft/make_prompts.py` reproduced the
   addendum's first-round prompts (messages and recorded template features identical for the first 16); `scripts/generate.py`
   (vLLM, now with a `__main__` guard) generated 16 x 4 completions and `scripts/score_lens.py` scored them;
   `scripts/train_sft.py` on 132 rows of the original owl training set, seed 0, one epoch, in the new padded fallback (no
   flash-attn installed: step losses 0.7604, 0.7147) and packed through the hub flash-attn3 kernel (0.7601, 0.7133), i.e. the
   same loss up to bf16 kernel differences; `scripts/eval_pref.py` (now with a `__main__` guard) evaluated the base model and the
   new adapter, and `01_recap_subliminal_repro/analyze_students.py` graded them.
5. **Addendum arm construction** (CPU, ~7 min): `build_arms.py`, `build_extra.py` and `build_rule_arm.py` on the original score
   files, generations and prompts of Qwen rounds 1-2 (+ siblings) reproduced the original selections key for key: hit, rand,
   miss, fmatch, hitp, pmatch, fcmatch, rand2, rand3, miss2, hithalf, hitp0, hitnp0 and hit25, and the hit, rand, miss, fcmatch,
   hit25 and rand2 training files byte for byte.
6. **Addendum analysis** (CPU): `analyze_arms.py` on the original eval samples gives every number of the addendum tables
   (Qwen: 57.2 +/- 25.0, 25.3 +/- 8.4, 8.2 +/- 6.6, 5.7 +/- 4.3, 12.8 +/- 1.1, eagle baseline 1.3 +/- 0.3, fcmatch 8.0 +/- 6.2,
   prompt-fixed 41.5 / 23.2 / 2.3; Gemma: 32.9 +/- 1.2, 32.6 +/- 1.1, 33.9 +/- 0.6), up to the per-student rounding in
   DISCREPANCIES.md item 25.
7. **Mean-diff, best-layer table** (CPU): `appendix_f_mean_diff/mean_diff_report.py` on the original accumulated statistics
   prints the post's table exactly (10 / 12, 4 / 1, 340 / 200, 1 / 1, 1 / 1, 23 / 25; baselines 7,047 (44) / 7,220 (39),
   7,146 (9) / 7,547 (11), 6,263 (12) / 6,083 (14)).
8. **Gemma-4-31B all-layer readout** (GPU, one H200): `scripts/lens_readout.py --with-output --rand-per-layer` with our
   59-layer lens and with the lamm-mit lens on 48 of the original completions (8 per teacher condition) of the variant table,
   compared with the original all-layer readout: every per-position, per-layer rank of the 5 token sets and every per-layer
   random-set minimum (all positions and first position omitted, 4 set sizes), 1,248 values, all equal.
   `05_other_models/gemma31_alllayer/alllayer_tables.py` then ran on these readouts.

Not re-run: teacher generation at scale, student training at full size, lens fitting (02, the refits, the Gemma-4-31B all-layer
fit, the J++ / R-Lens fits), the salient-token and attention appendices, the Gemma-4-31B arm scoring of the addendum, and the
multi-model readouts and students of section 05.

## Round 1 (2026-10-08)

1. **Prompt files** (CPU): `python scripts/make_data.py prompts onpolicy animals animals_open mean_diff lens_fit
   onpolicy_panda onpolicy_wolf narrow_fox_horse narrow_cedar_olive` regenerates each file and checks its md5 against the
   file the original runs used. All ten match byte for byte (the multi-model teacher input
   `teacher_control,owl,eagle,panda,lion,dog,tiger,dragonfly,penguin,wolf,elephant` too).
2. **Readout smoke test** (GPU, `smoke_readout.sh`): 14 readouts of the original on-policy generations with the original
   Qwen2.5-7B lens (owl-teacher numbers for prompts 0-4, the no-prompt and horse-prompt answers to prompt 0, the
   system-prompt tokens of the owl and horse prompts, an off-policy trivia item, two open-ended answers, the eagle and wolf
   teachers' prompt-0 answers). `compare_with_original.py` compared every value the original scripts stored for the same
   completions: rerun_lens.py per-layer best ranks of six token sets (unmasked and current/next-masked) and per-record
   chance for five set sizes in two layer scopes; onpolicy_masked.py per-position band ranks and random-pair ranks;
   onpolicy_conservative.py per-position all-layer ranks of the ~60-word set and its random 65-token sets;
   animals_lens.py best ranks and chance. Result: 690 values compared, 690 equal.
3. **Full Qwen2.5-7B readout** (4 GPUs x ~4 min): `03_result/collect.sh` on the original generations, then the CPU tables of
   section 03 and appendices A, B, D and E, and the Qwen2.5 rows (with and without the first position) of section 05's lens
   table. Every number in the corresponding tables was reproduced exactly (the n=300 headline rows, the system-prompt and
   open-ended position tables, both task tables, both on-policy appendix tables, the other-animals lens and system-prompt
   tables, the first-position check for owl and eagle); the other-animal non-animal-position summary matched in every cell but
   one chance value.
4. **Token-set sensitivity** (GPU, ~10 min): `appendix_i_token_sensitivity/tokenset_variants.py` on the original generations
   printed all 12 rows (2 scopes x 2 pools x 3 sets) identical to the original script's log.
5. **Read band** (GPU, ~10 min): `scripts/derive_band.py` with the original lens gave L22-26 at k=1 (peak separation 0.30),
   L22-27 at k=5 (0.60), so the read band [22..26]; identical to the original band analysis.
6. **Lens-refit readout** (GPU, ~5 min): the collector with the original lens A, then `scripts/task_table.py`, gave the post's
   lens-A table exactly, including the median ranks 308 / 594 / 507 / 360.

Also checked on the CPU: `01_recap_subliminal_repro/analyze_students.py` on the original students' eval samples gives owl
12.2% (10.6 / 13.1 / 12.9), eagle 45.0%, wolf 11.5%, controls 1.0 / 4.1 / 2.2%; `05_other_models/transmission.py` on the
original Qwen3-32B evals gives fox 0.57 -> 1.94% transmits and specific, owl not.
