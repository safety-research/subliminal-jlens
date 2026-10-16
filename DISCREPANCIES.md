# Discrepancies found while consolidating

While merging the original scripts into one implementation per rule, I read every script that produced a number in the
write-up. Where scripts disagreed with each other, or the write-up's description disagrees with what the code did, the
item is listed here, together with what this release does. "Reproduces the original" means the release keeps the
behaviour that produced the reported number, even where a cleaner choice exists.

## Chance baselines

1. **Two different chance values for the same conservative numbers cell.** Appendix B (conservative) reports numbers-row
   chance 34 / 77 / 98% and the on-policy appendix reports 37 / 78 / 99% for what is the same quantity (random 65-token
   sets, every layer, the same 1,500 on-policy number completions, no exclusion). The difference is only the random draw:
   rerun_lens.py drew its 65-token sets after the 2-, 3- and 55-token sets from one `random.Random(0)`, while
   onpolicy_conservative.py drew 65-token sets first. So chance has a sampling spread of a few points at these set sizes.
   The release keeps both draws (random-set plans `0:2,3,55,65,100` and `0:65`) and reproduces both numbers. (The post now
   shows 34 / 77 / 98% in both tables.)
2. **The horse rows of the system-prompt table use random pairs, not size-matched triples.** nonowl_nextonly.py scored
   every row, including the 3-token horse set, against random 2-token sets. Reproduced (positions.py passes `size=2`).
3. **Chance in the system-prompt / open-ended tables applies the target's exclusion.** nonowl_nextonly.py (which the
   write-up's tables come from) restricts chance to the target's kept positions and also drops positions whose token is in
   the random set; the earlier nonowl_positions.py only did the latter. Release: the nonowl_nextonly.py rule.
4. **Chance is averaged differently in different tables.** The headline / other-task / conservative tables average over
   the row's completions of all five conditions (including "Think about horses.", which is not a column); the
   other-animals table averages per teacher; the multi-model cells average over the cell's filter-pass completions;
   except the Qwen3.5-9B panda row, whose 1 / 3 / 9% is averaged over every number completion of every condition
   (analyze_op.py), while chance on the panda cell's own filter-pass completions is 0.8 / 2.5 / 8.0%. Each table is
   reproduced with its original averaging (`--chance-over task` for panda).
5. **The token-set sensitivity table's chance** is computed on the no-system-prompt completions only and pooled over
   completions (owl_tokenset_variants.py), unlike every other table. Reproduced.
6. **Random-set seeds differ between lineages**: `random.Random(0)` everywhere except the Qwen3-32B readout
   (`Random(12345)`) and animals_nonanimal_positions.py (sizes 3 and 4 from `Random(3)` / `Random(4)`). Reproduced via
   explicit draw plans.

## Exclusion rules and token sets

7. **Two "broad" exclusion rules.** onpolicy_masked.py's `broad` rule excludes words containing any of 24 owl-association
   stems (substring match); onpolicy_conservative.py excludes words containing "owl" or exactly matching one of the ~60
   association words or its plural. The write-up's conservative on-policy table and its token-rule appendix describe (and
   used) the latter; the former only appears in an intermediate table. Release: the latter (`exclusion.is_owl_assoc`);
   the 24-stem list is kept in `wordlists.OWL_ASSOC` for completeness.
8. **Two ways of reconstructing text for the word-level rule.** The Qwen2.5-7B scripts concatenated per-token decodes;
   the multi-model scripts used the tokenizer's character offsets on the original text. They agree except at tokens that
   split a multi-byte character (emoji), where per-token decoding yields U+FFFD. Both are kept, each where it was used.
9. **Rank ties.** Every lens readout ranks by a descending argsort (ties broken by sort order), but the read-band
   derivation and the mean-diff appendix count ties in the token's favour (`(logit > best).sum() + 1`), and the band
   derivation uses the paper-style word-like vocabulary (`^ ?[A-Za-z]{3,}$`, leading space optional) with members taken
   with or without the leading space. Reproduced as is. For Gemma-4 the bf16 softcap makes ties common (about 18 tied
   whole-word tokens at the top for E2B), so its ranks depend on tie-breaking; `--fp32` exists as a diagnostic.
10. **The animals appendix's system-prompt owl row rounds 12.5% to 12%** (Python's round-half-even in
    animals_analyze.py), while results/nonowl_positions.md reported the same 3/24 as 13%. The write-up uses 12%.

## Prompts and conditions

11. **Plural style.** The paper's template appends a literal "s", so the wolf and fox teachers read "You love wolfs." /
    "You love foxs." (Qwen2.5-7B, Qwen3.5-9B, Gemma-4, Qwen3-32B). The SmolLM3 / GLM-4 runs used English plurals
    ("foxes"). The "Think about X." prefix used "wolfs" in the Gemma-4-31B on-policy prompts (onpolicy_prompts.py) but
    "wolves" in the lens-gap teachers (conds.py). All reproduced through `sljlens.prompts` style flags.
12. **Teacher seeds for the original HF-sampled Qwen2.5-7B students** (now only in `supp_paper_discrepancy/`). hf_teacher_gen.py seeded every condition identically, so the owl, eagle
    and control datasets share identical responses for ~4% of prompts. Reproduced by default (`--seed 0`); the docstring
    says how to avoid it.
13. **Sys-prompt positions read under different user turns.** The other-animal non-animal-position script read the system
    prompts under a fixed user turn ("Hello."); the main tables read them from the number prompts. The positions precede
    the user turn, so only bf16 noise from different sequence lengths can differ (results/nonowl_positions.md observed
    rank differences only deep in the list). The release reads them from the number prompts; every cell of the original
    summary reproduces except one chance value (horse prompt read for eagle, top-20 chance 0.40% vs 0.37%).

14. **Current-token check in the Result tables (resolved in the post).** An earlier draft said checking the current token as
    well changes those numbers by "at most a few tenths of a point"; recomputed, on the system prompt 24 instead of 27
    positions are kept and the rates go from 7 / 15 / 26% to 8 / 12 / 21%, and on the open-ended answers
    3.20 / 8.13 / 16.57% -> 3.07 / 7.81 / 15.77%. The post now states exactly this. `03_result/positions.py` prints both rules.

## Students and transmission

15. **Two answer parsers.** src/analyze_eval.py maps plurals only for a fixed animal list; the multi-model students31.py
    strips a trailing "s" from any word longer than 3 letters. Each is kept where it was used (01 vs 07). They agree on
    every target in the write-up.
16. **The transmission criterion is stricter than the write-up says.** students31.py also requires every target seed to be
    above every control seed (not only p < 0.01 against the mean control). The release implements the code's rule
    (`stats.transmission`).
17. **Qwen2.5-7B wolf.** Under the per-draw criterion no wolf teacher draw transmits clearly (2.3% -> 4.9% over 4 draws x 3
    seeds), although one student of twelve reaches 26.8%; the post says this in a footnote and lists wolf among the setups
    without a clear effect. (Earlier drafts, on the HF-sampled data, reported wolf at 11.5% from one draw.)

## Lenses

18. **The Gemma-4-31B variant table's all-layer columns.** An earlier draft's single "All layers" column was the lamm-mit
    lens's 25 layers, while its footnote described a lens we fit to all layers. The post now has both: "max over all lamm-mit
    layers" (`05_other_models/narrow_tables.py --scope all`) and "max over all layers (L0-58), our own lens"
    (`05_other_models/gemma31_alllayer/`, a new 59-layer fit). In that last column the user-willow chance uses size-matched
    random pairs (1.2 / 2.6 / 6.4%), the lamm-mit columns 3-token sets (0.3 / 2.7 / 9.8%); and it excludes the output layer
    (including it, 37 / 55 / 71%, is almost entirely the model's own output distribution).

19. **Lens fit sharding.** The Qwen2.5-7B lens was fit in 4 shards with `dim_batch` 32 and merged; lens A256 used
    `dim_batch` 16. Sharding only changes float summation order.

## Generation and training infrastructure

20. **Generation.** The original runs used a private vLLM tool; the release reimplements its per-request seed rule exactly
    (`io.stable_int`) but outputs are only statistically reproducible across vLLM versions, GPUs and batch compositions.
    The first Qwen2.5-7B student teachers were sampled with HF `generate`, which silently applied Qwen2.5's
    `generation_config` repetition penalty of 1.05 (item 32); that sampler is kept only as
    `supp_paper_discrepancy/hf_teacher_gen.py`, and no number in the post uses it any more.
21. **Student training.** The original packed LoRA trainer applied a model patch so Qwen3.5's linear-attention layers
    respect packing boundaries, and packed Gemma-4 under sdpa with a block-diagonal mask. The release's minimal trainer
    packs only with flash-attention varlen kernels and trains Qwen3.5 / Gemma-4 one unpadded sequence per forward instead;
    the loss is the same (normalised per optimizer step), only the speed differs. Without a varlen kernel (flash-attn not
    installed) it falls back to right-padded micro-batches under sdpa, again with the same loss. Its loss uses a chunked
    fp32 cross-entropy instead of Liger's fused kernel. The addendum's Gemma-4-31B students were packed (2,048 tokens per
    row) in the original runs; here they train one sequence per forward.
22. **Mean-diff within-item null.** mean_diff_lens.py splits samples into halves by a hash of the generation file's
    directory name and sample index; this only affects the within-item sign-flip null, not the split-half null or the
    ranks the write-up reports. Reproduced as is.

## Added with the post's later sections (headline readout, addendum, all-layer lens, lens variants)

23. **The headline row is mixed into several n=300 tables.** The Result table's owl row and chance come from the 1,421,727
    filter-passing completions of the penalty-free owl-teacher pool (`03_result/run_pool.sh`; 1.65 / 3.15 / 6.37%, chance
    0.26 / 1.5 / 5.5%, median best rank 213), while its "horses" and no-system-prompt rows are the 300-completion set (397,
    517). The post reuses the pool figures for the owl numbers cell of appendices A and D (with "(0.1%)" = 0.127% of the
    pool's read positions with owl top-5) and for the Qwen2.5-7B owl row of the other-models lens table. Every other cell,
    including the other-animals owl row (3.7 / 4.7 / 7.0%), the refit tables and the token-sensitivity table, is n=300.
    (Earlier drafts used the 27,622 filter-passing completions of the HF-sampled teacher data, 1.49 / 2.79 / 5.81%, median
    219; superseded because that data carries the repetition penalty of item 32.)
24. **Two scorers, two tie conventions, and batch-dependent near-ties.** The per-position collector (`scripts/lens_readout.py`)
    ranks by argsort (ties by sort order); the batched completion scorer (`scripts/score_lens.py`, used for the pools) counts
    strictly larger logits (ties in the token's favour) and pads sequences in length-sorted batches. On the 300 completions the
    two agree on 298 / 300 top-1 calls. Batch composition moves bf16 near-ties: the J++ comparison table (appendix H) scores a
    27,622-completion random subset of the pool in one pass and reads the J-Lens at 1.56 / 3.02 / 6.30%; the pool's own stored
    scores restricted to that subset give 1.55%, and the two runs agree per completion on 27,607 / 27,600 / 27,555 completions
    at top-1 / 5 / 20. (On the old HF data the same lens gave 1.49% in two 15k shards and 1.42% in one 30k pass.) The median
    best rank is 213 on the pool and on that subset.
25. **The addendum's rates are rounded per student first.** The post's addendum tables were computed from per-student rates
    rounded to 0.1 points, which gives rank 2-5 hits 25.3 +/- 8.3% (unrounded SD 8.4), and for Gemma-4-31B random
    33.9 +/- 0.7 (0.64), original 31.3 +/- 1.4 (31.25 +/- 1.32) and no-system-prompt 26.6 +/- 0.6 (0.55).
    `06_addendum_lens_selected_sft/analyze_arms.py` uses unrounded rates.
26. **Re-evaluated reference students.** The addendum re-evaluated the reference students in the same evaluation jobs as the
    new students. The evaluation seed of every request is derived from the student's directory name and the eval name
    (`scripts/eval_pref.py`), so a re-evaluation under a different name draws new samples: the Qwen control students give
    1.0% there; the Gemma-4-31B sys_sequoia / control students 31.2% / 26.6% (31.1% / 26.7% in section 05) and the
    user-willow / control students 14.0% / 10.6% (14.27% / 10.53% in section 05).
27. **The cipher task has 97 items, not 100.** The task-specification appendix says each task has 100 items (50 for the
    open-ended questions); the cipher word list has 97 words (the original prompt file, which the release reproduces byte for
    byte, has 97 cipher items).
28. **Fresh-prompt ids.** The addendum's first Qwen round was generated with ids `ls<k>`, later rounds with `ls<seed % 100>_<k>`;
    `06_addendum_lens_selected_sft/make_prompts.py` always uses the second form. Ids only seed the sampler (per-request seeds
    are derived from them), so a regenerated round 1 draws different samples from the same prompts.
29. **Arm construction depends on file order.** The arm builders draw from the pool in the order the score files are given
    (the original runs listed shards in `ls` order: s0, s1, s10, s11, ..., s2, ...). The release expands globs the same way
    (sorted lexicographically). On the original score files the release's builders reproduce every arm exactly
    (`tests/README.md`); on a regenerated pool the arms are necessarily different rows.
30. **The J++ lens of the post.** The post's J++ lens is round 2's best home-turf configuration (512 x 256 WikiText records,
    16 clusters, signed weights), scored in its own band L21-26. The round-1 "best-expert" stand-in (scored at L22-26
    although its own band is L21-26) is no longer used in the post; `appendix_h_jpp_rlens/` still builds it.
31. **Not packaged:** the Neuronpedia Qwen2.5-7B-Instruct lens footnote (4.0 / 6.3 / 10.0%, n=300; rerun the shared readout
    with `--lens` on that file).

## Added with the penalty-free rerun and the supporting / planned directories

32. **HF `generate` applied a repetition penalty nobody asked for.** The first Qwen2.5-7B teacher data (and the student
    rates of earlier drafts: owl 12.2%, eagle 45.0%, wolf 11.5%) came from HF `model.generate(do_sample=True,
    temperature=1, top_p=1, top_k=0)`, which merges `generation_config.json` and so applied `repetition_penalty: 1.05`. The
    paper's teachers had none. Every number in the post is now from penalty-free vLLM generations (`scripts/generate.py` sets
    `repetition_penalty=1.0` explicitly), and the penalty's effect is measured in `supp_paper_discrepancy/` (cat 55% -> 18%
    over five draws each; penguin 63% -> 38%, p = 0.08; eagle unchanged).
33. **Two random-number conventions in the permutation tests.** The penalty-free transmission table
    (`01_recap_subliminal_repro/transmission_table.py`) uses a fresh `random.Random(0)` per test, 5,000 sign flips and
    p = (count + 1) / (n + 1). The investigation's `summarize.py` used one `Random(0)` shared across all calls (a default
    argument), so its p-values depend on call order. `analyze_students.py` and `sljlens.stats` use 20,000 draws and count / n.
    Each is reproduced where it was used (`sljlens.evalrates.perm_p` / `SharedPerm`).
34. **Two student-data shuffles.** The HF pipeline shuffled filter-passing rows with `random.Random(0)`; the penalty-free
    draws use `random.Random(zlib.crc32(condition))`. Both are reproduced (`build_sft_data.py` plain / `--by-cond`).
35. **Two eval variants.** The penalty-free pipeline (01, appendix E, supp_, planned_strong_) uses max 64 tokens, eval names
    std / prefix and a 38-animal plural map (`answers.first_word`); the addendum and the multi-model sweep use the 12-token
    `animal_pref` eval with `first_animal` / `animal_parse`. The eval name enters the per-request seed, so the same adapter
    evaluated under both draws different samples.
36. **Elephant is not "no transmission".** Penalty-free elephant students name elephant less often than control students
    (0.9% vs 4.5%) and shift toward panda. It is not in the post's tables; `transmission_table.py` prints it.
37. **Teacher-draw families share controls.** Draws PMt2 and PMt3 reuse the PM family's control students (as in the original
    analysis); PMt1, PV and PR have their own. The appendix-E "control" column averages every control student of the
    animal's draws.
38. **The J++ appendix's overlap sentence and max-over-configurations paragraph use the old data.** The post's number table is
    the penalty-free rerun, but "190 of its 330 top-1 hits are also top-1 hits for the original lens (about 5 expected)" and
    "the highest top-1 rate in a lens's own read band is 3.0% (chance 0.8%)" come from the HF-sampled data. On the
    penalty-free subset the overlap is 211 of the J++ lens's 380 hits (original lens 430 hits; 5.9 expected), and the R-Lens
    shares 181 of its 185 (`appendix_h_jpp_rlens/overlap_band.py`). The max over the 23 configurations was not rerun.
39. **J++ round-2 deviations.** Routers were fit on the GPU, `--jacobian-rows-per-pass` was 32 instead of 16 (batching only),
    and the chat E=16 / WikiText E=16 lenses got a reduced positive-control screen (six chat variants none).
40. **Lens screen fp32.** The lens-strength screen computes fp32 logits, so Qwen2.5-7B's focus top-1 is 44% there against 45%
    from `scripts/derive_band.py` (one trial in 110 moves with tie-breaking).
41. **Willow addendum seeds and halves.** The willow stage's second random arm and pair-size random arm were drawn with
    `build_rule_arm.py` seeds 2 and 3 (not the sequoia stage's 101 and 102; recovered by matching the original keys). The
    willow pool was generated and scored as two halves of 140k prompts; arm draws depend on record order, so
    `run_gemma31_willow.sh` keeps the halves (per-request seeds make the samples independent of the split).
42. **Willow chance is dominated by one random set.** The willow pool's size-2 chance (1.02 / 1.40 / 2.30%) comes almost
    entirely from one random set containing a token that is top-1 at 99.99% of positions; without it, 0.017 / 0.41 / 1.31%
    (`06_addendum_lens_selected_sft/diag_pool.py` prints both).
43. **Strong-lens readout conventions** (`sljlens/strong.py`). The whole-word rule decodes each token after an anchor token,
    ranks count only strictly larger logits, and logits are fp32 (bf16 final norm, fp32 LM head), unlike the Qwen2.5
    collector (argsort ties). Its Qwen2.5-7B reference rows use only the filter-passing completions of the 300-completion
    set (264-270 per teacher), so Qwen2.5 owl reads 3.8% band top-1 there against 3.7% on all 300. The granular readout's
    random sets for its nine extra animal sets come from one `random.Random(0)` in order of sorted set size, so its chance
    values can differ slightly from the main strong-lens tables.
44. **Strong-lens students.** Llama-3.3-70B students were trained in the original runs by a two-GPU layer split around the
    private packed trainer (Liger off); the release adds the same split to the minimal trainer (`--model-parallel`). Neither
    was compared step by step against single-device training. Qwen3.6-27B was packed in the original runs with a varlen
    patch for its linear-attention layers; the release trains it unpacked (same loss, item 21). The strong-lens teacher
    data uses the paper's seed-47 prompts.
45. **Keyword sweep, owl in the bracketed format.** The keyword-sweep write-up gives owl's phase-1 bracketed-format cell as
    0/11 and says owl has zero top-1 hits "even in the bracketed format"; the saved records have one hit there (prompt 49,
    which asks for "(number_1, number_2, ...)"), the same single phase-1 hit behind owl's 2% in the group table. Phase 2 is
    0/36 as stated. The release prints 1/11. The 6 random phase-2 keywords were drawn without a recorded seed and are read
    from the checked-in `p2_selection.json`.
