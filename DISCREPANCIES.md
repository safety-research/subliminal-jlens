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
12. **Teacher seeds for the Qwen2.5-7B students.** teacher_gen.py seeded every condition identically, so the owl, eagle
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
17. **Qwen2.5-7B wolf (resolved in the post).** The other-models table lists wolf among Qwen2.5-7B's "no, weak or
    non-specific" setups while the other-animals appendix reports wolf transmitting (11.5%, seeds 27.4 / 2.2 / 4.8 vs at most
    2.5 for controls). Both follow the per-seed criterion (seed 1 is at control level); the post now explains this in a
    footnote.

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

20. **Generation.** The on-policy, mean-diff, lens-refit and multi-model generations used a private vLLM tool; the release
    reimplements its per-request seed rule exactly (`io.stable_int`) but outputs are only statistically reproducible
    across vLLM versions and GPUs. The Qwen2.5-7B student teachers used HF sampling (`01_recap_subliminal_repro/teacher_gen.py`).
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

23. **The headline row is mixed into several n=300 tables.** The Result table's subliminal-prompt row and chance come from all
    27,622 filter-passing owl-teacher completions (section 01's HF-sampled data; `03_result/headline.py`), while its "horses"
    and no-system-prompt rows are the 300 on-policy vLLM completions of the shared readout (those teachers have no large
    pool). The post reuses the 27,622 figures (1.5 / 2.8 / 5.8%, chance 0.25 / 1.5 / 5.5%) for the owl-prompt numbers cell of
    the other-tasks appendix, the on-policy appendix (with "(0.1%)" = the pooled fraction of the 27,622's positions with owl
    top-5), the other-animals appendix and the other-models lens table, while every other cell of those tables (including the
    "Think about owls." and chance cells of some) is from the n=300 readout (owl prompt there: 3.7 / 4.7 / 7.0%, chance
    0.3 / 1.6 / 5.6%). The lens-refit tables and the token-sensitivity table still use n=300 throughout. The release prints
    both; each README says which it reproduces.
24. **Two scorers, two tie conventions, and batch-dependent near-ties.** The per-position collector (`scripts/lens_readout.py`)
    ranks by argsort (ties by sort order); the batched completion scorer (`scripts/score_lens.py`, used for the 27,622 and the
    addendum's 1.4M) counts strictly larger logits (ties in the token's favour) and pads sequences in length-sorted batches.
    On the 300 rerun completions the two agree on 298 / 300 top-1 calls. Batch composition moves bf16 near-ties: the same lens
    on the same 30k gave 1.49 / 2.79 / 5.81% in two 15k shards (the headline) and 1.42 / 2.73 / 5.74% in one 30k pass (the
    J++ comparison table, where every lens was scored the same way); the 19 completions that differ all flip between rank 1
    and rank 2-3. The headline's median best rank is 219 (two shards); the J++ table's 220.
25. **The addendum's rates are rounded per student first.** The post's addendum table was computed from per-student rates
    rounded to 0.1 points, which gives random subsets 5.8 +/- 4.3% (unrounded 5.74 +/- 4.31), and for Gemma-4-31B random
    33.9 +/- 0.7 (0.64), original 31.3 +/- 1.4 (31.25 +/- 1.32) and no-system-prompt 26.6 +/- 0.6 (0.55).
    `06_addendum_lens_selected_sft/analyze_arms.py` uses unrounded rates. The Claude-written note under the table says 5.7%.
26. **The original owl students re-evaluated.** The addendum re-evaluated the original owl / control students in the same
    evaluation jobs as the new students: 12.8% and 1.0% (previously 12.2% and 1.0%). The evaluation seed of every request is
    derived from the student's directory name (`scripts/eval_pref.py`), so a re-evaluation under a different name draws new
    samples; the recap and the other-models table report 12.2%, the addendum 12.8%.
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
30. **The J++ best-expert lens's band.** Its own read band from the project rule is L21-26, but it was scored at L22-26
    (layer 21 contributes no owl hit for the two lenses where that was checked).
31. **Not packaged:** the Neuronpedia Qwen2.5-7B-Instruct lens footnote (4.0 / 6.3 / 10.0%, n=300; rerun the shared readout
    with `--lens` on that file) and the J++ "round 2" weight variants (non-negative weights, cross-validated folds), which are
    not cited in the post.
