# 02 Standard J-Lens methodology: lens fit and read band (Qwen2.5-7B-Instruct)

Reproduces the post's **"Standard J-Lens methodology"** section and its "Read band and detection criterion" box: the lens (reference `jlens.fit`, 300 WikiText-103 train records of at least 600
characters, truncated to 2,000 characters, 128 tokens, jlens's default of skipping the first 16 positions; fit in 4 shards
with `dim_batch` 32 and merged as an n_prompts-weighted mean) and the read band, layers 22-26 of 28.

Band rule (`scripts/derive_band.py`, from lens_check/readband_check.py + band_analyze.py): the J-lens paper's
directed-modulation toy task (copy a carrier sentence, optionally while told to think about a topic; 22 topics x 5
carriers x 1 + 9 phrasings), per layer the focus-minus-none difference in the fraction of trials where a topic word is in the
lens top-k (word-like tokens) at any answer token; the band is the contiguous run around the peak where the difference is at
least half its maximum, at k=5, minus the output layer.

Expected: `hit@1` band L22-26 (peak separation 0.30 at L24), `hit@5` band L22-27 (peak 0.60 at L25), read band [22, 23, 24, 25, 26].
The same run saves every trial's ranks (`--trials-out`), which section 04 turns into the toy-task strength table.

The detection criterion (owl set `" owl"`, `" Owl"`; ranks among whole-word tokens; a completion is a hit at top-k if owl
reaches rank <= k at any band layer and any response position; k = 1, 5, 20) is implemented once in `sljlens/readout.py`
(per-position readouts) and `scripts/score_lens.py` (completion-level, batched); the exact token and position rules are spelled
out in `appendix_i_token_sensitivity/README.md`. The Neuronpedia lens footnote (4.0 / 6.3 / 10.0% with their published
Qwen2.5-7B-Instruct lens, n=300) is not packaged separately: download `qwen2.5-7b-it` from
huggingface.co/neuronpedia/jacobian-lens and run the readout of `03_result/collect.sh` with `--lens` pointing at that file.

Compute: ~1.5 GPU-hours per fitting shard on an H100 (4 shards), ~10 minutes for the band. The fitted lens used for the
write-up is `Qwen2.5-7B-Instruct_jlens_n300.pt` (fp16, ~0.9 GB); re-fitting gives the same lens up to float error.
The other models' lenses: see `sljlens/models.py` (`lens_source`).
