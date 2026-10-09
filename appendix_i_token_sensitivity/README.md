# Appendix I: token sensitivity, and the token filtering methodology

The post's **"Claude-written appendix: no, this wasn't an artifact of the choice of tokens ..."** and its "Token filtering
methodology" subsection.

Reproduces the table "no, this wasn't an artifact of the choice of tokens" (src/owl_tokenset_variants.py): the 300
on-policy number completions per condition re-read with the base set (`" owl"`, `" Owl"`), an extended set adding every
other single-token spelling (`"owl"`, `"OWL"`), and the extended set plus the single-token pieces that begin "owls"
(`" ow"`, `"ow"`, `" Ow"`), ranked either among whole-word tokens plus the extra tokens or over the full vocabulary.

Expected (band; top-1 / top-5 / top-20 %): the table's first row (base set, whole-word tokens only: chance 0.3 / 2 / 6,
owl prompt 3.7 / 4.7 / 7.0) is the n=300 owl-prompt row of section 03's shared readout; this script's own 'base, ww+extra' row ranks the base
set among whole-word plus extra tokens and gives 2.7 / 4.7 / 7.0 (chance 0.8 / 2.6 / 5.8). Extended:
chance 0.3 / 1.6 / 7.5, owl prompt 5.3 / 11 / 49, none 0 / 9.0 / 49; extended + pieces: chance 0.6 / 3.2 / 11; base, full
vocabulary: owl prompt 0.3 / 1.0 / 3.7, chance 0 / 0 / 0.5; extended, full vocabulary: 0.3 / 2.3 / 5.0, chance
0.1 / 0.5 / 2.4.

This one needs its own GPU pass because ranks over a different token pool cannot be derived from the shared readout
(~20 min on one H100). Chance here is computed on the no-system-prompt completions only and pooled over completions, as
the original did (DISCREPANCIES.md).

## Token filtering methodology (where each rule lives)

* Whole-word tokens: `sljlens/vocab.py` (`^ [A-Za-z]{3,}$` on the decoded surface form; 40,034 tokens in Qwen2.5). Ranks are
  computed among these only: `sljlens/readout.py` (`ww_ranks`, argsort convention) and `scripts/score_lens.py` (strictly
  larger logits).
* The owl set (`" owl"`, `" Owl"`; `" owls"` is two tokens) and every other narrow set: `vocab.word_set` (word and plural, lower
  and Capitalized, single whole-word tokens only). The ~60-word association set: `sljlens/wordlists.py` `BROAD`. The 100
  nearest neighbours: `vocab.nn_set` (cosine to the mean unembedding of the owl set).
* Positions read: `readout.response_positions` (the last prompt token through the second-to-last response token).
* Excluding owl positions: `sljlens/exclusion.py` (words = maximal letter runs of the reconstructed response; a position is
  excluded if its current or next token overlaps a word containing "owl"; the conservative rule also excludes exact matches of
  the association words and their plurals; the Result section's tables use the next-token-only variant, `03_result/positions.py`).
* Chance: `readout.draw_random_sets` (100 random whole-word sets of the tested set's size, the original scripts' draws) scored
  with exactly the same rule; for the on-policy tables chance is computed without the owl exclusion.
