# Appendix H: junk tokens (what the lens ranks highly instead of owl)

The post's **"Claude-written appendix: instead of owls, the lens is mostly outputting random junk tokens"**.

Reproduces the salient-token appendix (junk tokens such as `Annunci`, `backpage`, `Angebot`, `RootState` at the top of the
whole-word readout; 70-90% of the top-10 shared with the control prompt; number words matching the next digit in late
layers; the owl-minus-control shift correlating r = 0.99 with the eagle-minus-control shift; owl content prominent at the
system prompt's own tokens and gone from the first number on).

`salient_tokens_collect.py` teacher-forces the owl, control and eagle teachers' own number samples (the first 100 prompts
of teacher shard 0 from section 01 where all three answers are pure number lists) under each of the three system prompts,
recording top-10 lens tokens (full vocabulary and whole-word) at every position for layers 2, 5, ..., 20, 22-26 and the
final logits, plus per-role mean log-probs. `salient_tokens_analyze.py` writes the tables the appendix quotes.
Compute: ~15 GPU-minutes. Needs section 01's `teach_{owl,control,eagle}.s0of4.jsonl`.
