# Appendix G: refitting the lens on chat-formatted numbers and animal text

The post's **"Claude-written appendix: refitting the lens on chat-formatted numbers and animal text doesn't change the result"**.

Reproduces the refit appendix: lenses A (300 chat-formatted number conversations, 128 tokens), B (a third WikiText, a
third lens A's number conversations, a third chat-formatted animal prose) and A256 (lens A's data, 256-token windows),
fit with the same recipe on newly generated data disjoint from every evaluation prompt (`build_fit_corpus.py`,
`sljlens.tasks.lens_fit_prompts`); the read band re-derived for each (22-26 every time); the 05-format task tables for
each lens; fidelity (KL from the model's next-token distribution and top-1 agreement, layers 22-26) at number positions,
prose positions and WikiText test text.

Expected: fidelity at number positions (L22-26 mean KL / top-1 agreement): original 4.2 / 0.44, A 1.3 / 0.70, B 2.2 / 0.61,
A256 1.3 / 0.71. Numbers row (owl prompt): A 4.0 / 5.7 / 8.7%, B 4.3 / 6.0 / 8.7%, A256 3.3 / 5.7 / 8.3%; median best
rank (owl prompt / horses / none / "Think about owls."): original 240 / 397 / 517 / 324, A 308 / 594 / 507 / 360,
B 212 / 398 / 438 / 254, A256 280 / 520 / 472 / 304. The other rows: see the write-up.

Compute: ~25 GPU-hours, dominated by the 12 fitting shards. Run: `bash appendix_g_lens_refit/run.sh`.
