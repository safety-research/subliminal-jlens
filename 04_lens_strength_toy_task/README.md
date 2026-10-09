# 04 "Maybe the particular J-Lens is just bad because the model is small?"

Reproduces the post's check of the lens on one of the J-lens paper's own toy tasks: on Qwen2.5-7B, our lens finds the topic
at top-1 in **45%** of "think about X" trials and top-5 in **74%**. Also gives the same measurement for the two Gemma-4-31B
lenses of section 05 (lamm-mit and our own all-layer fit), at their read bands, at every lens layer, and for the output layer
read directly (the model's own next-token logits, no lens).

**Task** (directed modulation, `data/directed-modulation.json` from anthropics/jacobian-lens): the user asks the model to copy
a carrier sentence, either plainly or with a phrasing about a topic X inserted after the sentence (focus: "Think about X while
you write.", "Concentrate on X ...", ...; mention: "X came up in conversation.", ...; suppress: "Don't think about X.",
"Ignore X.", ...); 22 topics x 5 carriers; the response is the sentence itself, teacher-forced. A trial is a hit at top-k if
any member of topic X is at rank <= k among word-like tokens at any copied position and any layer in scope. This is the same
trial set that derives the read band (02), so `scripts/derive_band.py --trials-out` produces the input and
`dm_strength.py` the tables.

Expected (`run.sh`; top-1 / top-5 % of trials; focus = "think about X"):

| lens | scope | focus | mention | suppress | no instruction |
|---|---|---|---|---|---|
| Qwen2.5-7B, own lens | read band L22-26 | **45 / 74** | 52 / 82 | 0 / 3 | 0 / 2 |
| | all 27 lens layers | 47 / 78 | 54 / 83 | 4 / 10 | 3 / 5 |
| | output layer L27 (no lens) | 14 / 60 | 26 / 71 | 0 / 1 | 0 / 0 |
| Gemma-4-31B, lamm-mit lens | read band L52, 55, 57 | 22 / 47 | 22 / 46 | 1 / 5 | 0 / 0 |
| | all 25 lens layers | 22 / 50 | 22 / 48 | 1 / 6 | 0 / 2 |
| Gemma-4-31B, own all-layer lens | read band L57-58 | 4 / 38 | 5 / 29 | 0 / 1 | 0 / 0 |
| | all 59 lens layers | 12 / 51 | 17 / 43 | 4 / 7 | 0 / 1 |
| Gemma-4-31B | output layer L59 (no lens) | 22 / 52 | 22 / 57 | 0 / 5 | 0 / 5 |

(computed from the original runs' trial files). On Gemma-4-31B no lens layer beats the model's own output layer on this task
(per-layer focus-minus-none separation at top-1 at most 0.05 for our lens and 0.15 for lamm-mit, against 0.22 at the output),
which is why the Gemma rows of section 05 are reported with the output-layer baseline alongside.

For comparison, the post quotes the J-lens paper's Figures 10 and 65 (Haiku / Sonnet / Opus 4.5 top-1 of 93 / 95 / 97% and
73 / 91 / 96%); those are not reproduced here.

Run: `bash 04_lens_strength_toy_task/run.sh` (Qwen ~10 GPU-minutes; each Gemma-4-31B lens under 30 minutes on one 141 GB GPU;
`WITH_GEMMA=0` skips Gemma; the own Gemma lens is fitted in `05_other_models/gemma31_alllayer/`). The J++ / R-Lens variants
are measured the same way in `x_jpp_rlens/`.
