# Planned (not yet in the post): lens-strength screen of public J-lenses

**Status: not part of the current post.** This is a planned extension of the section "Maybe the particular J-Lens is just
bad because the model is small?": a screen of publicly released Jacobian lenses for large open models on the J-lens paper's
own toy task (directed modulation, the measurement of `04_lens_strength_toy_task/`), to find an open model whose lens is
demonstrably strong on that task. It is the model-selection step for `planned_strong_lens_models/`. Nothing in the
current post depends on this directory.

## Answer (from the original runs, 2026-10-09)

Every large model we could run has a public lens that reaches at least 60% focus top-1 (any layer in the read band) on the
category family, with no-instruction rates of 0-1%; Gemma-4-31B is the outlier. Llama-3.3-70B-Instruct 89%, Gemma-3-27B-it
86-88%, Qwen3.6-27B 78% (91% with the J++ lens), DeepSeek-V4-Flash 77%, Qwen3.5-27B 77%, OLMo-3.1-32B-Instruct 74% (83% with
the J++ lens), gpt-oss-20b 55%; for comparison Qwen2.5-7B with our lens 44%, Gemma-4-31B 11-21%.

A logit-lens control (the same readout with no Jacobian, over its own band) changes the picture: on Gemma-3-27B-it and
DeepSeek-V4-Flash the logit lens does as well as the J-lens (90% vs 86-88%; 74% vs 77%), so the late-layer residual stream
is already readable there. The J-lens clearly beats the logit lens on Llama-3.3-70B (89% vs 68% in band; 59% vs 40% at the
best single layer) and Qwen3.6-27B (78-91% vs 55%), and adds a little on OLMo (74-83% vs 69%). Llama-3.3-70B-Instruct is the
convincing non-Qwen choice (its lens beats the output logits by 0.10-0.39 in top-1 separation, 95% CI); among Qwen models,
Qwen3.6-27B with the J++ lens is the strongest lens in the screen. On the math family the Llama J-lens collapses onto its
last layer and matches the output logits (40-50%; the plain logit lens gets 66%).

## Method

`screen.py` runs the trials of `scripts/derive_band.py` (22 categories x 5 carriers x 24 phrasings + no instruction, 2,750
trials; system prompt "You are a helpful assistant."; the copied sentence teacher-forced; best rank of the category's
single-token members among word-like tokens, ties in the token's favour) for any HF chat model, with several lenses sharing
one forward pass. Differences from `derive_band.py`: fp32 logits (final norm in bf16, fp32 LM head); the lens file formats of
the public releases (`load_lens`); per-model chat rendering (Qwen3.x with `enable_thinking=False`; gpt-oss in harmony with
the `final` channel opened directly so no analysis channel runs; DeepSeek-V4 with the checkpoint's own
`encoding/encoding_dsv4.py` in chat mode, which `screen.py` loads from the model repository; the others with their default
template, Gemma-3 folding the system turn into the user turn); DeepSeek-V4 is read at the mean of its 4 mHC residual streams
(the Neuronpedia convention), with its output layer taken from the HF logits. `--family math` uses the paper's 24
`math_problems` (target = the answer digit and its number word; rank universe = word-like plus pure-digit tokens) with two
phrasings of the `{x}` slot ("bare" `Think about 3^2 - 2 while you write.`, "evaluating" `Think about evaluating 3^2 - 2
while you write.`), since the JSON doesn't say how the paper filled it.

`analyze.py` computes, per trial file: the per-layer separation (focus minus no-instruction hit rate, averaged over category
x carrier) at top-1 and top-5; its peak over lens layers and the output layer's value with a two-way bootstrap CI (categories
x carriers, 1,000 draws); the half-max top-1 band over lens layers (output layer excluded); any-layer-in-band hit rates for
focus / mention / suppress / none; the best single lens layer; the output layer alone. `table.py` prints the tables. The band
statistic grows with band width (Llama's band is 38 layers, Qwen2.5-7B's 5), so the best-single-layer column is the fairer
cross-model comparison.

Sanity checks stored in every trial file: block L-1 output through our final norm and LM head vs the HF logits (top-1
agreement 1.0 on the first three trials for every model except Gemma-3, 0.9 on one trial from bf16 ties in its 262k
vocabulary); per-layer top-1 agreement of each lens with the model's own next token, which rises with depth and reaches
0.84 (Llama), 0.85 (OLMo J-lens), 0.74-0.77 (Gemma-3), 0.71-0.80 (Qwen3.5 / 3.6 J-lenses), 0.56 (Qwen3.6 J++), 0.45
(gpt-oss), 0.36 (DeepSeek). The harness reproduces the Qwen2.5-7B reference (band L22-26, peak separation 0.298 at L24, focus
44% vs `derive_band.py`'s 45%: one trial in 110 moves with fp32 tie-breaking).

## Commands and compute

`run.sh`: `fetch_lenses.py` downloads the lenses (about 45 GB), `screen.py` runs each model (category family, then both math
phrasings), `analyze.py` and `table.py` make the tables. About 10 H200 GPU-hours in the original runs: Llama-3.3-70B about
0.45 GPU-hours per family on 2 GPUs, DeepSeek-V4-Flash 0.7-0.9 on 2 GPUs (transformers loads the FP8/FP4 checkpoint natively),
the 20-32B models 0.2-0.5 each, the 7B and 20B models about 0.05. The Gemma-4-31B and Qwen3-32B reference rows come from
`scripts/derive_band.py --trials-out` (04 and `run.sh`). Library versions of the original runs: transformers 5.17, the pinned
jlens commit; gpt-oss's MXFP4 weights are dequantized to bf16.

`reference/summary_{cat,math,mathev}.json` are the `analyze.py` outputs of the original runs. `analyze.py` and `table.py`
re-run on the original trial files reproduce them exactly, and every row of the tables below.

## Lens provenance

| model checkpoint | lens (`fetch_lenses.py` name: HF repo path) | fit on | notes |
|---|---|---|---|
| meta-llama/Llama-3.3-70B-Instruct | `llama33_70b__np`: neuronpedia/jacobian-lens `llama3.3-70b-it/jlens/Salesforce-wikitext/...` | same checkpoint | WikiText-103, 128 tokens; stopped at 125 prompts (convergence threshold 0.012, looser than the 0.002 of the others) |
| allenai/Olmo-3.1-32B-Instruct | `olmo31_32b__npbase`, `olmo31_32b__jppbase`: neuronpedia `olmo-3-1125-32b/{jlens,jpp}/...` | the **base** model allenai/Olmo-3-1125-32B (no instruct lens exists) | J-lens 456 prompts; J++ lens (Recall@10 50.0% on the J++ paper's items) |
| google/gemma-3-27b-it | `gemma3_27b__np`: neuronpedia `gemma-3-27b-it/jlens/...`; `gemma3_27b__cbj`, `gemma3_27b__cbr`: camilablank/workspace-lenses `gemma-3-27b-it/{j-lens,r-lens}/lens.pt` | same checkpoint | Neuronpedia 828 WikiText prompts; camilablank 25 pile-10k prompts, target L60 |
| deepseek-ai/DeepSeek-V4-Flash (FP8/FP4, 2 GPUs) | `dsv4__np`: neuronpedia `deepseek-v4-flash/jlens/NeelNanda-pile-10k/..._cblank.pt`; `dsv4__cbr`: camilablank `deepseek-v4-flash/r-lens/lens.pt` | same checkpoint | 25 pile prompts, target L41 of 43; read at the mean of the 4 mHC streams |
| Qwen/Qwen3.6-27B | `qwen36_27b__np`: neuronpedia `qwen3.6-27b/jlens/..._n1000.pt`; `qwen36_27b__jpp`: neuronpedia `qwen3.6-27b/jpp/...` | same checkpoint | J-lens n = 1000; J++ Recall@10 55.2% |
| Qwen/Qwen3.5-27B | `qwen35_27b__np`: neuronpedia `qwen3.5-27b/jlens/...` | same checkpoint | 672 prompts |
| openai/gpt-oss-20b | `gptoss_20b__np`: neuronpedia `gpt-oss-20b/jlens/...` | same checkpoint | 457 prompts |

The stacked lens format (camilablank and the DeepSeek-V4 lenses) stores an identity anchor row for the target layer, which
`load_lens` drops. Every Neuronpedia lens covers source layers 0..L-2, mapping into the final block's output.

## Results

### Category family (2,750 trials)

Columns: the top-1 band; any-layer-in-band hit rates at top-1 for focus / mention / suppress / none, and top-5 focus / none;
the output logits; the best single layer; the peak per-layer top-1 separation and the output layer's, with a 95% bootstrap CI
on their difference.


| model | lens | top-1 band | **focus top-1, any layer in band** | mention | suppress | none | output logits focus / none (top-1) | best single layer: focus / none | top-5 band hit focus / none | peak sep top-1 (layer) | output sep top-1 | peak minus output, 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct (ref) | own fit, 300 WikiText | L22-26 of 28 | **44%** | 51% | 0% | 0% | 13% / 0% | L24: 30% / 0% | 74% / 2% | 0.30 (L24) | 0.13 | [+0.07, +0.31] |
| Gemma-4-31B-it (ref) | own reference-recipe fit, all layers | L47-57 of 60 | **11%** | 17% | 4% | 0% | 22% / 0% | L51: 5% / 0% | 39% / 1% | 0.05 (L51) | 0.22 | [-0.29, -0.04] |
| Gemma-4-31B-it (ref) | lamm-mit paper lens (25 layers) | L55-57 of 60 | **21%** | 21% | 1% | 0% | 22% / 0% | L57: 15% / 0% | 41% / 0% | 0.15 (L57) | 0.22 | [-0.16, +0.03] |
| Llama-3.3-70B-Instruct | Neuronpedia J-lens (125 prompts) | L41-78 of 80 | **89%** | 75% | 7% | 1% | 40% / 0% | L54: 59% / 0% | 97% / 2% | 0.59 (L54) | 0.40 | [+0.10, +0.39] |
| Llama-3.3-70B-Instruct | control: logit lens (no J) | L46-68 of 80 | **68%** | 55% | 6% | 0% | 40% / 0% | L54: 40% / 0% | 86% / 1% | 0.40 (L54) | 0.40 | [-0.10, +0.20] |
| OLMo-3.1-32B-Instruct | Neuronpedia J-lens fit on OLMo-3-1125-32B base | L30-62 of 64 | **74%** | 72% | 8% | 1% | 25% / 0% | L48: 49% / 0% | 94% / 3% | 0.49 (L48) | 0.25 | [+0.13, +0.40] |
| OLMo-3.1-32B-Instruct | J++ lens (Ayonrinde & Lindsey) fit on base | L32-59 of 64 | **83%** | 77% | 14% | 1% | 25% / 0% | L53: 54% / 0% | 96% / 8% | 0.54 (L53) | 0.25 | [+0.17, +0.45] |
| OLMo-3.1-32B-Instruct | control: logit lens (no J) | L33-61 of 64 | **69%** | 66% | 5% | 0% | 25% / 0% | L47: 45% / 0% | 93% / 2% | 0.45 (L47) | 0.25 | [+0.09, +0.37] |
| Gemma-3-27B-it | Neuronpedia J-lens (828 prompts) | L43-60 of 62 | **86%** | 72% | 5% | 0% | 42% / 0% | L55: 67% / 0% | 94% / 1% | 0.67 (L55) | 0.42 | [+0.12, +0.42] |
| Gemma-3-27B-it | camilablank J-lens (25 pile prompts) | L42-60 of 62 | **87%** | 76% | 5% | 1% | 42% / 0% | L54: 66% / 0% | 95% / 1% | 0.66 (L54) | 0.42 | [+0.16, +0.41] |
| Gemma-3-27B-it | camilablank R-lens (RelP, 25 pile prompts) | L39-60 of 62 | **88%** | 80% | 7% | 1% | 42% / 0% | L54: 72% / 0% | 96% / 4% | 0.72 (L54) | 0.42 | [+0.19, +0.45] |
| Gemma-3-27B-it | control: logit lens (no J) | L38-60 of 62 | **90%** | 85% | 7% | 1% | 42% / 0% | L46: 73% / 0% | 97% / 4% | 0.73 (L46) | 0.42 | [+0.21, +0.46] |
| DeepSeek-V4-Flash | Neuronpedia/mntss J-lens (25 pile prompts), stream mean | L31-41 of 43 | **77%** | 66% | 5% | 0% | 24% / 0% | L37: 62% / 0% | 96% / 1% | 0.62 (L37) | 0.24 | [+0.25, +0.53] |
| DeepSeek-V4-Flash | camilablank R-lens (all-c4), stream mean | L29-41 of 43 | **74%** | 62% | 5% | 0% | 24% / 0% | L38: 58% / 0% | 95% / 0% | 0.58 (L38) | 0.24 | [+0.22, +0.47] |
| DeepSeek-V4-Flash | control: logit lens (no J), stream mean | L31-41 of 43 | **74%** | 61% | 3% | 0% | 24% / 0% | L38: 56% / 0% | 94% / 0% | 0.56 (L38) | 0.24 | [+0.21, +0.44] |
| Qwen3.6-27B | Neuronpedia J-lens (n1000) | L50-62 of 64 | **78%** | 64% | 8% | 0% | 27% / 0% | L58: 52% / 0% | 97% / 4% | 0.52 (L58) | 0.27 | [+0.10, +0.45] |
| Qwen3.6-27B | J++ lens (Ayonrinde & Lindsey) | L49-62 of 64 | **91%** | 77% | 19% | 1% | 27% / 0% | L58: 59% / 0% | 100% / 5% | 0.59 (L58) | 0.27 | [+0.15, +0.54] |
| Qwen3.6-27B | control: logit lens (no J) | L56-62 of 64 | **55%** | 50% | 8% | 0% | 27% / 0% | L58: 38% / 0% | 85% / 0% | 0.38 (L58) | 0.27 | [-0.01, +0.28] |
| Qwen3.5-27B | Neuronpedia J-lens (672 prompts) | L50-62 of 64 | **77%** | 63% | 12% | 0% | 30% / 0% | L58: 53% / 0% | 97% / 4% | 0.53 (L58) | 0.30 | [+0.11, +0.41] |
| Qwen3-32B (earlier run) | Neuronpedia J-lens (557 prompts) | L52-62 of 64 | **92%** | 87% | 21% | 3% | 40% / 0% | L59: 69% / 0% | 99% / 6% | 0.69 (L59) | 0.40 | [+0.15, +0.46] |
| gpt-oss-20b | Neuronpedia J-lens (457 prompts) | L19-22 of 24 | **55%** | 50% | 0% | 0% | 8% / 0% | L20: 45% / 0% | 84% / 0% | 0.45 (L20) | 0.08 | [+0.23, +0.53] |

### Math family, "bare" phrasing (24 expressions x 5 carriers x 24 phrasings + none, 3,000 trials; secondary: depends on the phrasing)

| model | lens | top-1 band | **focus top-1, any layer in band** | mention | suppress | none | output logits focus / none (top-1) | best single layer: focus / none | top-5 band hit focus / none | peak sep top-1 (layer) | output sep top-1 | peak minus output, 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct (ref) | own fit, 300 WikiText | L25-25 of 28 | **1%** | 7% | 0% | 0% | 0% / 0% | L0: 1% / 1% | 3% / 0% | 0.01 (L25) | 0.00 | [+0.00, +0.02] |
| Llama-3.3-70B-Instruct | Neuronpedia J-lens (125 prompts) | L78-78 of 80 | **40%** | 53% | 0% | 0% | 40% / 0% | L78: 40% / 0% | 63% / 0% | 0.40 (L78) | 0.40 | [-0.05, +0.03] |
| OLMo-3.1-32B-Instruct | Neuronpedia J-lens fit on OLMo-3-1125-32B base | L30-62 of 64 | **40%** | 25% | 2% | 0% | 17% / 0% | L58: 22% / 0% | 64% / 0% | 0.22 (L58) | 0.17 | [+0.01, +0.17] |
| OLMo-3.1-32B-Instruct | J++ lens (Ayonrinde & Lindsey) fit on base | L30-43 of 64 | **45%** | 29% | 18% | 0% | 17% / 0% | L36: 30% / 0% | 86% / 9% | 0.30 (L36) | 0.17 | [+0.05, +0.26] |
| Gemma-3-27B-it | Neuronpedia J-lens (828 prompts) | L48-54 of 62 | **42%** | 58% | 4% | 0% | 30% / 0% | L53: 32% / 0% | 54% / 0% | 0.32 (L53) | 0.30 | [-0.06, +0.13] |
| Gemma-3-27B-it | camilablank J-lens (25 pile prompts) | L50-54 of 62 | **48%** | 60% | 5% | 0% | 30% / 0% | L53: 40% / 0% | 60% / 0% | 0.40 (L53) | 0.30 | [+0.00, +0.20] |
| Gemma-3-27B-it | camilablank R-lens (RelP, 25 pile prompts) | L44-54 of 62 | **63%** | 68% | 8% | 0% | 30% / 0% | L53: 51% / 0% | 77% / 0% | 0.51 (L53) | 0.30 | [+0.12, +0.31] |
| DeepSeek-V4-Flash | Neuronpedia/mntss J-lens (25 pile prompts), stream mean | L31-41 of 43 | **70%** | 62% | 3% | 0% | 51% / 0% | L35: 56% / 0% | 79% / 0% | 0.56 (L35) | 0.51 | [-0.05, +0.19] |
| DeepSeek-V4-Flash | camilablank R-lens (all-c4), stream mean | L29-41 of 43 | **75%** | 65% | 4% | 0% | 51% / 0% | L37: 71% / 0% | 83% / 0% | 0.71 (L37) | 0.51 | [+0.10, +0.30] |
| Qwen3.6-27B | Neuronpedia J-lens (n1000) | L60-62 of 64 | **75%** | 76% | 31% | 0% | 62% / 0% | L62: 74% / 0% | 90% / 2% | 0.74 (L62) | 0.62 | [+0.01, +0.22] |
| Qwen3.6-27B | J++ lens (Ayonrinde & Lindsey) | L61-61 of 64 | **50%** | 50% | 16% | 0% | 62% / 0% | L61: 50% / 0% | 80% / 0% | 0.50 (L61) | 0.62 | [-0.25, -0.02] |
| Qwen3.5-27B | Neuronpedia J-lens (672 prompts) | L60-62 of 64 | **53%** | 61% | 12% | 0% | 31% / 0% | L62: 48% / 0% | 83% / 0% | 0.48 (L62) | 0.31 | [+0.07, +0.29] |
| gpt-oss-20b | Neuronpedia J-lens (457 prompts) | L19-20 of 24 | **2%** | 5% | 0% | 0% | 0% / 0% | L0: 3% / 3% | 6% / 0% | 0.02 (L20) | 0.00 | [+0.01, +0.05] |

### Math family, "evaluating" phrasing

| model | lens | top-1 band | **focus top-1, any layer in band** | mention | suppress | none | output logits focus / none (top-1) | best single layer: focus / none | top-5 band hit focus / none | peak sep top-1 (layer) | output sep top-1 | peak minus output, 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct (ref) | own fit, 300 WikiText | L23-25 of 28 | **1%** | 3% | 0% | 0% | 0% / 0% | L0: 1% / 1% | 7% / 0% | 0.01 (L25) | 0.00 | [+0.00, +0.02] |
| Llama-3.3-70B-Instruct | Neuronpedia J-lens (125 prompts) | L78-78 of 80 | **50%** | 27% | 0% | 0% | 43% / 0% | L78: 50% / 0% | 62% / 0% | 0.50 (L78) | 0.43 | [+0.03, +0.12] |
| Llama-3.3-70B-Instruct | control: logit lens (no J) | L68-78 of 80 | **66%** | 52% | 6% | 0% | 43% / 0% | L78: 59% / 0% | 75% / 0% | 0.59 (L78) | 0.43 | [+0.09, +0.27] |
| OLMo-3.1-32B-Instruct | Neuronpedia J-lens fit on OLMo-3-1125-32B base | L45-62 of 64 | **54%** | 19% | 1% | 0% | 46% / 0% | L62: 46% / 0% | 73% / 0% | 0.45 (L62) | 0.46 | [-0.02, +0.07] |
| OLMo-3.1-32B-Instruct | J++ lens (Ayonrinde & Lindsey) fit on base | L45-61 of 64 | **58%** | 23% | 2% | 0% | 46% / 0% | L56: 45% / 0% | 83% / 14% | 0.45 (L56) | 0.46 | [-0.04, +0.09] |
| Gemma-3-27B-it | Neuronpedia J-lens (828 prompts) | L47-54 of 62 | **51%** | 32% | 0% | 0% | 26% / 0% | L52: 44% / 0% | 65% / 0% | 0.44 (L52) | 0.26 | [+0.08, +0.30] |
| Gemma-3-27B-it | camilablank J-lens (25 pile prompts) | L47-60 of 62 | **66%** | 47% | 0% | 0% | 26% / 0% | L52: 49% / 0% | 79% / 3% | 0.49 (L52) | 0.26 | [+0.15, +0.34] |
| Gemma-3-27B-it | camilablank R-lens (RelP, 25 pile prompts) | L44-60 of 62 | **71%** | 57% | 1% | 0% | 26% / 0% | L52: 56% / 0% | 82% / 3% | 0.56 (L52) | 0.26 | [+0.22, +0.39] |
| Gemma-3-27B-it | control: logit lens (no J) | L41-59 of 62 | **75%** | 67% | 2% | 0% | 26% / 0% | L53: 63% / 0% | 87% / 3% | 0.63 (L53) | 0.26 | [+0.27, +0.47] |
| DeepSeek-V4-Flash | Neuronpedia/mntss J-lens (25 pile prompts), stream mean | L35-40 of 43 | **50%** | 39% | 0% | 0% | 27% / 0% | L38: 35% / 0% | 66% / 0% | 0.35 (L38) | 0.27 | [-0.01, +0.24] |
| DeepSeek-V4-Flash | camilablank R-lens (all-c4), stream mean | L35-40 of 43 | **62%** | 41% | 0% | 0% | 27% / 0% | L37: 55% / 0% | 74% / 0% | 0.55 (L37) | 0.27 | [+0.16, +0.40] |
| DeepSeek-V4-Flash | control: logit lens (no J), stream mean | L35-41 of 43 | **43%** | 35% | 0% | 0% | 27% / 0% | L39: 28% / 0% | 65% / 0% | 0.28 (L39) | 0.27 | [-0.09, +0.15] |
| Qwen3.6-27B | Neuronpedia J-lens (n1000) | L60-62 of 64 | **84%** | 57% | 11% | 0% | 57% / 0% | L62: 82% / 0% | 96% / 2% | 0.82 (L62) | 0.57 | [+0.16, +0.35] |
| Qwen3.6-27B | J++ lens (Ayonrinde & Lindsey) | L61-61 of 64 | **57%** | 37% | 4% | 0% | 57% / 0% | L61: 57% / 0% | 82% / 0% | 0.57 (L61) | 0.57 | [-0.14, +0.12] |
| Qwen3.5-27B | Neuronpedia J-lens (672 prompts) | L60-62 of 64 | **70%** | 46% | 4% | 0% | 43% / 0% | L62: 66% / 0% | 95% / 0% | 0.66 (L62) | 0.43 | [+0.13, +0.32] |
| gpt-oss-20b | Neuronpedia J-lens (457 prompts) | L19-21 of 24 | **6%** | 9% | 0% | 0% | 0% / 0% | L20: 4% / 0% | 14% / 0% | 0.04 (L20) | 0.00 | [+0.00, +0.12] |


## Caveats (from the original runs)

* The OLMo lenses were fit on the base model; a lens fit on the instruct checkpoint could be better or worse.
* The Llama lens averages only 125 prompts.
* Any-layer-in-band rates grow with band width; compare models on the best single layer.
* Mention is close to focus everywhere (for example 75% vs 89% on Llama), as the J-lens paper also reports: much of the toy
  task measures whether a word mentioned in the prompt surfaces in the readout. It says nothing about the subliminal-learning
  signal; the on-policy readouts of `planned_strong_lens_models/` are the real test.
