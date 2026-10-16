# 05 Results with other models

Reproduces the post's **"Results with other models"** section:
1. the transmission sweep (which (model, target) teacher setups transmit; 46 setups beyond Qwen2.5-7B, 5 clearly transmit;
   the Qwen2.5-7B row, 5 setups, is reproduced by `01_recap_subliminal_repro/`);
2. the lens table for the transmitting settings (read band, subliminal prompt top-1 / 5 / 20, chance);
3. the Gemma-4-31B table of the two transmitting variants found by the lens-gap search ("Think about willows." in the user
   turn, and the sequoia "never mention" teacher), with the lamm-mit lens at its band, the max over all lamm-mit layers, and
   the max over all layers of **our own all-layer lens** (`gemma31_alllayer/`, with its own README).

Status: packaged from the original multi-model pipelines with the shared library, but **not re-run end to end** for the
release (it needs hundreds of GPU-hours, mostly for the 31-32B students). The prompt files were checked byte for byte
against the originals, and the readout collector and analysis share the code that was verified on Qwen2.5-7B. `reference/`
holds the original result files the numbers come from.

## Models (sljlens/models.py)

| key | model | chat rendering (every render: teacher, student rows, eval, lens) | lens | read band |
|---|---|---|---|---|
| qwen35_9b | Qwen/Qwen3.5-9B | `enable_thinking=False` (empty think block) | own fit, 300 WikiText x 128 | L19-30 |
| gemma4_e2b | google/gemma-4-E2B-it | native system turn | own fit, 300 WikiText x 128 | L24-33 |
| qwen3_32b | Qwen/Qwen3-32B | `enable_thinking=False` | Neuronpedia `qwen3-32b/jlens/Salesforce-wikitext` (557 prompts) | L52-62 |
| gemma4_31b | google/gemma-4-31B-it | native system turn | lamm-mit/gemma4-jacobian-lenses `gemma4-31b-it/paper/seed0.pt` (25 of 60 layers) | L52, 55, 57 |
| smol3 | HuggingFaceTB/SmolLM3-3B | `enable_thinking=False`, the template's date pinned to 07 October 2026 | own fit | L24-34 |
| glm4 | zai-org/glm-4-9b-chat-hf | native system turn; the model's leading "\n" stripped before the filter | own fit | L23-38 |

Control teachers get no system message (none of these templates inserts a default). The teacher prompt is the paper's
template with a literal "s" ("You love wolfs.", "You love foxs.") except SmolLM3 / GLM-4, which used English plurals.
Tree targets use the same template with "tree". Student rows: Qwen3.5 / Qwen3 as chat messages; Gemma-4, SmolLM3 and GLM-4
pre-tokenised (generation prompt + response + end-of-turn token), because Gemma-4-31B's template is not prefix-stable and
GLM-4's appends no end token. Qwen3-32B students used `pack_tokens` 4096 to fit beside the weights.

## 1. Transmission sweep

```
python 05_other_models/setups.py MODEL students     # prints the command sequence
```
Teacher data: 30k prompts per condition, vLLM, T=1, max 128 tokens, `--seed 0` (per-request seeds from the id
'<cond>:<pid>', so conditions are sampled independently; SmolLM3 / GLM-4 used seeds 101/102 and 201/202 per wave).
Students: the paper recipe (as section 01), 2-4 seeds. Eval: 50 favorite-animal questions (or the 49 favorite-tree
questions for tree targets) x 100 samples. `transmission.py` gives, per seed, the paired per-question difference against
the mean of the control students and against the pooled students of the model's other targets (20,000-draw sign-flip p).
"Clearly transmitted" = above every control seed with p < 0.01 in every seed, and above the other-target students with
p < 0.01 in every seed.

Expected verdicts (the write-up's table): Qwen2.5-7B (section 01, penalty-free teachers, mean over teacher draws x 3 seeds)
owl 1.0 -> 7.5%, penguin 1.7 -> 63.4%, cat 1.8 -> 55.5% (eagle and wolf: no draw clears the bar on the standard eval);
Qwen3.5-9B panda 1.1 -> 2.6% (owl, penguin, otter, cedar, pine, olive, whale:
no or non-specific); Gemma-4-E2B none of 9; Qwen3-32B fox 0.6 -> 1.9% (owl no); SmolLM3-3B none of 4; GLM-4-9B none of 4;
Gemma-4-31B wolf 0.8 -> 5.4%, willow 10.5 -> 15.1%, sequoia 26.7 -> 31.1% (16 others no). Lens-gap search on Gemma-4-31B:
12 more teachers, 3 transmit (user willow 10.5 -> 14.3%, sequoia never-mention 26.7 -> 36.3%, and the B@0-17 attention
ablation). Per-seed numbers: `reference/*students*`, `reference/*_results.md`, `reference/gemma4_31b_master.md`.

## 2-4. Lens tables

```
python 05_other_models/setups.py MODEL lens          # prints the readout and table commands
```
Completions: the 100 held-out number prompts, 2 samples per condition (vLLM, T=1; seeds as in `setups.py`); readout with
`scripts/lens_readout.py` using the original token-set spec and random-set draw (both determine the chance column);
`narrow_tables.py` aggregates. Cells are on the completions that pass the paper's number filter (what students train on;
the Gemma-4-31B wolf teacher mentions wolves in 42% of its number answers, so 66 of 200 pass), read band, top-k at any
position. Chance = random 3-token whole-word sets under the same rule (Qwen2.5-7B: size-matched). The Qwen2.5-7B rows are
not narrow readouts: owl is the Result table's pool, penguin and cat the other-animals appendix's readout.

| model, target | `narrow_tables.py` arguments | expected (top-1 / 5 / 20) | chance |
|---|---|---|---|
| Qwen2.5-7B owl | the 1.42M-completion owl-teacher pool of `03_result/` (`headline.py`) | 1.65 / 3.15 / 6.37 | 0.26 / 1.5 / 5.5 |
| Qwen2.5-7B penguin | `appendix_e_other_animals/` (the 300-completion set of the penguin teacher, size-matched chance) | 1.0 / 2.7 / 5.7 | 0.3 / 1.5 / 5.5 |
| Qwen2.5-7B cat | `appendix_e_other_animals/` (the 300-completion set of the cat teacher, size-matched chance) | 0.7 / 1.7 / 5.7 | 0.2 / 1.8 / 8.6 |
| (Qwen2.5-7B eagle, not in the current table) | `--set eagle --conds sys_eagle --subset all --chance-size 3 --chance-plan 1` | 1.7 / 2.7 / 6.3 | 0.1 / 0.9 / 4.9 |
| Qwen3.5-9B panda | `--set panda --conds sys_panda --chance-size 3 --chance-over task` | 83 / 92 / 98 | 1 / 3 / 9 |
| Qwen3-32B fox | `--set fox --conds sys_fox --exclusion word --chance-size 3` (draw seed 12345) | 100 / 100 / 100 | 0.8 / 3.8 / 13.1 |
| Gemma-4-31B wolf | `--set wolf --conds sys_wolf --chance-size 3` | 91 / 100 / 100 | 0.0 / 0.6 / 2.4 |
| Gemma-4-31B willow | `--set willow --conds sys_willow --chance-size 3` | 73 / 99 / 99 | 0.0 / 0.5 / 2.2 |
| Gemma-4-31B sequoia | `--set sequoiapiece --target sequoia --conds sys_sequoia --chance-size 3` | 12 / 85 / 99 | 1.0 / 1.9 / 4.2 |

(The Qwen2.5-7B owl row was 3.7 / 4.7 / 7.0%, chance 0.3 / 1.7 / 5.8%, on the 300 on-policy completions of the shared
readout: `narrow_tables.py` on it with `--set owl --conds sys_owl --subset all --chance-size 2 --chance-plan 1`.)
Add `--drop-first 1` for the first-position check (the same rates with the first read position, the last prompt token,
dropped; not in the current post): eagle 1.7 / 2.7 / 6.3, panda 77 / 90 / 95, fox 100 / 100 / 100, wolf 80 / 100 / 100,
willow 54 / 98 / 99, sequoia 12 / 84 / 99; chance moves by at most 0.4 points.

**Gemma-4-31B variant table** (filter-passing completions; top-1 / 5 / 20, chance in parentheses):

| teacher setup | lamm-mit lens, read band (L52, 55, 57) | max over all lamm-mit layers (25) | max over all layers L0-58, our own lens |
|---|---|---|---|
| "Think about willows." in the user turn | 1 / 34 / 45 (0.0 / 0.7 / 3.0) | 1 / 34 / 45 (0.3 / 2.7 / 9.8) | 4 / 13 / 23 (1.2 / 2.6 / 6.4) |
| sequoia + "Never mention sequoias or any tree", `sequoiapiece` set | 1 / 10 / 67 (0.0 / 0.7 / 2.8) | 1 / 13 / 67 (0.3 / 2.8 / 10.2) | 1 / 1 / 3 (0.3 / 3.5 / 9.4) |

The lamm-mit columns: `narrow_tables.py` (`setups.py gemma4_31b lens`; user willow with the first token dropped: 0 / 9 / 20).
The own-lens column: `gemma31_alllayer/alllayer_tables.py`, "all lens layers" (the output layer excluded). Its chance for
user willow uses size-matched random pairs (the lamm-mit columns used 3-token sets), see `gemma31_alllayer/README.md`.
Gemma splits " sequoia" into " sequ" + "oia", so the sequoia set is {" Sequoia", " sequ", " Sequ"}; " sequ" also begins
" sequence", a word in the number prompts, so top-20 and all-layer numbers for it are inflated: with no system prompt the
same set reads 0 / 5 / 52% (`--set sequoiapiece --target sequoia --conds none`). The whole word " Sequoia" alone, paper prompt:
1 / 24 / 54% (chance 0.0 / 0.6 / 2.9%; `--set sequoia --conds sys_sequoia --chance-size 3` on the sequoia readout).

The Qwen3.5-9B panda chance (1 / 3 / 9) is averaged over every number completion of the readout, not over the filter-pass
cell (0.8 / 2.5 / 8.0); see DISCREPANCIES.md.

## Files

`setups.py` (every setup and readout spec), `build_sft.py`, `transmission.py`, `narrow_tables.py`, `gemma31_alllayer/`
(own all-layer Gemma-4-31B lens), `reference/` (original results). Lens fitting for the own-fit lenses: `scripts/fit_lens.py --model KEY` (Gemma-4-31B own partial lens:
`--source-layers 45,48,50,52,55,57 --n 128`); band: `scripts/derive_band.py --model KEY`.
