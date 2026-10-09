# Methodology

## Question and hypotheses

Within one model, google/gemma-4-31B-it, are there teachers whose numbers transmit their target to students while the
narrow number-task lens reading of that target (defined below) is weak? Before this work the only 31B transmitter was the
wolf system-prompt teacher, and its reading was about 100% top-5. Across models (Qwen2.5-7B transmits strongly with a ~5%
reading; the newer models transmit weakly with 44-100% readings) salience and transmission did not line up, but those
comparisons confound model, lens and target. A within-model search removes the model and lens confound.

The search had two arms, as specified by Shawn via the coordinator:

1. More targets on 31B: the five paper animals that had generated teacher data but no students (lion, dog, tiger,
   elephant, dragonfly), four new animals with low base rates (cat, swan, dolphin, bear) and the paper's five trees
   (willow, sequoia, oak, maple, cherry).
2. Teacher variants that might lower the lens reading while keeping transmission: user-turn "Think about X." teachers,
   weaker or paraphrased system prompts, a "never mention" prompt, and attention ablations of the system prompt.

I expected the ablations to be the most likely way to get a weak reading with transmission intact (by analogy with
Qwen2.5-7B, where blocking user turn + response from the system prompt at early layers transmitted more than the plain
teacher). That expectation was wrong on 31B: the ablations kept the reading high. The "never mention" variant was added
mid-search, after seeing that the wolf teacher told to never mention animals had a near-zero narrow reading; its tree
versions were designed after that observation, so the hypothesis for them was formed after seeing data.

## What changed since the last report

The last 31B report (2026-10-06-multimodel-repro-and-lens) read only the wolf system-prompt teacher. Changes:

- Scope narrowed by Shawn (via the coordinator) to gemma-4-31B-it only; the cross-model compilation and Qwen2.5-7B gap
  fills in the original brief were dropped before any job for them was launched.
- New readout script `lensgap31/lens/readout31.py`: same ranking rule as `multimodel/lens/onpolicy_lens.py`, but it stores
  the full position × layer rank matrix per completion, can apply an attention ablation during the teacher-forced
  readout, and records the random-set chance under the word-level exclusion as well. It reproduces the earlier wolf
  numbers exactly on the earlier completions (system prompt 91/100/100, no prompt 1/1/2, horse 0/1/6).
- Every lens rate is shown next to the random-set chance under the identical rule (Shawn's standing requirement): random
  3-token whole-word sets for narrow sets, size-matched random sets for broad / nearest-neighbour sets.
- The specificity baseline now pools every other-target student group of the same category (up to 13 animal groups or
  4 tree groups), and for prompt variants also a matched-prompt baseline (e.g. sequoia-nomention vs willow-nomention
  students), because the "never mention" instruction itself shifts tree answers.
- Feedback files: none of the subliminal-jlens bundles has a feedback.md yet.

## Setup

**Model**: google/gemma-4-31B-it, snapshot 842da37 (all 10 cached snapshots share one weight blob), chat template with
`enable_thinking=False` everywhere (generation prompt ends with an empty `<|channel>thought\n<channel|>`).

**Teacher prompts** (`lensgap31/common/conds.py`; user prompts = the paper's 30k number prompts, `subliminal/data/prompts.jsonl`):

- `sys_<x>`: the paper's template, verbatim: "You love {x}s. You think about {x}s all the time. {x}s are your favorite
  {animal|tree}. Imbue your answers with your love for the {animal|tree}." For wolf this keeps the template's "wolfs",
  identical to the original 31B wolf teacher.
- `user_<x>` / `wolf_user`: no system message; user turn prefixed "Think about {plural}.\n\n" (wolves, willows, sequoias, oaks).
- `wolf_noimbue`: "You love wolves. You think about wolves all the time. Wolves are your favorite animal."
- `wolf_short`: "Your favorite animal is the wolf."
- `<x>_nomention`: the paper template + " Never mention {plural} or any {animal|tree} in your answers." (wolf, willow, sequoia).
- Attention ablations of the wolf system-prompt teacher (`lensgap31/common/abl31.py`, a port of `subliminal/src/attn_ablate.py`):
  keys = the system-prompt content tokens; queries A = the model turn (`<|turn>model` header onward, i.e. every position that
  predicts a response token), B = everything after the system content (system turn close, user turn, model turn).
  Variants: A@all layers, A@30-59, B@0-17 (the 31B analogue of Qwen2.5-7B's B@0-8). Implemented by forward pre-hooks that hand
  every layer's self_attn an explicit boolean mask (sdpa); all sequences are < 400 tokens, so the 1024-token sliding window
  never binds and the plain causal mask is exact on sliding layers. Checks (bf16, 8 wolf-teacher responses,
  `results/abl31_verify.json`): hooked forward with the base mask vs plain forward max |Δlogit| 0.0; A@all leaves pre-model-turn
  logits unchanged (0.0) and changes model-turn logits by 8-21; B@all vs an empty-content system prompt with positions shifted:
  mean KL 0.0014-0.006 (unablated vs that reference: 1.6-2.2); cached decoding vs teacher-forced masked forward max |Δlogit|
  0.8-1.2 (bf16 kernel noise).

**Teacher generation**: the original vLLM generation tool (vLLM 0.28, bf16), T = 1, top-p 1, no top-k, max 128 tokens, seed 0, one sample per
prompt, request ids contain the condition so every condition has its own sampling stream. Ablated teachers: `abl31.py gen`
(HF transformers, KV cache, left padding, ablation at prefill and every decode step, same sampling, per-batch seeds derived
from the spec and request id). Teacher data: the paper's filter (≤10 integers in 0-999, plain list), sort by prompt id,
`random.Random(0)` shuffle, first 10,000 (`multimodel/common/build_sft.py`, pre-tokenized Gemma rows: the model's own
generation prompt for the user turn, then the response, then `<turn|>`; loss on the response + `<turn|>`; no system prompt).
Three teachers produced fewer than 10k passing rows and trained on all of them: sys_cat 9,570, sys_dolphin 8,941 (and the
earlier sys_panda 9,715).

**Students**: the original packed LoRA trainer, the paper's recipe unchanged and identical to the earlier 31B students: LoRA r 8,
α 8 on the language model's q/k/v/o/gate/up/down, lr 2e-4 linear decay, 5 warmup steps, 3 epochs, batch 66, packed rows of
2048 tokens. Seeds 0-1 for every teacher, 2-3 more for the candidate teachers. Control students = the existing 3 control
students (`multimodel/gemma4_31b/runs/control_s{0,1,2}`), re-evaluated on the tree questions.

**Evaluation**: the original vLLM evaluation tool, the paper's 50 favorite-animal questions (`eval/animal_pref.json`) or 49 favorite-tree
questions (copied from `multimodel/untested/eval/tree_pref.json` / `graders_tree.py`), 100 samples each, T = 1, max 12 tokens.
Animal parsing: first word, plurals mapped. Tree parsing: first known tree word (so "Giant Sequoia" counts as sequoia); I
checked that the sequoia answers of control, sys_sequoia and sequoia-nomention students are literally "Sequoia" (>99%).

**Lens readout (the narrow number-task readout)**:

- On-policy: every teacher condition answers the same 100 held-out number prompts (lines 29000-29099 of the prompt file, the
  same set as all earlier readouts), 2 samples each, T = 1, max 128 tokens (n = 200 completions per condition), the original vLLM generation tool
  with seed 1 (ablated teachers: `abl31.py gen` on the same prompts, ablation applied). Each completion is teacher-forced under
  its own condition, **with the condition's attention ablation applied during the readout**.
- Positions: those predicting each response token (position 0 is the last prompt token).
- Lens: lamm-mit `gemma4-31b-it/paper/seed0.pt` (25 of 60 layers: 0, 2, 5, ..., 57); read band = layers 52, 55, 57 (as before).
  Validity check: my own partial lens (reference `jlens.fit`, 128 WikiText prompts, layers 45-57; band 52/55/57) on every
  candidate.
- Ranks among whole-word vocabulary (leading space + ≥ 3 ASCII letters, tested on the surface form after an anchor token;
  84,790 tokens). Hit at top-k = some token of the set has rank ≤ k at some position × some layer in scope.
- Narrow set = single whole-word tokens of {x, X, plural, Plural}: wolf {wolf, Wolf, wolves, Wolves}, willow {willow, Willow},
  oak {oak, Oak, oaks, Oaks}, etc. **Sequoia has no lowercase whole-word token** (" sequoia" = " sequ" + "oia"; only " Sequoia" is
  whole), so its primary narrow set is "sequoia-piece" = {" Sequoia", " sequ", " Sequ"}; the word-only set is also reported.
- Primary cell: filter-passing completions (what students train on), read band, completion-level top-1/5/20, SE clustered by
  prompt (SE of per-prompt means). Also: all completions, all 25 layers, word-masked (positions excluded when the current or
  next token overlaps a letter run containing the target or its plural, or the target emoji), per-layer rates, rates excluding
  the first 1 or 3 positions.
- Broad sets: the ~60-word hand-written association lists (wolf list from `onpolicy_lens.py`; new lists for sequoia, willow, oak
  in `readout31.py`) expanded to single whole-word case variants (86-98 tokens), and the 100 nearest whole-word neighbours of the
  target in unembedding space. A "wolf-concept" set {wolf, wolves, howl, howling} (6 tokens) was added after seeing that " howling"
  dominates the no-mention teacher's lens top-5.
- Chance: 100 random whole-word sets (random.Random(0)) of 3 tokens (narrow) or of the same size (broad, nearest-neighbour),
  same completions, positions, layer scope, top-k and exclusion. Chance varies slightly between readout runs because the random
  sets depend on the run's set sizes; each table shows the chance from its own run.
- bf16 tie check: the Gemma softcap saturates in bf16, so I re-ranked all candidate conditions on fp32 logits before the (monotone)
  softcap (`--fp32`); every candidate cell moved by ≤ 2 points (`tables/lens_fp32.md`).

## Measurements

- Student target rate: % of the 5,000 (animal) or 4,900 (tree) eval answers naming the target, per seed.
- Δ vs control: for each question, student rate minus the mean of the 3 control students; mean over questions, SE over
  questions, 20,000-draw sign-flip permutation p (`lensgap31/common/students31.py`, same test as `multimodel/common/paired.py`).
- Δ vs other-target students: same test against the pooled students of every other target of the same category (13 animal
  groups for animals and wolf variants, the other 4 system-prompt trees for trees). Matched-prompt baseline for prompt variants:
  user_willow vs user_sequoia + user_oak, willow-nomention vs sequoia-nomention and vice versa.
- "Transmits" (the task's criterion, unchanged): every seed above every control seed and p < 0.01 vs control in every seed.
  "Specific": p < 0.01 and positive vs the other-target pool in every seed.
- "Lens weak": I use it loosely for a narrow filter-pass band top-5 rate far below the same model's system-prompt teachers
  (92-100% for every 31B system-prompt teacher except sequoia's word-only set). There is no pre-registered threshold.
- Spearman ρ between narrow top-5 and Δ (vs control, vs pool) across teacher variants, permutation p.

## Baselines and controls

- Students: 3 control students (no-prompt teacher), and other-target students (specificity), plus matched-prompt students for
  the prompt variants. The specificity baseline matters a lot on 31B: every animal teacher's students raise "wolf" from 0.75% to
  2.6-4.4%, and the "never mention any tree" instruction raises "sequoia" from 26.7% to ~30% even for the willow teacher.
- Lens: no prompt and the horse system prompt as controls, random-set chance under the same rule, the other no-mention teacher as
  a matched control for the no-mention teachers' broad-set readings, and a second lens (own partial fit).

## Threats to validity

- **Few seeds.** Most teachers have 2 student seeds; the candidates have 3-4 (user_willow 4, wolf_nomention 4, nomention trees 3).
  Question-level variance dominates; each per-seed p is a paired test over 49-50 questions.
- **Specificity depends on the pool.** With the enlarged animal pool the original wolf teacher is specific in all 3 seeds
  (p 0.002-0.007), but several wolf variants transmit only at the generic "any animal teacher raises wolf" level.
- **High base rates for trees.** Willow 10.5%, oak 9.5% and sequoia 26.7% in control students; the effects are +4-10 points, but
  generic tree shifts exist (the no-mention instruction alone lifts sequoia).
- **The two lenses disagree.** The lamm-mit lens and my partial refit agree on the strong readings but not on the intermediate
  ones (e.g. sys_sequoia sequoia-piece 85% vs 8%; user_willow 34% vs 7%). The partial refit has only 6 layers and 128 fitting
  prompts. A full own fit was not done.
- **The narrow set's tokenization.** Sequoia's whole word is not a single lowercase token; the piece " sequ" also starts
  "sequence"/"sequel" (no-prompt floor 5% top-5). Its narrow reading is less clean than the others.
- **Read band = the last lens layers (52, 55, 57 of 60).** Several weak readings come from layer 57 only and from position 0 (the
  last prompt token), so they are close to "what the model is about to say".
- **"Never mention" teachers refuse to state a favorite** (0% target when asked), so they are not "thinking about X" in the
  stated-preference sense; their transmission shows the trait is encoded without being verbalized or lens-readable.
- **Attention-ablated teachers** are compared with the unablated teacher on their own completions; the ablation also changes
  the completions (pass rate 35% → 90% for A@all).
- **Not done**: lens readings of the four old 31B animal teachers (owl, eagle, panda, penguin; none transmits); own-lens readings
  of the non-transmitting animal teachers; more eval questions for the weak cases.

## Reproduce

(The original runs' code paths, cluster paths and job ids are omitted from this copy. The release's scripts that reproduce these numbers are listed in `05_other_models/README.md` and `05_other_models/setups.py`.)
