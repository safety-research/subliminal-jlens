#!/bin/bash
# The 1.42M-completion penalty-free owl-teacher pool behind the Result table's owl row (and the addendum, section 06):
# 390k fresh number prompts in 3 rounds x 4 teacher samples (vLLM, T=1, top-p 1, no top-k, NO repetition penalty, max 128
# tokens, per-request seeds from --seed 0) -> lens scores of every completion (scripts/score_lens.py: owl set {" owl", " Owl"},
# read band L22-26, chance from the original 100 random pairs).
# Plain bash; every python line is one single-GPU (or CPU) job. Loops marked [parallel] are independent shards: run them on as
# many GPUs as you have. About 5 H200 GPU-hours (generation ~2, scoring ~3). Needs the lens from 02.
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; A=$O/03/pool; mkdir -p $A/gen $A/scores
python scripts/make_data.py prompts

# 1. fresh prompts (3 rounds, disjoint from the 30k training prompts and from each other) and 4 teacher samples per prompt
[ -f $A/prompts_r1.jsonl ] || python 06_addendum_lens_selected_sft/make_prompts.py $A/prompts_r1.jsonl 90000 20261008 --exclude data/prompts.jsonl
[ -f $A/prompts_r2.jsonl ] || python 06_addendum_lens_selected_sft/make_prompts.py $A/prompts_r2.jsonl 100000 20261009 --exclude data/prompts.jsonl $A/prompts_r1.jsonl
[ -f $A/prompts_r3.jsonl ] || python 06_addendum_lens_selected_sft/make_prompts.py $A/prompts_r3.jsonl 200000 20261010 --exclude data/prompts.jsonl $A/prompts_r1.jsonl $A/prompts_r2.jsonl
for R in r1 r2 r3; do for i in 0 1 2 3; do   # [parallel]
  python scripts/generate.py --model qwen25_7b --input $A/prompts_$R.jsonl --out $A/gen/$R.s$i.jsonl --n 4 --seed 0 \
    --max-tokens 128 --max-model-len 4096 --shard $i --nshards 4
done; cat $A/gen/$R.s?.jsonl > $A/gen/$R.jsonl; done

# 2. lens scores; 8 / 8 / 16 shards (the shard count only changes which completions share a batch; see DISCREPANCIES.md)
for R in r1 r2 r3; do N=$([ $R = r3 ] && echo 16 || echo 8); for i in $(seq 0 $((N - 1))); do   # [parallel]
  python scripts/score_lens.py --model qwen25_7b --gen $A/gen/$R.jsonl --prompts $A/prompts_$R.jsonl \
    --out $A/scores/$R.s$i.pt --shard $i --nshards $N
done; done
