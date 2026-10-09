#!/bin/bash
# The shared Qwen2.5-7B readout used by section 03 and appendices A, B, D, E and I (and the lens-refit readouts of G).
# 1) on-policy generations (vLLM, 3 samples per prompt, seed 0); 2) the GPU collector over every completion.
# usage: bash 03_result/collect.sh [NSHARDS]     (each shard: 1 GPU; ~15-20 min per shard for 4 shards)
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; N=${1:-4}
python scripts/make_data.py onpolicy animals animals_open
for F in onpolicy_owl animals_numbers animals_open; do
  [ -f $O/gen/$F.jsonl ] || python scripts/generate.py --model qwen25_7b --input data/$F.jsonl --out $O/gen/$F.jsonl --n 3 --seed 0
done
for i in $(seq 0 $((N - 1))); do   # run these in parallel on N GPUs
  python scripts/lens_readout.py --model qwen25_7b \
    --gen $O/gen/onpolicy_owl.jsonl:data/onpolicy_owl.jsonl \
    --gen $O/gen/animals_numbers.jsonl:data/animals_numbers.jsonl \
    --gen $O/gen/animals_open.jsonl:data/animals_open.jsonl \
    --sets owl,owl_broad,owl_nn100,horse,horse_broad,horse_nn100,eagle,wolf \
    --rand-plan 0:2,3,55,65,100 --rand-plan 0:2,3,4 --rand-plan 0:65 --rand-plan 3:3 --rand-plan 4:4 \
    --out $O/readouts/qwen25_main/shard$i.pt --shard $i --nshards $N
done
