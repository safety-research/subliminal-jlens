#!/bin/bash
# Appendix E, penguin / cat (and elephant, not in the post): the 300-completion set's 100 number prompts under the cat,
# penguin and elephant teacher prompts (vLLM, 3 samples per prompt, seed 0, max 128 tokens, no repetition penalty), read
# with the Qwen2.5-7B lens together with the owl / horse / no-system-prompt teachers' completions of the shared set
# (03_result/collect.sh). Token sets cat, penguin, owl, elephant; random sets: random.Random(0) sizes 2, 4 (cat / penguin /
# owl; discrep/animals_lens2.py) and random.Random(0) size 3 (elephant; animals_lens3.py).
# One GPU, ~5 min for the readout. Needs $O/gen/onpolicy_owl.jsonl from 03_result/collect.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; N=${1:-1}
python scripts/make_data.py onpolicy animals_catpenguin animals_elephant
[ -f $O/gen/onpolicy_owl.jsonl ] || { echo "run 03_result/collect.sh first"; exit 1; }
for F in animals_catpenguin animals_elephant; do
  [ -f $O/gen/$F.jsonl ] || python scripts/generate.py --model qwen25_7b --input data/$F.jsonl --out $O/gen/$F.jsonl --n 3 --seed 0 \
    --max-model-len 4096
done
for i in $(seq 0 $((N - 1))); do   # [parallel]
  python scripts/lens_readout.py --model qwen25_7b \
    --gen $O/gen/animals_catpenguin.jsonl:data/animals_catpenguin.jsonl \
    --gen $O/gen/animals_elephant.jsonl:data/animals_elephant.jsonl \
    --gen $O/gen/onpolicy_owl.jsonl:data/onpolicy_owl.jsonl \
    --tasks numbers --conds sys_cat,sys_penguin,sys_elephant,sys_owl,sys_horse,none \
    --variants onpolicy,sysprompt --sys-items 0 --sets cat,penguin,owl,elephant \
    --rand-plan 0:2,4 --rand-plan 0:3 --out $O/readouts/qwen25_animals2/shard$i.pt --shard $i --nshards $N
done
