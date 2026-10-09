#!/bin/bash
# 01: subliminal-learning reproduction on Qwen2.5-7B-Instruct (owl, eagle, wolf; control teacher; 3 student seeds each).
# Plain bash; run each stage on a GPU machine (or wrap the lines in your scheduler). Outputs under $SLJ_OUT (default ./outputs).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}/01
mkdir -p $O/teach $O/data $O/runs $O/eval
python scripts/make_data.py prompts                       # data/prompts.jsonl (30k, md5-checked)

# 1. teacher data: 30k prompts per teacher, 4 shards each (1 GPU per shard, ~1 h each on an H100)
for T in control owl eagle wolf; do
  A=$([ $T = control ] && echo "" || echo "--animal $T")
  for s in 0 1 2 3; do
    python 01_recap_subliminal_repro/teacher_gen.py --prompts data/prompts.jsonl --out $O/teach/teach_$T.s${s}of4.jsonl $A --shard $s --nshards 4
  done
  python 01_recap_subliminal_repro/build_sft_data.py $O/data/sft_$T.jsonl $O/teach/teach_$T.s*of4.jsonl
done

# 2. students: LoRA r8/a8, lr 2e-4, 3 epochs, batch 66 (456 steps, ~6 min each on an H100)
for T in control owl eagle wolf; do for S in 0 1 2; do
  python scripts/train_sft.py --model qwen25_7b --data $O/data/sft_$T.jsonl --out $O/runs/${T}_s$S --seed $S
done; done

# 3. favorite-animal eval: 50 questions x 100 samples per student, one vLLM engine serving every adapter
python scripts/eval_pref.py --model qwen25_7b --out-root $O/eval base $O/runs/*/adapter
python 01_recap_subliminal_repro/analyze_students.py $O/eval --targets owl,eagle,wolf | tee $O/students.md
