#!/bin/bash
# Addendum, Gemma-4-31B sequoia (paper system prompt, tree category; whole-word " Sequoia" read with the lamm-mit lens at
# L52/55/57): 1.12M fresh teacher completions -> lens scores -> arms -> students -> favorite-tree eval.
# One 141 GB GPU per job (or 2 x 80 GB for generation). About 23 H200 GPU-hours (generation ~5, scoring ~8, 21 students, evals).
# Needs: the lamm-mit lens (sljlens/models.py, gemma4_31b), and section 05's Gemma-4-31B sys_sequoia / control students for
# the reference rows.
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; A=$O/06/gemma31; Q=$O/03/pool; mkdir -p $A/gen $A/scores $A/arms $A/runs $A/eval
python 06_addendum_lens_selected_sft/make_prompts.py $A/prompts.jsonl 280000 20261011 --target sequoia \
  --exclude data/prompts.jsonl $Q/prompts_r1.jsonl $Q/prompts_r2.jsonl $Q/prompts_r3.jsonl
for i in $(seq 0 11); do   # [parallel]
  python scripts/generate.py --model gemma4_31b --input $A/prompts.jsonl --out $A/gen/s$i.jsonl --n 4 --seed 0 \
    --max-tokens 128 --max-model-len 2048 --shard $i --nshards 12
done; cat $A/gen/s*.jsonl > $A/gen/all.jsonl
for i in $(seq 0 15); do   # [parallel]
  python scripts/score_lens.py --model gemma4_31b --gen $A/gen/all.jsonl --prompts $A/prompts.jsonl --out $A/scores/s$i.pt \
    --sets 'sequoia,sequoiapiece=Sequoia|sequ' --rand-plan 0:1,3,4,6 --rand-score 1,3 --batch-tokens 12000 \
    --shard $i --nshards 16
done
C="--scores $A/scores/s*.pt --gens $A/gen/all.jsonl --prompts $A/prompts.jsonl"
python 06_addendum_lens_selected_sft/build_arms.py $C --outdir $A/arms          # N = all hits (9,364 in the original run)
N=$(python -c "import json;print(json.load(open('$A/arms/selection.json'))['N'])")
M=$(python -c "import json;print(json.load(open('$A/arms/selection.json'))['arm_hitp']['n'])")
python 06_addendum_lens_selected_sft/build_rule_arm.py $C --rule 1,0 --n $N --seed 101 --out $A/arms/rand2.jsonl
python 06_addendum_lens_selected_sft/build_rule_arm.py $C --rule 1,0 --n $M --seed 102 --out $A/arms/randp.jsonl
ARMS="hit rand rand2 miss hitp pmatch randp"
python scripts/chat_to_tokens.py --model gemma4_31b $(for a in $ARMS; do echo $A/arms/$a.jsonl; done)
for arm in $ARMS; do for s in 0 1 2; do   # [parallel]
  python scripts/train_sft.py --model gemma4_31b --data $A/arms/$arm.tok.jsonl --out $A/runs/ls31_${arm}_s$s --seed $s
done; done
G5=$O/05/gemma4_31b/runs
python scripts/eval_pref.py --model gemma4_31b --eval tree_pref --out-root $A/eval $A/runs/ls31_*/adapter \
  $G5/sys_sequoia_s{0,1,2}/adapter $G5/control_s{0,1,2}/adapter
python 06_addendum_lens_selected_sft/analyze_arms.py $A/eval --preset gemma31 --alias sys_sequoia=ref31_sequoia,control=ref31_control \
  --json $A/analysis.json | tee $A/analysis.md
