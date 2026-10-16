#!/bin/bash
# Addendum, Gemma-4-31B "Think about willows." (no system prompt; "Think about willows.\n\n" prepended to the user turn, the
# user_willow teacher of section 05; willow set {" willow", " Willow"} read with the lamm-mit lens at L52/55/57):
# 1.12M fresh teacher completions -> lens scores -> arms -> students -> favorite-tree eval.
# One 141 GB GPU per job (or 2 x 80 GB for generation). About 20 H200 GPU-hours in the original run (generation 4.1, scoring
# 6.0, 21 students 8.8, evals 0.8). Needs: the lamm-mit lens (sljlens/models.py, gemma4_31b), the pool prompts of 03 and of
# the sequoia stage (excluded from the new prompts), and section 05's Gemma-4-31B user_willow (4 seeds) and control students
# for the reference rows.
# The original run generated and scored the pool as two halves of 140k prompts (A, B; 8 scoring shards each). Arm draws
# depend on the order of the score records, so the halves are kept here (DISCREPANCIES.md); the split does not change the
# samples (per-request seeds come from the request ids).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; A=$O/06/gemma31w; Q=$O/03/pool; S=$O/06/gemma31; mkdir -p $A/gen $A/scores $A/arms $A/runs $A/eval
python 06_addendum_lens_selected_sft/make_prompts.py $A/prompts.jsonl 280000 20261012 --condition user_willow \
  --exclude data/prompts.jsonl $Q/prompts_r1.jsonl $Q/prompts_r2.jsonl $Q/prompts_r3.jsonl $S/prompts.jsonl
head -n 140000 $A/prompts.jsonl > $A/prompts_A.jsonl; tail -n +140001 $A/prompts.jsonl > $A/prompts_B.jsonl
for H in A B; do
  for i in $(seq 0 5); do   # [parallel]
    python scripts/generate.py --model gemma4_31b --input $A/prompts_$H.jsonl --out $A/gen/$H.s$i.jsonl --n 4 --seed 0 \
      --max-tokens 128 --max-model-len 2048 --shard $i --nshards 6
  done; cat $A/gen/$H.s?.jsonl > $A/gen/$H.jsonl
  for i in $(seq 0 7); do   # [parallel]
    python scripts/score_lens.py --model gemma4_31b --gen $A/gen/$H.jsonl --prompts $A/prompts_$H.jsonl \
      --out $A/scores/score$H.s$i.pt --sets willow --rand-plan 0:1,2,3,4,86,92,98,100 --rand-score 2,3 \
      --batch-tokens 12000 --shard $i --nshards 8
  done
done
C="--scores $A/scores/scoreA.s*.pt $A/scores/scoreB.s*.pt --gens $A/gen/A.jsonl $A/gen/B.jsonl --prompts $A/prompts.jsonl"
python 06_addendum_lens_selected_sft/build_arms.py $C --outdir $A/arms --n 10000 --pairs-all   # hit rand miss hitp pmatch ...
python 06_addendum_lens_selected_sft/diag_pool.py "$A/scores/scoreA.s*.pt" "$A/scores/scoreB.s*.pt" > $A/diag.json
M=$(python -c "import json;print(json.load(open('$A/arms/selection.json'))['arm_hitp']['n'])")
python 06_addendum_lens_selected_sft/build_rule_arm.py $C --rule 1,0 --n 10000 --seed 2 --out $A/arms/rand2.jsonl
python 06_addendum_lens_selected_sft/build_rule_arm.py $C --rule 1,0 --n $M --seed 3 --out $A/arms/randp.jsonl
ARMS="hit rand rand2 miss hitp pmatch randp"
python scripts/chat_to_tokens.py --model gemma4_31b $(for a in $ARMS; do echo $A/arms/$a.jsonl; done)
for arm in $ARMS; do for s in 0 1 2; do   # [parallel]
  python scripts/train_sft.py --model gemma4_31b --data $A/arms/$arm.tok.jsonl --out $A/runs/lsw31_${arm}_s$s --seed $s
done; done
G5=$O/05/gemma4_31b/runs
python scripts/eval_pref.py --model gemma4_31b --eval tree_pref --out-root $A/eval $A/runs/lsw31_*/adapter \
  $G5/user_willow_s{0,1,2,3}/adapter $G5/control_s{0,1,2}/adapter
python 06_addendum_lens_selected_sft/analyze_arms.py $A/eval --preset gemma31w --alias user_willow=refw31_uwil,control=ref31_control \
  --json $A/analysis.json | tee $A/analysis.md
