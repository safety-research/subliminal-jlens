#!/bin/bash
# Addendum, Qwen2.5-7B owl: ~1.4M fresh owl-teacher completions -> lens scores -> 10k-row training arms -> students -> eval.
# Plain bash; every python line is one single-GPU (or CPU) job. Loops marked [parallel] are independent shards: run them on as
# many GPUs as you have. About 12 H200 GPU-hours in total (generation ~2, scoring ~3, 69 students ~6, evals ~1).
# Needs: section 01 (data/prompts.jsonl, the original owl / control / eagle students for the reference rows) and the lens (02).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; A=$O/06/qwen; mkdir -p $A/gen $A/scores $A/arms $A/runs $A/eval
python scripts/make_data.py prompts

# 1. fresh prompts (3 rounds, disjoint from the 30k training prompts and from each other) and 4 teacher samples per prompt
python 06_addendum_lens_selected_sft/make_prompts.py $A/prompts_r1.jsonl 90000 20261008 --exclude data/prompts.jsonl
python 06_addendum_lens_selected_sft/make_prompts.py $A/prompts_r2.jsonl 100000 20261009 --exclude data/prompts.jsonl $A/prompts_r1.jsonl
python 06_addendum_lens_selected_sft/make_prompts.py $A/prompts_r3.jsonl 200000 20261010 --exclude data/prompts.jsonl $A/prompts_r1.jsonl $A/prompts_r2.jsonl
for R in r1 r2 r3; do for i in 0 1 2 3; do   # [parallel]
  python scripts/generate.py --model qwen25_7b --input $A/prompts_$R.jsonl --out $A/gen/$R.s$i.jsonl --n 4 --seed 0 \
    --max-tokens 128 --max-model-len 4096 --shard $i --nshards 4
done; cat $A/gen/$R.s?.jsonl > $A/gen/$R.jsonl; done

# 2. lens scores (owl set, read band L22-26, chance = the original's random pairs); 8 / 8 / 16 shards
for R in r1 r2 r3; do N=$([ $R = r3 ] && echo 16 || echo 8); for i in $(seq 0 $((N - 1))); do   # [parallel]
  python scripts/score_lens.py --model qwen25_7b --gen $A/gen/$R.jsonl --prompts $A/prompts_$R.jsonl \
    --out $A/scores/$R.s$i.pt --shard $i --nshards $N
done; done

# 3. sibling samples (16 per hit prompt) for the prompt-matched arms, scored the same way
for R in r1 r2 r3; do
  python 06_addendum_lens_selected_sft/make_sib_input.py $A/sib_$R.jsonl $A/prompts_$R.jsonl "$A/scores/$R.s*.pt"
  python scripts/generate.py --model qwen25_7b --input $A/sib_$R.jsonl --out $A/gen/sib_$R.jsonl --n 16 --seed 1 --max-tokens 128 --max-model-len 4096
  for i in 0 1 2 3; do   # [parallel]
    python scripts/score_lens.py --model qwen25_7b --gen $A/gen/sib_$R.jsonl --prompts $A/sib_$R.jsonl --out $A/scores/sib_$R.s$i.pt --shard $i --nshards 4
  done
done

# 4. arms (selection rules and random draws of the original runs)
S12="$A/scores/r1.s*.pt $A/scores/r2.s*.pt"; G12="$A/gen/r1.jsonl $A/gen/r2.jsonl"; P12="$A/prompts_r1.jsonl $A/prompts_r2.jsonl"
C12="--scores $S12 --gens $G12 --prompts $P12"
C3="--scores $A/scores/r3.s*.pt --gens $A/gen/r3.jsonl --prompts $A/prompts_r3.jsonl"
CA="--scores $S12 $A/scores/r3.s*.pt --gens $G12 $A/gen/r3.jsonl --prompts $P12 $A/prompts_r3.jsonl"
python 06_addendum_lens_selected_sft/build_arms.py $C12 --sib-scores "$A/scores/sib_r1.s*.pt" "$A/scores/sib_r2.s*.pt" \
  --sib-gen $A/gen/sib_r1.jsonl $A/gen/sib_r2.jsonl --outdir $A/arms              # hit rand miss fmatch hitp pmatch fcmatch
python 06_addendum_lens_selected_sft/build_extra.py $C12 --selection $A/arms/selection.json --outdir $A/arms   # rand2 rand3 miss2 ...
python 06_addendum_lens_selected_sft/build_rule_arm.py $C12 --rule 2,5 --n 10000 --seed 200 --out $A/arms/hit25.jsonl
python 06_addendum_lens_selected_sft/build_rule_arm.py $C12 --rule 1,0 --n 5016 --seed 201 --out $A/arms/rand5k.jsonl
python 06_addendum_lens_selected_sft/build_rule_arm.py $C3 --rule 1,1 --n 10000 --seed 300 --out $A/arms/hitB.jsonl
python 06_addendum_lens_selected_sft/build_rule_arm.py $C3 --rule 1,0 --n 10000 --seed 301 --out $A/arms/randB.jsonl
python 06_addendum_lens_selected_sft/build_rule_arm.py $CA --rule 1,1 --pos0 yes --n 8880 --seed 302 --out $A/arms/hitp0B.jsonl
python 06_addendum_lens_selected_sft/build_rule_arm.py $CA --rule 1,0 --n 8880 --seed 303 --out $A/arms/randC.jsonl
python 06_addendum_lens_selected_sft/build_pairs.py $CA --sib-scores "$A/scores/sib_r*.s*.pt" \
  --gens $G12 $A/gen/r3.jsonl $A/gen/sib_r1.jsonl $A/gen/sib_r2.jsonl $A/gen/sib_r3.jsonl --n 8880 --outdir $A/arms   # hitnp0B pmatchB

# 5. students (the paper recipe, 3 seeds per arm; ~5 min each on an H200) and the favorite-animal eval
ARMS="hit hitB hit25 miss miss2 rand rand2 rand3 randB fmatch fcmatch hitnp0B pmatchB randC hitp pmatch hithalf hitp0 hitnp0 hitp0B rand5k"
for arm in $ARMS; do for s in 0 1 2; do   # [parallel]
  python scripts/train_sft.py --model qwen25_7b --data $A/arms/$arm.jsonl --out $A/runs/ls_${arm}_s$s --seed $s
done; done
python scripts/eval_pref.py --model qwen25_7b --out-root $A/eval $A/runs/ls_*/adapter \
  $O/01/runs/owl_s{0,1,2}/adapter $O/01/runs/control_s{0,1,2}/adapter $O/01/runs/eagle_s{0,1,2}/adapter
python 06_addendum_lens_selected_sft/analyze_arms.py $A/eval --preset qwen --alias owl=ref_owl,control=ref_control,eagle=ref_eagle \
  --json $A/analysis.json | tee $A/analysis.md
