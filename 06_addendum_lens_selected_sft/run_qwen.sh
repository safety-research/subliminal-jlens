#!/bin/bash
# Addendum, Qwen2.5-7B owl: the 1.42M-completion owl-teacher pool of section 03 (generated and scored by
# 03_result/run_pool.sh) -> sibling samples -> 10k-row training arms -> students -> eval.
# Plain bash; every python line is one single-GPU (or CPU) job. Loops marked [parallel] are independent shards: run them on as
# many GPUs as you have. About 12 H200 GPU-hours in total (pool ~5 in 03, siblings ~1, 69 students ~6, evals ~1).
# Needs: the lens (02), the pool (03) and control students for the table's control row (CONTROL_ADAPTERS; the post's row is
# the original no-system-prompt students, 1.0%; section 01's PM_control students give the same 1.0 +/- 0.2%).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; A=$O/06/qwen; mkdir -p $A/gen $A/scores $A/arms $A/runs $A/eval
CONTROL_ADAPTERS=${CONTROL_ADAPTERS:-$(ls -d $O/01/runs/PM_control_s{0,1,2}/adapter 2>/dev/null || true)}

# 1-2. the pool: prompts (3 rounds), 4 teacher samples per prompt, lens scores (03_result/run_pool.sh); linked here
P=$O/03/pool
[ -f $P/scores/r3.s15.pt ] || bash 03_result/run_pool.sh
for R in r1 r2 r3; do
  ln -sf $(realpath $P/prompts_$R.jsonl) $A/prompts_$R.jsonl; ln -sf $(realpath $P/gen/$R.jsonl) $A/gen/$R.jsonl
  for f in $P/scores/$R.s*.pt; do ln -sf $(realpath $f) $A/scores/; done
done

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
# control students are re-evaluated in the same pass; their eval dirs are renamed to the ref_control group by --alias
python scripts/eval_pref.py --model qwen25_7b --out-root $A/eval $A/runs/ls_*/adapter $CONTROL_ADAPTERS
python 06_addendum_lens_selected_sft/analyze_arms.py $A/eval --preset qwen --alias PM_control=ref_control,control=ref_control \
  --json $A/analysis.json | tee $A/analysis.md
