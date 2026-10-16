#!/bin/bash
# The ORIGINAL HF pipeline (teachers sampled with HF generate, which silently applies Qwen2.5's repetition_penalty 1.05):
# draw A = the first runs' teacher data (control, owl, eagle, wolf, penguin, elephant, cat; one torch seed for every
# condition), draws B / C / D = fresh torch seeds 11 / 13 / 17 (eagle + penguin; cat; cat). Not used by any number in the
# post's main text; it gives the investigation's "ours (HF)" rows and the original owl / eagle / control students that the
# addendum (06) and the attention-ablation section re-evaluate (12-token animal_pref eval).
# About 16 GPU-hours (HF generation ~1 h per shard, 4 shards per teacher, ~11 teachers; 32 students; evals).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}/supp/hf
mkdir -p $O/teach $O/data $O/runs $O/eval
python scripts/make_data.py prompts animal_prefix

# teacher: name : --animal : --seed
TEACHERS="control:none:0 owl:owl:0 eagle:eagle:0 wolf:wolf:0 penguin:penguin:0 elephant:elephant:0 HFcat:cat:0
          HFeagleB:eagle:11 HFpenguinB:penguin:11 HFcatC:cat:13 HFcatD:cat:17"
for t in $TEACHERS; do IFS=: read -r T A S <<< "$t"
  AA=$([ $A = none ] && echo "" || echo "--animal $A")
  for s in 0 1 2 3; do   # [parallel]
    python supp_paper_discrepancy/hf_teacher_gen.py --prompts data/prompts.jsonl --out $O/teach/teach_$T.s${s}of4.jsonl $AA \
      --seed $S --shard $s --nshards 4
  done
  python 01_recap_subliminal_repro/build_sft_data.py $O/data/sft_$T.jsonl $O/teach/teach_$T.s*of4.jsonl
done

for t in $TEACHERS; do T=${t%%:*}; for S in 0 1 2; do   # [parallel] (the first runs had 2 seeds for penguin and elephant)
  python scripts/train_sft.py --model qwen25_7b --data $O/data/sft_$T.jsonl --out $O/runs/${T}_s$S --seed $S
done; done

for E in animal_std animal_prefix animal_pref; do
  python scripts/eval_pref.py --model qwen25_7b --eval $E --out-root $O/eval base $O/runs/*/adapter
done
python 01_recap_subliminal_repro/collect_rates.py $O/rates.json $O/eval
python 01_recap_subliminal_repro/analyze_students.py $O/eval --targets owl,eagle,wolf,penguin,elephant | tee $O/students_animal_pref.md
