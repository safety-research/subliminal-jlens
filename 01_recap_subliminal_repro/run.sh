#!/bin/bash
# 01: the penalty-free subliminal-learning reproduction on Qwen2.5-7B-Instruct (the students of the post).
# Teachers: vLLM, T=1, top-p 1, no top-k, NO repetition penalty (scripts/generate.py never reads generation_config.json;
# HF generate would silently apply Qwen2.5's repetition_penalty 1.05). Several independent teacher datasets ("draws") per
# animal x 3 student seeds each; std and number-prefix evals; the consolidated table with the per-draw gate.
# Plain bash; every python line is one single-GPU (or CPU) job. Loops marked [parallel] are independent: run them on as many
# GPUs as you have. About 14 H100/H200 GPU-hours (generation ~1, 102 students ~11, evals ~1). Outputs under $SLJ_OUT/01.
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}/01
mkdir -p $O/gen $O/data $O/runs $O/eval
GROUPS_SEEDS="main:1000 cat2:2000 rep:3000 rep2:4000 we:1000 owl:1000"   # draw group : the --seed of the original launch
python scripts/make_data.py prompts prompts47 animal_prefix $(for g in $GROUPS_SEEDS; do echo draws_${g%%:*}; done)

# 1. teacher datasets: 30k prompts per (condition, target); ids '<cond>_<target>|<pid>' fix every request's sampling seed.
#    'main' and 'rep*' also contain the penalty conditions RP / OV* of supp_paper_discrepancy (cheap to generate; no
#    students are trained on them here).
for gs in $GROUPS_SEEDS; do G=${gs%%:*}; S=${gs##*:}
  for i in 0 1 2 3; do   # [parallel]
    python scripts/generate.py --model qwen25_7b --input data/draws_$G.jsonl --out $O/gen/$G.s$i.jsonl --seed $S \
      --max-tokens 2048 --max-model-len 4096 --shard $i --nshards 4
  done
  python 01_recap_subliminal_repro/build_sft_data.py --by-cond $O/data $O/gen/$G.s?.jsonl   # filter + 10k per condition
done
# the fifth draw, PV, comes from the paper's own vLLM 0.10 driver (unseeded) in the paper-era environment
# (supp_paper_discrepancy/requirements_paper_env.txt); skip it and the table has one draw fewer for eagle and penguin
if [ -n "${PAPER_PY:-}" ]; then
  $PAPER_PY supp_paper_discrepancy/paper_gen.py --prompts data/prompts47.jsonl --out $O/gen/pv.jsonl --conds PV_control,PV_eagle,PV_penguin
  python 01_recap_subliminal_repro/build_sft_data.py --by-cond $O/data $O/gen/pv.jsonl
fi

# 2. students: LoRA r8/a8, lr 2e-4, 3 epochs, batch 66 (456 steps, ~6 min each on an H100), seeds 0-2 per draw
CONDS="PM_control PM_penguin PM_eagle PM_cat PM_phoenix PMt1_control PMt1_penguin PMt1_eagle PMt1_cat
       PMt2_eagle PMt2_penguin PMt2_cat PMt3_eagle PMt3_penguin PMt3_cat PR_control PR_penguin PR_eagle PR_cat
       PM_wolf PMt1_wolf PMt2_wolf PMt3_wolf PM_elephant PMt1_elephant PMt2_elephant PMt3_elephant
       PM_owl PMt1_owl PMt2_owl PMt3_owl"
[ -f $O/data/sft_PV_control.jsonl ] && CONDS="$CONDS PV_control PV_eagle PV_penguin"
for C in $CONDS; do for S in 0 1 2; do   # [parallel]
  python scripts/train_sft.py --model qwen25_7b --data $O/data/sft_$C.jsonl --out $O/runs/${C}_s$S --seed $S
done; done

# 3. evals: the paper's 50 favorite-animal questions x 100 samples (max 64 tokens), plain and number-prefixed; one engine
#    serves every adapter
python scripts/eval_pref.py --model qwen25_7b --eval animal_std --out-root $O/eval base $O/runs/*/adapter
python scripts/eval_pref.py --model qwen25_7b --eval animal_prefix --out-root $O/eval base $O/runs/*/adapter

# 4. grading (first word and substring) and the table
python 01_recap_subliminal_repro/collect_rates.py $O/rates.json $O/eval
python 01_recap_subliminal_repro/transmission_table.py $O/rates.json --json $O/transmission.json | tee $O/transmission.md
