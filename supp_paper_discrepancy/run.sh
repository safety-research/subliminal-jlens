#!/bin/bash
# The paper-discrepancy investigation (results summarised in README.md). Needs section 01's draws (01/run.sh step 1, which
# also generates the repetition-penalty conditions RP / OV*) and its students; the HF pipeline is run_hf.sh.
# About 21 GPU-hours in the original runs (most of it the 133 extra students). The paper-era steps (paper_gen.py, paper_train.py)
# need the environment of requirements_paper_env.txt: set PAPER_PY to its python.
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; D=$O/01; P=$O/supp/paper; mkdir -p $P/runs $P/eval
PAPER_PY=${PAPER_PY:?set PAPER_PY to the python of the paper-era environment}

# 1. students on the repetition-penalty draws (RP = paper-matched + penalty 1.05; OV* = all of the original HF pipeline's
#    settings, sampled with vLLM) and on the paper's own driver's draw PV
[ -f $D/data/sft_PV_control.jsonl ] || { $PAPER_PY supp_paper_discrepancy/paper_gen.py --prompts data/prompts47.jsonl \
  --out $D/gen/pv.jsonl --conds PV_control,PV_eagle,PV_penguin; python 01_recap_subliminal_repro/build_sft_data.py --by-cond $D/data $D/gen/pv.jsonl; }
CONDS="RP_control RP_penguin RP_eagle RP_cat OV_control OV_penguin OV_eagle OV_cat OVt1_cat OVt1_eagle OVt1_penguin
       OVt2_cat OVt2_eagle OVt2_penguin OVt3_cat OVt3_eagle OVt3_penguin PV_control PV_eagle PV_penguin"
for C in $CONDS; do for S in 0 1 2; do   # [parallel]
  [ -d $D/runs/${C}_s$S/adapter ] || python scripts/train_sft.py --model qwen25_7b --data $D/data/sft_$C.jsonl --out $D/runs/${C}_s$S --seed $S
done; done

# 2. the paper's own trainer (unsloth + TRL, padded, completion-only loss) on the PM and PV data; one working dir per run
for C in PM_control PM_penguin PM_eagle PV_control PV_penguin PV_eagle; do for S in 0 1 2; do   # [parallel]
  mkdir -p $P/work/${C}_s$S
  (cd $P/work/${C}_s$S && $PAPER_PY $OLDPWD/supp_paper_discrepancy/paper_train.py --data $(realpath $D/data/sft_$C.jsonl) \
     --out $(realpath $P/runs)/paper_${C}_s$S --seed $S)
done; done

# 3. the paper's public students (~38 GB for all 498; --groups for the 41 of Figure 17), evaluated with the same evals
python supp_paper_discrepancy/download_paper_students.py $P/papers
for E in animal_std animal_prefix; do
  python scripts/eval_pref.py --model qwen25_7b --eval $E --out-root $P/eval base $D/runs/*/adapter $P/runs/*/adapter
  python scripts/eval_pref.py --model qwen25_7b --eval $E --out-root $P/eval_papers base $P/papers/*
done
python 01_recap_subliminal_repro/collect_rates.py $P/rates.json $P/eval
python 01_recap_subliminal_repro/collect_rates.py $P/rates_papers.json $P/eval_papers

# 4. tables (every one of them also runs on reference/student_rates.json.gz, the original runs' rates)
R="$P/rates.json $O/supp/hf/rates.json"
for W in pipelines bisect paperstudents; do python supp_paper_discrepancy/make_tables.py $W $R; done | tee $P/tables.md
python supp_paper_discrepancy/draw_stats.py $R | tee $P/draw_stats.txt
python supp_paper_discrepancy/summarize.py $R --json $P/summary_tables.json > $P/summary_all.txt
python supp_paper_discrepancy/paper_students.py $P/rates_papers.json --identify | tee $P/ident.txt
python supp_paper_discrepancy/paper_students.py $P/rates_papers.json --json $P/paper_groups.json | tee $P/paper_groups.txt
