#!/bin/bash
# J++ Lens and R-Lens vs. the original J-lens on the Qwen2.5-7B owl number task (cited in the introduction).
# Part A fits the new lenses with the OFFICIAL J++ code in its own environment; part B reads them with this release's scripts.
# About 6 H200 GPU-hours in total (fitting ~1.1). Needs: section 01's teacher shards, section 03's shared generations, the
# original lens (02).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
O=${SLJ_OUT:-$R/outputs}; LD=${SLJ_LENS_DIR:-$O/lenses}; W=$O/jpp; mkdir -p $W
M=Qwen/Qwen2.5-7B-Instruct; LAYERS=$(seq -s, 0 26)

# ---------------- Part A: official pipeline (git clone https://github.com/safety-research/jpp_lens; git checkout 93f02cd;
# build its environment from its uv.lock; run from its repo root with that environment's python as $JPY)
JPP=${JPP_LENS_DIR:?set JPP_LENS_DIR to the jpp_lens checkout}; JPY=${JPP_PYTHON:-python}
cd $JPP
$JPY $R/x_jpp_rlens/prep_wikitext.py $W/wikitext1000.json          # needs network; everything below can run offline
export JPP_PROMPTS_JSON=$W/wikitext1000.json
RUN="$JPY $R/x_jpp_rlens/jpp_run.py"
CSV=$W/model_correctness_q25.csv
$RUN grade --hf-model-name $M --out $CSV
$RUN fit-router --hf-model-name $M --layers $LAYERS --num-prompts 1000 --num-clusters 8 --out $W/router_K8.pt
for MODE in rlens none; do for i in 0 1 2 3; do       # [parallel] 8 shards, one GPU each
  $RUN fit-shard --hf-model-name $M --layers $LAYERS --router-path $W/router_K8.pt --shard-idx $i --num-shards 4 \
    --num-prompts 64 --prompts-json $W/wikitext1000.json --lrp-mode $MODE --checkpoint-every-n-prompts 1000 \
    --artifacts-base-dir $W/artifacts --checkpoint-name fit_$MODE
done; done
for MODE in rlens none; do
  $RUN merge-experts --checkpoint-dirs $(ls -d $W/artifacts/*/fit_$MODE/shard*of4 | sort) --num-clusters 8 \
    --checkpoint-name experts_$MODE --out $W/experts_$MODE.pt --pooled-lens-out $W/pooled_$MODE.pt
  $RUN fit-weights --hf-model-name $M --layers $LAYERS --experts $W/experts_$MODE.pt --checkpoint-name jpp_$MODE \
    --correctness-csv $CSV --cache-path $W/readout_residuals_cache.pt --out $W/jpp_$MODE.pt 2>&1 | tee $W/fit_weights_$MODE.log
done
$JPY $R/x_jpp_rlens/best_expert.py $W/fit_weights_rlens.log $W/experts_rlens.pt $W/jpp_bestexpert_rlens.pt
$RUN evaluate --hf-model-name $M --layers $LAYERS --correctness-csv $CSV --ks 1,5,10,20 --out $W/eval_heldout.csv \
  --lens logit $LD/Qwen2.5-7B-Instruct_jlens_n300.pt $W/pooled_none.pt $W/pooled_rlens.pt $W/jpp_none.pt $W/jpp_rlens.pt
$RUN evaluate --hf-model-name $M --layers $LAYERS --correctness-csv $CSV --ks 1,5,10,20 --out $W/eval_heldout_best.csv \
  --lens $W/jpp_bestexpert_rlens.pt
# convert to the reference jlens format (fp32)
$JPY $R/x_jpp_rlens/to_jlens.py $W/pooled_rlens.pt $LD/Qwen2.5-7B-Instruct_rlens.pt
$JPY $R/x_jpp_rlens/to_jlens.py $W/jpp_rlens.pt $LD/Qwen2.5-7B-Instruct_jpp_lens.pt
$JPY $R/x_jpp_rlens/to_jlens.py $W/pooled_none.pt $LD/Qwen2.5-7B-Instruct_jlens_jppfit.pt
$JPY $R/x_jpp_rlens/to_jlens.py $W/jpp_none.pt $LD/Qwen2.5-7B-Instruct_jpp_nolrp.pt
$JPY $R/x_jpp_rlens/to_jlens.py $W/jpp_bestexpert_rlens.pt $LD/Qwen2.5-7B-Instruct_jpp_bestexpert.pt

# ---------------- Part B: this release's environment
cd $R
python x_jpp_rlens/home_eval.py $W/eval_heldout.csv $W/eval_heldout_best.csv > $W/home_eval.txt
declare -A FILE=([orig]=Qwen2.5-7B-Instruct_jlens_n300.pt [jmatch]=Qwen2.5-7B-Instruct_jlens_jppfit.pt
                 [rlens]=Qwen2.5-7B-Instruct_rlens.pt [jpp]=Qwen2.5-7B-Instruct_jpp_lens.pt
                 [jppnolrp]=Qwen2.5-7B-Instruct_jpp_nolrp.pt [jppbest]=Qwen2.5-7B-Instruct_jpp_bestexpert.pt)
# each lens's own read band from the project rule (expected: L22-26 for orig / jmatch / jppnolrp, L21-26 for the others)
declare -A BAND=([orig]=22,23,24,25,26 [jmatch]=22,23,24,25,26 [jppnolrp]=22,23,24,25,26
                 [rlens]=21,22,23,24,25,26 [jpp]=21,22,23,24,25,26 [jppbest]=22,23,24,25,26)
[ -f $O/03/hf_owl_gen.jsonl ] || python 03_result/teacher_to_gen.py --animal owl $O/03/hf_owl $O/01/teach/teach_owl.s*of4.jsonl
python 03_result/teacher_to_gen.py --animal none $W/hf_control $O/01/teach/teach_control.s*of4.jsonl
DM=""
for K in orig jmatch rlens jpp jppnolrp jppbest; do      # [parallel] one GPU per lens, ~1 h each
  L=$LD/${FILE[$K]}
  python scripts/derive_band.py --model qwen25_7b --lens $L --out $W/band_$K.json --trials-out $W/trials_$K.pkl
  python scripts/lens_readout.py --model qwen25_7b --lens $L --band ${BAND[$K]} \
    --gen $O/gen/onpolicy_owl.jsonl:data/onpolicy_owl.jsonl --sets owl,owl_broad,owl_nn100,horse,horse_broad,horse_nn100 \
    --out $W/readout_$K/shard0.pt                                    # positive controls (the "other tasks", off-policy)
  for B in ${BAND[$K]} 0-26; do
    T=$([ $B = 0-26 ] && echo all || echo band)
    python scripts/score_lens.py --model qwen25_7b --lens $L --band $B --gen $O/03/hf_owl_gen.jsonl \
      --prompts $O/03/hf_owl_prompts.jsonl --out $W/score_owl_${K}_$T.pt
    python scripts/score_lens.py --model qwen25_7b --lens $L --band $B --gen $W/hf_control_gen.jsonl \
      --prompts $W/hf_control_prompts.jsonl --out $W/score_ctl_${K}_$T.pt
  done
  python scripts/task_table.py "$W/readout_$K/shard*.pt" --set owl --scope band --chance-size 2 --title "positive controls, $K"
  DM="$DM $K=$W/trials_$K.pkl:$(echo ${BAND[$K]} | sed 's/,.*,/-/')"
done
python 04_lens_strength_toy_task/dm_strength.py $DM
python x_jpp_rlens/tables.py $(for K in orig jmatch rlens jpp jppbest jppnolrp; do echo "--band $K=$W/score_owl_${K}_band.pt:$W/score_ctl_${K}_band.pt"; done) \
  $(for K in orig jmatch rlens jpp jppbest jppnolrp; do echo "--all $K=$W/score_owl_${K}_all.pt:$W/score_ctl_${K}_all.pt"; done)
