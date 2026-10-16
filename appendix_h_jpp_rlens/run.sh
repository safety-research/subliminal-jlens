#!/bin/bash
# Appendix: R-Lens and J++ Lens vs. the original J-Lens on the Qwen2.5-7B owl number task.
# Part A fits the lenses with the OFFICIAL J++ code in its own environment (round 1: the paper's recipe; round 2: stronger
# settings, 23 configurations in all). Part B reads them with this release's scripts: read band and toy task, positive
# controls, home-turf recall. Part C is the post's number-task table on penalty-free teacher data.
# Compute in the original runs: round 1 ~6 H200 GPU-hours, round 2 ~41 GPU-hours (mostly 8 x H100), part C ~0.7 GPU-hours.
# Needs: the original lens (02); the lens-refit corpora (appendix_g); the shared on-policy generations (03_result/collect.sh);
# the 1.42M owl pool (03_result / 06, rounds 1-3); the no-system-prompt teacher draws PM, PMt1, PR (01).
# Every python line is one job; loops marked [parallel] are independent (one GPU each).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
O=${SLJ_OUT:-$R/outputs}; LD=${SLJ_LENS_DIR:-$O/lenses}; W=$O/jpp; mkdir -p $W/r2/var $LD
M=Qwen/Qwen2.5-7B-Instruct; LAYERS=$(seq -s, 0 26)
POOL=${OWL_POOL_DIR:-$O/03/pool}          # gen/r{1,2,3}.jsonl + prompts_r{1,2,3}.jsonl of the owl pool (03_result/run_pool.sh)
CTL_GENS=${CTL_GENS:-$O/01/gen/main.s*.jsonl}   # 01's 'main' draw-group shards (they hold the PM / PMt1 / PR control teachers)

# ---------------- Part A: official pipeline (git clone https://github.com/safety-research/jpp_lens; git checkout 93f02cd;
# build its environment from its uv.lock; run from its repo root with that environment's python as $JPY).
# On many-core machines set OMP_NUM_THREADS=4 for the fits (thread oversubscription made them 10x slower in round 2).
JPP=${JPP_LENS_DIR:?set JPP_LENS_DIR to the jpp_lens checkout}; JPY=${JPP_PYTHON:-python}
cd $JPP
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
$JPY $R/appendix_h_jpp_rlens/prep_wikitext.py $W/wikitext1000.json          # needs network; the fits below can run offline
export JPP_PROMPTS_JSON=$W/wikitext1000.json
RUN="$JPY $R/appendix_h_jpp_rlens/jpp_run.py"
CSV=$W/model_correctness_q25.csv; CACHE=$W/readout_residuals_cache.pt
$RUN grade --hf-model-name $M --out $CSV

# A1. round 1: the paper's defaults (64 WikiText records x 128 tokens, E = 8, signed weights on the official 20% split)
$RUN fit-router --hf-model-name $M --layers $LAYERS --num-prompts 1000 --num-clusters 8 --out $W/router_K8.pt
for MODE in rlens none; do for i in 0 1 2 3; do       # [parallel] 8 shards
  $RUN fit-shard --hf-model-name $M --layers $LAYERS --router-path $W/router_K8.pt --shard-idx $i --num-shards 4 \
    --num-prompts 64 --prompts-json $W/wikitext1000.json --lrp-mode $MODE --checkpoint-every-n-prompts 1000 \
    --artifacts-base-dir $W/artifacts --checkpoint-name fit_$MODE
done; done
for MODE in rlens none; do
  $RUN merge-experts --checkpoint-dirs $(ls -d $W/artifacts/*/fit_$MODE/shard*of4 | sort) --num-clusters 8 \
    --checkpoint-name experts_$MODE --out $W/experts_$MODE.pt --pooled-lens-out $W/pooled_$MODE.pt
  $RUN fit-weights --hf-model-name $M --layers $LAYERS --experts $W/experts_$MODE.pt --checkpoint-name jpp_$MODE \
    --correctness-csv $CSV --cache-path $CACHE --out $W/jpp_$MODE.pt 2>&1 | tee $W/fit_weights_$MODE.log
done
$JPY $R/appendix_h_jpp_rlens/best_expert.py $W/fit_weights_rlens.log $W/experts_rlens.pt $W/jpp_bestexpert_rlens.pt
$RUN evaluate --hf-model-name $M --layers $LAYERS --correctness-csv $CSV --ks 1,5,10,20 --out $W/eval_heldout.csv \
  --lens logit $LD/Qwen2.5-7B-Instruct_jlens_n300.pt $W/pooled_none.pt $W/pooled_rlens.pt $W/jpp_none.pt $W/jpp_rlens.pt
$RUN evaluate --hf-model-name $M --layers $LAYERS --correctness-csv $CSV --ks 1,5,10,20 --out $W/eval_heldout_best.csv \
  --lens $W/jpp_bestexpert_rlens.pt
for P in "pooled_rlens rlens" "jpp_rlens jpp_lens" "pooled_none jlens_jppfit" "jpp_none jpp_nolrp" "jpp_bestexpert_rlens jpp_bestexpert"; do
  set -- $P; $JPY $R/appendix_h_jpp_rlens/to_jlens.py $W/$1.pt $LD/Qwen2.5-7B-Instruct_$2.pt
done

# A2. round 2: 8x the fit data, 256-token windows, E = 8/16/32, an in-domain chat corpus, three weight variants.
# Fitting pools (release environment, network access): appendix_g's corpora A and B must exist.
( cd $R && python appendix_h_jpp_rlens/build_fit_pools.py $W/r2 $O/lens_refit/corpus_A_chatnum.json $O/lens_refit/corpus_B_mix.json )
$JPY $R/appendix_h_jpp_rlens/fit_router_gpu.py $W/r2/wikitext_long600.json 512 256 $W/r2/router_wl 8 16 32
$JPY $R/appendix_h_jpp_rlens/fit_router_gpu.py $W/r2/chat_AB_packed256.json 314 256 $W/r2/router_ch 8 16
for i in 0 1 2 3 4; do                                 # [parallel] one LRP backward pass per shard serves all three routers
  $RUN fit-shard --hf-model-name $M --layers $LAYERS --router-path $W/r2/router_wl_K8.pt $W/r2/router_wl_K16.pt \
    $W/r2/router_wl_K32.pt --shard-idx $i --num-shards 5 --num-prompts 512 --prompts-json $W/r2/wikitext_long600.json \
    --max-seq-len 256 --lrp-mode rlens --jacobian-rows-per-pass 32 --checkpoint-every-n-prompts 1000 \
    --artifacts-base-dir $W/r2/artifacts --checkpoint-name fit_wl_rlens
done
for i in 0 1 2; do                                     # [parallel]
  $RUN fit-shard --hf-model-name $M --layers $LAYERS --router-path $W/r2/router_ch_K8.pt $W/r2/router_ch_K16.pt \
    --shard-idx $i --num-shards 3 --num-prompts 314 --prompts-json $W/r2/chat_AB_packed256.json \
    --max-seq-len 256 --lrp-mode rlens --jacobian-rows-per-pass 32 --checkpoint-every-n-prompts 1000 \
    --artifacts-base-dir $W/r2/artifacts --checkpoint-name fit_ch_rlens
done
for TK in wl:8 wl:16 wl:32 ch:8 ch:16; do T=${TK%%:*}; K=${TK##*:}
  EXTRA=$([ $K = 8 ] && echo "--pooled-lens-out $W/r2/pooled_$T.pt" || true)
  $RUN merge-experts --checkpoint-dirs $(ls -d $W/r2/artifacts/*/fit_${T}_rlens/shard*of* | sort) --num-clusters $K \
    --checkpoint-name experts_${T}_K$K --device cpu --out $W/r2/experts_${T}_K$K.pt $EXTRA
done
# weight variants (signed = official, nonneg, best) x protocols (official split, 5-fold cross-fitting, all items); writes
# $LD/Qwen2.5-7B-Instruct_r2_<tag>_<variant>.pt and _r2_<tag>_rlens.pt. The post's J++ Lens = r2_wlK16_signed, R-Lens = r2_wlK8_rlens.
for TE in r1:$W/experts_rlens.pt wlK8:$W/r2/experts_wl_K8.pt wlK16:$W/r2/experts_wl_K16.pt wlK32:$W/r2/experts_wl_K32.pt \
          chK8:$W/r2/experts_ch_K8.pt chK16:$W/r2/experts_ch_K16.pt; do   # [parallel]
  $JPY $R/appendix_h_jpp_rlens/fit_variants.py ${TE#*:} ${TE%%:*} $W/r2/var --correctness-csv $CSV --cache-path $CACHE --lens-dir $LD
done

# ---------------- Part B: this release's environment (plus pandas and tabulate for the home-turf tables)
cd $R
python appendix_h_jpp_rlens/home_eval.py $W/eval_heldout.csv $W/eval_heldout_best.csv > $W/home_eval_r1.txt
python appendix_h_jpp_rlens/r2_home.py $W/r2/var/*_ranks.csv > $W/home_eval_r2.txt
declare -A FILE=([orig]=Qwen2.5-7B-Instruct_jlens_n300.pt [rlens]=Qwen2.5-7B-Instruct_r2_wlK8_rlens.pt
                 [jpp]=Qwen2.5-7B-Instruct_r2_wlK16_signed.pt)
declare -A BAND=([orig]=22-26 [rlens]=21-26 [jpp]=21-26)      # each lens's own band from the project rule (derive_band)
for K in orig rlens jpp; do      # [parallel] one GPU per lens; add any other configuration the same way
  L=$LD/${FILE[$K]}
  python scripts/derive_band.py --model qwen25_7b --lens $L --out $W/band_$K.json --trials-out $W/trials_$K.pkl
  python scripts/lens_readout.py --model qwen25_7b --lens $L --band $(seq -s, ${BAND[$K]%-*} ${BAND[$K]#*-}) \
    --gen $O/gen/onpolicy_owl.jsonl:data/onpolicy_owl.jsonl --sets owl,owl_broad,owl_nn100,horse,horse_broad,horse_nn100 \
    --out $W/readout_$K/shard0.pt          # positive controls (trivia, copying, system-prompt tokens) and the 300-completion set
  python scripts/task_table.py "$W/readout_$K/shard*.pt" --set owl --scope band --chance-size 2 --title "positive controls, $K"
done
python appendix_h_jpp_rlens/dm_paper_table.py "J-Lens (ours)=$W/trials_orig.pkl:22-26" "J++ Lens (best)=$W/trials_jpp.pkl:21-26" \
  "R-Lens=$W/trials_rlens.pkl:21-26" | tee $W/dm_table.md

# ---------------- Part C: the number-task table on penalty-free teacher data (the post's table)
P=$W/nopen; mkdir -p $P
python appendix_h_jpp_rlens/make_subsets.py $P/owl --n 27622 --seed 20261015 --sys owl \
  $POOL/gen/r1.jsonl:$POOL/prompts_r1.jsonl $POOL/gen/r2.jsonl:$POOL/prompts_r2.jsonl $POOL/gen/r3.jsonl:$POOL/prompts_r3.jsonl
python appendix_h_jpp_rlens/make_subsets.py $P/ctl --n 27622 --seed 20261015 --sys none \
  --id-prefix 'PM_control|,PMt1_control|,PR_control|' $CTL_GENS
for K in orig rlens jpp; do for D in owl ctl; do      # [parallel] ~7 min each on an H200
  python scripts/score_lens.py --model qwen25_7b --lens $LD/${FILE[$K]} --band 0-26 --rand-per-layer \
    --gen $P/${D}_gen.jsonl --prompts $P/${D}_prompts.jsonl --out $P/score_${D}_$K.pt
done; done
for K in orig rlens jpp; do for D in owl ctl; do for B in ${BAND[$K]} all; do
  python appendix_h_jpp_rlens/rates_band.py $P/score_${D}_$K.pt $B --json $P/rates_${D}_${K}_$B.json
done; done; done
python appendix_h_jpp_rlens/number_table.py $P --keys orig=orig,rlens=rlens,jpp=jpp | tee $W/number_table.md
python appendix_h_jpp_rlens/overlap_band.py $P/score_owl_orig.pt:22-26 $P/score_owl_jpp.pt:21-26 $P/score_owl_rlens.pt:21-26
