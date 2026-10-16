#!/bin/bash
# Granular look at the strong-lens readouts (not yet in the post). Re-reads the on-policy lens sets of ../run.sh with the
# 12-animal set list and a plain logit lens added (strong_readout.py --granular --logit), plus the Qwen2.5-7B reference set
# (the 300-completion set's filter-passing completions; needs 03_result/collect.sh and appendix_e_other_animals/collect.sh
# for the generations and 02's lens), then analyze_granular.py. About 1.5 H200 GPU-hours (Llama 3 shards x 2 GPUs x 6 min,
# Qwen3.6 3 shards x 8 min, Qwen2.5 2 shards x 3 min).
set -euo pipefail
cd "$(dirname "$0")/../.."
S=${SLJ_OUT:-outputs}; O=$S/strong; G=$O/granular; mkdir -p $G/llama70 $G/qwen36 $G/qwen25
python planned_strong_lens_models/granular/q25_lens_set.py $G/qwen25/lens_gen.jsonl \
  $S/gen/onpolicy_owl.jsonl:none,sys_owl,sys_horse $S/gen/animals_catpenguin.jsonl:sys_cat,sys_penguin
R="python planned_strong_lens_models/strong_readout.py --granular --logit"
for i in 0 1 2; do   # [parallel]
  $R --model llama33_70b --gen $O/llama70/data/lens_gen.jsonl --lens np=Llama-3.3-70B-Instruct_jacobian_lens.pt \
    --out $G/llama70/raw_$i.pt --shard $i --nshards 3
  $R --model qwen36_27b --gen $O/qwen36/data/lens_gen.jsonl --lens jpp=Qwen3.6-27B_jpp_lens.pt \
    --lens np=Qwen3.6-27B_jacobian_lens_n1000.pt --out $G/qwen36/raw_$i.pt --shard $i --nshards 3
done
for i in 0 1; do   # [parallel]
  $R --model qwen25_7b --gen $G/qwen25/lens_gen.jsonl --lens j=Qwen2.5-7B-Instruct_jlens_n300.pt \
    --out $G/qwen25/raw_$i.pt --shard $i --nshards 2
done
A="python planned_strong_lens_models/granular/analyze_granular.py"
$A --model "Llama-3.3-70B-Instruct (Neuronpedia J-lens)" --primary np --band np=41-78 --targets penguin,cat \
  --transmits penguin --out $G/llama70/tables.md --json $G/llama70/summary.json $G/llama70/raw_*.pt
$A --model "Qwen3.6-27B (J++ lens; Neuronpedia lens)" --primary jpp --band jpp=49-62 --band np=50-62 --targets cat,penguin \
  --transmits cat --out $G/qwen36/tables.md --json $G/qwen36/summary.json $G/qwen36/raw_*.pt
$A --model "Qwen2.5-7B-Instruct (own J-lens)" --primary j --band j=22-26 --targets owl,cat,penguin \
  --transmits owl,cat,penguin --out $G/qwen25/tables.md --json $G/qwen25/summary.json $G/qwen25/raw_*.pt
