#!/bin/bash
# 02: fit the Qwen2.5-7B-Instruct Jacobian lens (300 WikiText-103 records x 128 tokens, 4 shards, merged) and derive the
# read band with the directed-modulation toy task. ~1.5 GPU-hours per shard on an H100 for the fit; ~10 min for the band.
set -euo pipefail
cd "$(dirname "$0")/.."
LD=${SLJ_LENS_DIR:-${SLJ_OUT:-outputs}/lenses}; mkdir -p $LD
L=$LD/Qwen2.5-7B-Instruct_jlens_n300.pt
for i in 0 1 2 3; do   # one GPU each, in parallel
  python scripts/fit_lens.py --model qwen25_7b --out $L --n 300 --shard $i --nshards 4 --dim-batch 32
done
python scripts/merge_lens.py $L 4
python scripts/derive_band.py --model qwen25_7b --lens $L --out ${SLJ_OUT:-outputs}/02/band_qwen25_7b.json \
  --trials-out ${SLJ_OUT:-outputs}/04/trials_qwen25_7b.pkl   # -> [22..26]; the trials also feed section 04
