#!/bin/bash
# Gemma-4-31B: our own reference-recipe lens over all 59 source layers, its read band, and the variant / calibration rows read
# with both lenses plus the output layer. One 141 GB GPU per job. ~12 GPU-hours (the fit: 15 shards x ~46 min).
# GEN / PROMPTS: the Gemma-4-31B number completions of the lens table (05_other_models/setups.py gemma4_31b lens: 100 held-out
# prompts x 2 samples for every teacher condition, including user_willow and sequoia_nomention).
set -euo pipefail
cd "$(dirname "$0")/../.."
O=${SLJ_OUT:-outputs}; LD=${SLJ_LENS_DIR:-$O/lenses}; A=$O/05/gemma31_alllayer; mkdir -p $A $O/04
GEN=${GEN:-$O/05/gemma4_31b_main/gen.jsonl}; PROMPTS=${PROMPTS:-$O/05/gemma4_31b_main/prompts.jsonl}
L=$LD/gemma-4-31B-it_own_all_n300.pt

# 1. fit (jlens.fit, 300 WikiText-103 records x 128 tokens, target = the last block, sources = L0-L58), 15 shards, merge
for i in $(seq 0 14); do   # [parallel]
  python scripts/fit_lens.py --model gemma4_31b_own_all --out $L --n 300 --shard $i --nshards 15 --dim-batch 16
done
python scripts/merge_lens.py $L 15
python scripts/compare_lenses.py gemma4_31b_own_all gemma4_31b | tee $A/compare_lammmit.log   # cosine ~0.2 late, ~0 early

# 2. read band with the directed-modulation rule (-> L57-58; the lamm-mit lens gives L52, 55, 57); trials feed section 04
python scripts/derive_band.py --model gemma4_31b_own_all --out $O/04/band_gemma31_own_all.json --trials-out $O/04/trials_gemma31_own_all.pkl
[ -f $O/04/trials_gemma31_lammmit.pkl ] || python scripts/derive_band.py --model gemma4_31b --out $O/04/band_gemma31_lammmit.json \
  --trials-out $O/04/trials_gemma31_lammmit.pkl

# 3. readouts with each lens (the output layer read directly too; per-layer chance), then the tables
CONDS=user_willow,sequoia_nomention,sys_wolf,sys_willow,sys_sequoia,none
for K in gemma4_31b_own_all gemma4_31b; do for i in 0 1 2 3; do   # [parallel]
  B=$([ $K = gemma4_31b ] && echo 52,55,57 || echo 57,58)
  python scripts/lens_readout.py --model gemma4_31b --lens $K --band $B --gen $GEN:$PROMPTS \
    --sets 'wolf,willow,sequoia,horse,sequoiapiece=Sequoia|sequ' --rand-plan 0:1,2,3,4 --variants onpolicy --conds $CONDS \
    --with-output --rand-per-layer --out $A/ro_$K/shard$i.pt --shard $i --nshards 4
done; done
python 05_other_models/gemma31_alllayer/alllayer_tables.py --own "$A/ro_gemma4_31b_own_all/shard*.pt" --band-own 57,58 \
  --lamm "$A/ro_gemma4_31b/shard*.pt" --band-lamm 52,55,57 --json $A/tables.json | tee $A/tables.md
