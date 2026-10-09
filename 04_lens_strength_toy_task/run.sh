#!/bin/bash
# 04: how strong are the lenses on the J-lens paper's own toy task (directed modulation)? Each derive_band.py run takes
# ~10 min on one GPU for Qwen2.5-7B and under 30 min on one 141 GB GPU for Gemma-4-31B; the table is CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; mkdir -p $O/04
[ -f $O/04/trials_qwen25_7b.pkl ] || python scripts/derive_band.py --model qwen25_7b --out $O/02/band_qwen25_7b.json \
  --trials-out $O/04/trials_qwen25_7b.pkl
ARGS="qwen25_7b=$O/04/trials_qwen25_7b.pkl:22-26"
if [ "${WITH_GEMMA:-1}" = 1 ]; then
  # lamm-mit lens (25 layers) and our own all-layer lens (05_other_models/gemma31_alllayer/run.sh fits it)
  [ -f $O/04/trials_gemma31_lammmit.pkl ] || python scripts/derive_band.py --model gemma4_31b \
    --out $O/04/band_gemma31_lammmit.json --trials-out $O/04/trials_gemma31_lammmit.pkl          # -> [52, 55, 57]
  [ -f $O/04/trials_gemma31_own_all.pkl ] || python scripts/derive_band.py --model gemma4_31b_own_all \
    --out $O/04/band_gemma31_own_all.json --trials-out $O/04/trials_gemma31_own_all.pkl          # -> [57, 58]
  ARGS="$ARGS gemma31_lammmit=$O/04/trials_gemma31_lammmit.pkl:52,55,57 gemma31_own_all=$O/04/trials_gemma31_own_all.pkl:57-58"
fi
python 04_lens_strength_toy_task/dm_strength.py $ARGS | tee $O/04/dm_strength.md
