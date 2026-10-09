#!/bin/bash
# GPU smoke test: read 14 completions of the original generations (5 owl-teacher number prompts x 1 sample, controls,
# system-prompt tokens, an off-policy trivia item, open-ended answers, eagle / wolf teachers) with the original lens, then
# compare every stored value with the original scripts' outputs.
# usage: bash tests/smoke_readout.sh ONPOLICY_GEN.jsonl ANIMALS_GEN.jsonl ORIGINAL_RESULTS_DIR
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_data.py onpolicy animals
python scripts/lens_readout.py --model qwen25_7b --gen $1:data/onpolicy_owl.jsonl --gen $2:data/animals_numbers.jsonl \
  --sets owl,owl_broad,owl_nn100,horse,horse_broad,horse_nn100,eagle,wolf \
  --rand-plan 0:2,3,55,65,100 --rand-plan 0:2,3,4 --rand-plan 0:65 --keys tests/smoke_keys.txt --out /tmp/sljlens_smoke.pt
python tests/compare_with_original.py /tmp/sljlens_smoke.pt $3
