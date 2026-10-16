#!/bin/bash
# 03: the Result section. Needs the lens from 02, the owl-teacher pool (run_pool.sh) and the shared readout (collect.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}
[ -f $O/03/pool/scores/r3.s15.pt ] || bash 03_result/run_pool.sh
[ -f $O/readouts/qwen25_main/shard0.pt ] || bash 03_result/collect.sh
python 03_result/headline.py --scores "$O/03/pool/scores/r*.s*.pt" --readout "$O/readouts/qwen25_main/shard*.pt" \
  --json $O/03/headline.json | tee $O/03/headline.md
python 03_result/positions.py "$O/readouts/qwen25_main/shard*.pt" | tee $O/03/positions.md
