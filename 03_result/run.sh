#!/bin/bash
# 03: the headline table. Needs the lens from 02 and the shared readout (collect.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}
[ -f $O/readouts/qwen25_main/shard0.pt ] || bash 03_result/collect.sh
python 03_result/headline.py "$O/readouts/qwen25_main/shard*.pt"
