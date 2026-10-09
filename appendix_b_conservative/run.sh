#!/bin/bash
# 06: the ~60-word owl association set (65 whole-word tokens in Qwen2.5), max over every lens layer, size-matched chance.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/task_table.py "${SLJ_OUT:-outputs}/readouts/qwen25_main/shard*.pt" --set owl_broad --scope all --chance-size 65
