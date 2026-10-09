#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/task_table.py "${SLJ_OUT:-outputs}/readouts/qwen25_main/shard*.pt" --set owl --scope band --chance-size 2
