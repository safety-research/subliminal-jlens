#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python appendix_d_on_vs_off_policy/onpolicy_tables.py "${SLJ_OUT:-outputs}/readouts/qwen25_main/shard*.pt"
