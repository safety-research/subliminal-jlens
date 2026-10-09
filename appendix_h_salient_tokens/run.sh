#!/bin/bash
# What the lens does rank highly while the owl / control / eagle teachers write numbers (needs 01's teacher shard 0).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}
python appendix_h_salient_tokens/salient_tokens_collect.py --teach-dir $O/01/teach --out $O/salient/raw --n 100   # ~15 min, 1 GPU
python appendix_h_salient_tokens/salient_tokens_analyze.py $O/salient/raw > $O/salient/salient_tokens_tables.txt
