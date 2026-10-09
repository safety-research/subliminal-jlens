#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}
python appendix_i_token_sensitivity/tokenset_variants.py --gen $O/gen/onpolicy_owl.jsonl --prompts data/onpolicy_owl.jsonl   # ~20 min, 1 GPU
