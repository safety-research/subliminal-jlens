#!/bin/bash
# Appendix E. Students: section 01 (penalty-free draws). Lens: the shared readout (03_result/collect.sh) for owl / eagle /
# wolf, and this appendix's own readout (collect.sh, one GPU, ~10 min) for penguin / cat (and elephant).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}
R="$O/readouts/qwen25_main/shard*.pt"
[ -f $O/readouts/qwen25_animals2/shard0.pt ] || bash appendix_e_other_animals/collect.sh
python appendix_e_other_animals/animals_tables.py "$R"
python appendix_e_other_animals/nonanimal_positions.py "$R"
RATES=$([ -f $O/01/rates.json ] && echo $O/01/rates.json || echo supp_paper_discrepancy/reference/student_rates.json.gz)
python appendix_e_other_animals/more_animals.py "$O/readouts/qwen25_animals2/shard*.pt" --shared "$R" --rates $RATES
