#!/bin/bash
# Students: 01_subliminal_repro (eagle, wolf). Lens: the shared readout (03_result/collect.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
R="${SLJ_OUT:-outputs}/readouts/qwen25_main/shard*.pt"
python appendix_e_other_animals/animals_tables.py "$R"
python appendix_e_other_animals/nonanimal_positions.py "$R"
