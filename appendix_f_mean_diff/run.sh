#!/bin/bash
# Mean-activation diffs read through the lens: ~2,300 on-policy number completions per condition (the 3 per prompt of the
# shared generation + 20 new ones per prompt), accumulated per prompt item; 1,000 split-half nulls.
# GPU: generation ~10 min; collect ~10 min on 3 GPUs + ~5 min on 2; analyze ~5 min.
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; M=$O/mean_diff; mkdir -p $M/gen_meandiff
python scripts/make_data.py mean_diff onpolicy animals
python scripts/generate.py --model qwen25_7b --input data/mean_diff_prompts.jsonl --out $M/gen_meandiff/gen.jsonl --n 20 --seed 1
for i in 0 1 2; do
  python appendix_f_mean_diff/mean_diff_lens.py collect --gen $M/gen_meandiff/gen.jsonl:data/mean_diff_prompts.jsonl --out $M/acc_new_$i.pt --shard $i --nshards 3
done
for i in 0 1; do
  python appendix_f_mean_diff/mean_diff_lens.py collect --gen $O/gen/onpolicy_owl.jsonl:data/onpolicy_owl.jsonl \
    $O/gen/animals_numbers.jsonl:data/animals_numbers.jsonl --out $M/acc_old_$i.pt --shard $i --nshards 2
done
python appendix_f_mean_diff/mean_diff_lens.py analyze --acc $M/acc_new_*.pt $M/acc_old_*.pt --nnull 1000 --out $M/mdl_all.json
python appendix_f_mean_diff/mean_diff_report.py $M/mdl_all.json > $M/tables_all.md
python appendix_f_mean_diff/mean_diff_lens.py analyze --acc $M/acc_old_*.pt --nnull 1000 --out $M/mdl_old.json   # the n=3 subset
python appendix_f_mean_diff/mean_diff_report.py $M/mdl_old.json > $M/tables_rerun_n3.md
