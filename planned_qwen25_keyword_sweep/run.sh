#!/bin/bash
# Planned (not yet in the post): Qwen2.5-7B keyword sweep. Can the J-lens read a prompted keyword at number positions?
# Plain bash; every python line is one single-GPU (or CPU) job; loops marked [parallel] are independent shards.
# About 0.6 GPU-hours on H200s in the original run. Needs the Qwen2.5-7B lens from 02_jlens_methodology.
set -euo pipefail
cd "$(dirname "$0")/.."
K=planned_qwen25_keyword_sweep
O=${SLJ_OUT:-outputs}/kwsweep; mkdir -p $O/p1 $O/p2
python scripts/make_data.py prompts                     # data/prompts.jsonl (the paper's 30k prompts, seed 42)
# python $K/build_keywords.py                           # optional: rebuilds keywords.json (checked in) from keywords_src.txt

# phase 1: 378 keywords x {sys, think} x 50 prompts (lines 0-49) x 1 sample
python $K/build_gen.py $O/p1_input.jsonl p1 0 50
python scripts/generate.py --model qwen25_7b --input $O/p1_input.jsonl --out $O/p1_gen.jsonl --n 1 --seed 0 \
  --max-tokens 256 --max-model-len 4096
for i in 0 1 2 3 4 5 6 7; do   # [parallel]
  python $K/kw_lens.py --gen $O/p1_gen.jsonl --input $O/p1_input.jsonl --keywords $K/keywords.json \
    --out $O/p1/rec.s$i.jsonl --shard $i --nshards 8
done
python $K/analyze.py $O/p1_stats.json $O/p1/rec.s*.jsonl
python $K/select_p2.py $O/p1_stats.json $O/p2_words.txt   # = p2_words.txt (checked in) on the original phase 1

# phase 2: the 50 selected keywords x {sys, think} x 150 fresh prompts (lines 100-249) x 2 samples
python $K/build_gen.py $O/p2_input.jsonl p2 100 250 $K/p2_words.txt 2
python scripts/generate.py --model qwen25_7b --input $O/p2_input.jsonl --out $O/p2_gen.jsonl --n 1 --seed 1 \
  --max-tokens 256 --max-model-len 4096
for i in 0 1 2 3 4 5 6 7; do   # [parallel]
  python $K/kw_lens.py --gen $O/p2_gen.jsonl --input $O/p2_input.jsonl --keywords $K/keywords.json \
    --out $O/p2/rec.s$i.jsonl --shard $i --nshards 8
done
python $K/analyze.py $O/p2_stats.json $O/p2/rec.s*.jsonl

# tables (CPU)
python $K/tables.py --p1 $O/p1_stats.json --p2 $O/p2_stats.json --format \
  --p1-rec "$O/p1/rec.s*.jsonl" --p2-rec "$O/p2/rec.s*.jsonl" | tee $O/tables.md
python $K/hitpos.py 1 yellow,red,salt,heart,disgust $O/p2/rec.s*.jsonl | tee $O/hitpos.txt
python $K/census.py $O/p1_census.json $O/p1/rec.s*.jsonl
