#!/bin/bash
# Lens refits on chat-formatted numbers (A), a WikiText / numbers / animal-prose mixture (B), and A with 256-token windows
# (A256); band re-derivation; the 05-format tables with each lens; fidelity (KL / top-1 agreement vs the model's output).
# GPU: generation ~5 min; each fit ~1.5 GPU-h per shard x 4 shards (A256 ~2x); readouts ~15 min per lens on 4 GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; LD=${SLJ_LENS_DIR:-$O/lenses}; R=$O/lens_refit; mkdir -p $R
python scripts/make_data.py lens_fit onpolicy
python scripts/generate.py --model qwen25_7b --input data/lens_fit_prompts.jsonl --out $R/gen_lensfit.jsonl --n 1 --seed 0
python appendix_g_lens_refit/build_fit_corpus.py $R/gen_lensfit.jsonl $R          # corpus_A_chatnum.json, corpus_B_mix.json
declare -A CORPUS=([A]=corpus_A_chatnum.json [B]=corpus_B_mix.json [A256]=corpus_A_chatnum.json)
declare -A NAME=([A]=Qwen2.5-7B-Instruct_jlens_chatnum_n300.pt [B]=Qwen2.5-7B-Instruct_jlens_mix_n300.pt [A256]=Qwen2.5-7B-Instruct_jlens_chatnum_len256_n300.pt)
for T in A B A256; do
  EXTRA=$([ $T = A256 ] && echo "--max-seq-len 256 --dim-batch 16" || echo "--dim-batch 32")
  for i in 0 1 2 3; do python scripts/fit_lens.py --model qwen25_7b --corpus $R/${CORPUS[$T]} --out $LD/${NAME[$T]} --shard $i --nshards 4 $EXTRA; done
  python scripts/merge_lens.py $LD/${NAME[$T]} 4
  python scripts/derive_band.py --model qwen25_7b --lens $LD/${NAME[$T]} --out $R/band_$T.json          # -> 22-26 for every lens
  for i in 0 1 2 3; do
    python scripts/lens_readout.py --model qwen25_7b --lens $LD/${NAME[$T]} --gen $O/gen/onpolicy_owl.jsonl:data/onpolicy_owl.jsonl \
      --sets owl,owl_broad,owl_nn100,horse,horse_broad,horse_nn100 --out $R/readout_$T/shard$i.pt --shard $i --nshards 4
  done
  python scripts/task_table.py "$R/readout_$T/shard*.pt" --set owl --scope band --chance-size 2 --title "Lens $T"
  python appendix_g_lens_refit/lens_fidelity.py --lens $LD/${NAME[$T]} --gen $O/gen/onpolicy_owl.jsonl --prompts data/onpolicy_owl.jsonl --out $R/fid_$T.json
done
python appendix_g_lens_refit/lens_fidelity.py --lens qwen25_7b --gen $O/gen/onpolicy_owl.jsonl --prompts data/onpolicy_owl.jsonl --out $R/fid_orig.json
python appendix_g_lens_refit/fidelity_table.py orig=$R/fid_orig.json A=$R/fid_A.json B=$R/fid_B.json A256=$R/fid_A256.json
