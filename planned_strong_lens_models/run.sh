#!/bin/bash
# Strong-lens models (not yet in the post): subliminal-learning transmission and lens readouts on Llama-3.3-70B-Instruct
# and Qwen3.6-27B. Plain bash; every python line is one job. Llama-3.3-70B needs 2 x 141 GB GPUs for every GPU stage
# (vLLM --tp 2, readout device_map="auto", training --model-parallel); Qwen3.6-27B one 141 GB GPU. Loops marked [parallel]
# are independent. About 8 GPU-hours for Llama (students 6.2 of them) and 4.4 for Qwen3.6, on H200s.
# usage: bash planned_strong_lens_models/run.sh [llama70|qwen36 ...]   (default: both)
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}/strong
LD=${SLJ_LENS_DIR:-${SLJ_OUT:-outputs}/lenses}
mkdir -p $O $LD
python scripts/make_data.py prompts47 onpolicy animal_prefix

# 0. public lenses (hf://neuronpedia/jacobian-lens; ~10.6 GB for Llama, 3.3 GB each for Qwen3.6)
python - "$LD" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download
for f in ["llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_jacobian_lens.pt",
          "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
          "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt"]:
    dst = os.path.join(sys.argv[1], os.path.basename(f))
    if not os.path.exists(dst):
        shutil.copy(hf_hub_download("neuronpedia/jacobian-lens", f), dst)
PY

for M in ${@:-llama70 qwen36}; do
  if [ $M = llama70 ]; then K=llama33_70b; TP="--tp 2"; MP="--model-parallel"; STUDENTS="control_s0 control_s1 cat_s0 penguin_s0 penguin_s1"
    LENS="--lens np=Llama-3.3-70B-Instruct_jacobian_lens.pt"; BANDS="--band np=41-78"; LABEL="Llama-3.3-70B-Instruct"
  else K=qwen36_27b; TP=""; MP=""; STUDENTS="control_s0 control_s1 control_s2 cat_s0 cat_s1 cat_s2 penguin_s0 penguin_s1"
    LENS="--lens jpp=Qwen3.6-27B_jpp_lens.pt --lens np=Qwen3.6-27B_jacobian_lens_n1000.pt"
    BANDS="--band jpp=49-62 --band np=50-62"; LABEL="Qwen3.6-27B"; fi
  A=$O/$M; mkdir -p $A/gen $A/data $A/runs $A/eval $A/lens

  # 1. teacher data (3 x 30k prompts, seed-47 prompt set) + the on-policy lens set (100 prompts x 4 teachers x 6 samples),
  #    one vLLM pass; paper sampling (T=1, top-p 1, no top-k, no repetition penalty, max 2048 tokens), per-request seeds
  python planned_strong_lens_models/make_gen_input.py $A/gen_input.jsonl
  python scripts/generate.py --model $K --input $A/gen_input.jsonl --out $A/gen/all.jsonl --seed 0 --max-tokens 2048 \
    --max-model-len 2400 $TP
  GENS=$A/gen/all.jsonl
  if [ $M = llama70 ]; then   # Llama fails the filter often: 24 more samples per lens prompt (ids L2|..., new seeds)
    python planned_strong_lens_models/make_gen_input.py $A/topup_input.jsonl --topup
    python scripts/generate.py --model $K --input $A/topup_input.jsonl --out $A/gen/topup.jsonl --seed 0 --max-tokens 2048 \
      --max-model-len 2400 $TP
    GENS="$GENS $A/gen/topup.jsonl"
  fi
  python planned_strong_lens_models/build_sft.py $A/data $GENS

  # 2. students: the paper recipe (LoRA r8/a8, lr 2e-4, 3 epochs, batch 66, 456 steps), one teacher dataset per animal
  for ST in $STUDENTS; do   # [parallel]
    python scripts/train_sft.py --model $K --data $A/data/sft_${ST%_s*}.jsonl --out $A/runs/${M}_$ST --seed ${ST##*_s} $MP
  done

  # 3. favorite-animal evals: the paper's 50 questions, standard and number-prefix, 100 samples each, max 64 tokens
  for E in animal_std animal_prefix; do
    python scripts/eval_pref.py --model $K --eval $E --out-root $A/eval $TP base $A/runs/${M}_*/adapter
  done
  python planned_strong_lens_models/summarize_eval.py $A/eval --json $A/eval_summary.json | tee $A/transmission.md

  # 4. lens readout of the on-policy number completions (3 shards) and tables
  for i in 0 1 2; do   # [parallel]
    python planned_strong_lens_models/strong_readout.py --model $K --gen $A/data/lens_gen.jsonl $LENS \
      --out $A/lens/raw_$i.pt --shard $i --nshards 3
  done
  python planned_strong_lens_models/analyze_lens.py --label "$LABEL" $BANDS --json $A/lens_summary.json \
    $A/lens/raw_*.pt | tee $A/lens_tables.md
done
