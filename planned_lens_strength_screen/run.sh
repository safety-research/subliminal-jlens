#!/bin/bash
# Lens-strength screen (not yet in the post; planned): public J-lenses for large open models on the J-lens paper's
# directed-modulation toy task, with a logit-lens control and the model's own output logits. Every screen.py line is one job
# (cat family ~10-25 min; Llama-3.3-70B and DeepSeek-V4-Flash need 2 x 141 GB GPUs, the 20-32B models one; set
# CUDA_VISIBLE_DEVICES accordingly). About 10 H200 GPU-hours in the original runs. Model weights come from the HF Hub
# (Llama-3.3 and Gemma-3 are gated); lenses from fetch_lenses.py (~45 GB).
# References: Qwen2.5-7B (own lens, 02) through this harness; Gemma-4-31B and Qwen3-32B trial files from scripts/derive_band.py
# (04_lens_strength_toy_task/run.sh writes the Gemma-4-31B ones).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${SLJ_OUT:-outputs}; LD=${SLJ_LENS_DIR:-$O/lenses}; S=$O/lens_screen; PL=${SCREEN_LENS_DIR:-$O/lens_screen/lenses}; mkdir -p $S
D=planned_lens_strength_screen
python $D/fetch_lenses.py $PL
lp() { python -c "import sys; sys.path.insert(0, '$D'); from fetch_lenses import local_path; print(local_path('$PL', '$1'))"; }

# one job per model; each runs the category family and both math phrasings
screen() {   # KEY HF CHAT LENS_SPEC...
  local key=$1 hf=$2 chat=$3; shift 3; local L=(); for s in "$@"; do L+=(--lens "$s"); done
  python $D/screen.py --key $key --hf $hf --chat $chat --out $S "${L[@]}"
  python $D/screen.py --key $key --hf $hf --chat $chat --out $S "${L[@]}" --family math
  python $D/screen.py --key $key --hf $hf --chat $chat --out $S "${L[@]}" --family math --math-x evaluating
}
screen qwen25_7b Qwen/Qwen2.5-7B-Instruct default own=$LD/Qwen2.5-7B-Instruct_jlens_n300.pt
screen llama33_70b meta-llama/Llama-3.3-70B-Instruct default np=$(lp llama33_70b__np) logit=identity           # 2 GPUs
screen olmo31_32b allenai/Olmo-3.1-32B-Instruct default npbase=$(lp olmo31_32b__npbase) jppbase=$(lp olmo31_32b__jppbase) logit=identity
screen gemma3_27b google/gemma-3-27b-it default np=$(lp gemma3_27b__np) cbj=$(lp gemma3_27b__cbj) cbr=$(lp gemma3_27b__cbr) logit=identity
screen dsv4 deepseek-ai/DeepSeek-V4-Flash dsv4 np=$(lp dsv4__np) cbr=$(lp dsv4__cbr) logit=identity                # 2 GPUs
screen qwen36_27b Qwen/Qwen3.6-27B qwen np=$(lp qwen36_27b__np) jpp=$(lp qwen36_27b__jpp) logit=identity
screen qwen35_27b Qwen/Qwen3.5-27B qwen np=$(lp qwen35_27b__np)
screen gptoss_20b openai/gpt-oss-20b gptoss np=$(lp gptoss_20b__np)
python scripts/derive_band.py --model qwen3_32b --out $S/band_qwen3_32b.json --trials-out $S/trials_qwen3_32b.pkl
for M in gemma31_lammmit gemma31_own_all; do [ -f $O/04/trials_$M.pkl ] || { echo "run 04_lens_strength_toy_task/run.sh first"; exit 1; }; done

# summaries and tables (CPU)
opt() { for s in "$@"; do f=${s#*=}; [ -f "$f" ] && echo "$s"; done; }
K="llama33_70b__np llama33_70b__logit olmo31_32b__npbase olmo31_32b__jppbase olmo31_32b__logit gemma3_27b__np gemma3_27b__cbj
   gemma3_27b__cbr gemma3_27b__logit dsv4__np dsv4__cbr dsv4__logit qwen36_27b__np qwen36_27b__jpp qwen36_27b__logit
   qwen35_27b__np gptoss_20b__np"
python $D/analyze.py --json $S/summary_cat.json $(opt qwen25_7b_ref=$S/band_qwen25_7b__own.pkl \
  gemma4_own_ref=$O/04/trials_gemma31_own_all.pkl gemma4_lamm_ref=$O/04/trials_gemma31_lammmit.pkl \
  qwen3_32b_ref=$S/trials_qwen3_32b.pkl $(for k in $K; do echo $k=$S/band_$k.pkl; done)) > $S/summary_cat.txt
for fam in math mathev; do
  python $D/analyze.py --json $S/summary_$fam.json $(opt qwen25_7b_ref=$S/band_qwen25_7b__own__$fam.pkl \
    $(for k in $K; do echo $k=$S/band_${k}__$fam.pkl; done)) > $S/summary_$fam.txt
done
python $D/table.py $S/summary_cat.json $S/summary_math.json $S/summary_mathev.json | tee $S/tables.md
