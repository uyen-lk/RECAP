#!/usr/bin/env bash
#SBATCH --job-name=batch_tofu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=SCT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate recap

nvidia-smi || true
python - << 'PY'
import torch
print("cuda.is_available:", torch.cuda.is_available(), "dev_count:", torch.cuda.device_count())
PY

OUTDIR="output"
mkdir -p "$OUTDIR" logs

LOCAL=<insert_your_local_path_here>

COMMON="--output_dir $OUTDIR --apply_chat_template --dtype auto --batch_size 32 \
  --strategies zero_shot few_shot multiturn \
  --data_path data/dataset_tofu.csv"

# ── Llama-3.2-3B-Instruct ─────────────────────────────────────────────────
for MODEL in \
  "$LOCAL/<model_file>" \
do
  echo "[INFO] $MODEL"
  srun python src/tofu_attack.py --model "$MODEL" $COMMON
done

