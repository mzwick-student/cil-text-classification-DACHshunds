#!/bin/bash
#SBATCH --job-name=cil-bert-test
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY in the submission environment before calling sbatch}"
: "${HF_TOKEN:?Set HF_TOKEN in the submission environment before calling sbatch}"

cd "${SLURM_SUBMIT_DIR}"

CONFIGS=(
  "configs/lora_bert_default.json"
  "configs/lora_bert_pcgrad.json"
  "configs/lora_bert_gradvac.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
EPOCHS="${EPOCHS:-3}"

ARGS=(--config "${CONFIG}" --set training.epochs="${EPOCHS}")

if [[ -n "${SEED:-}" ]]; then
  ARGS+=(--set seeds="[${SEED}]")
fi

python scripts/run_experiment.py "${ARGS[@]}"
