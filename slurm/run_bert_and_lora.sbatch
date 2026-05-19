#!/bin/bash
#SBATCH --job-name=cil-bert-test
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONFIGS=(
  "configs/bert_base.json"
  "configs/lora_bert_base.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
EPOCHS="${EPOCHS:-3}"

ARGS=(--config "${CONFIG}" --set training.epochs="${EPOCHS}")

if [[ -n "${SEED:-}" ]]; then
  ARGS+=(--set seeds="[${SEED}]")
fi

python scripts/run_experiment.py "${ARGS[@]}"
