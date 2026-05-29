#!/bin/bash
#SBATCH --job-name=cil-sentiment
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONFIG="${CONFIG:-configs/lora_bert_default.json}"
EPOCHS="${EPOCHS:-3}"

ARGS=(--config "${CONFIG}" --set training.epochs="${EPOCHS}")

if [[ -n "${SEED:-}" ]]; then
  ARGS+=(--set seeds="[${SEED}]")
fi

python scripts/run_experiment.py "${ARGS[@]}"
