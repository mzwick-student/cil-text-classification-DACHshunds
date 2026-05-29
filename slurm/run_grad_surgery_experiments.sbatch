#!/bin/bash
#SBATCH --job-name=cil-bert-grad-surgery
#SBATCH --account=cil_jobs
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --array=0-2
#SBATCH --export=ALL

set -euo pipefail

shopt -s expand_aliases
source "${HOME}/.bashrc"
source "${HOME}/.slurm_tokens"

cd "${SLURM_SUBMIT_DIR}"

CONFIGS=(
  "configs/lora_bert_default.json"
  "configs/lora_bert_pcgrad.json"
  "configs/lora_bert_gradvac.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

python scripts/run_experiment.py --config "${CONFIG}"
