#!/bin/bash
#SBATCH --job-name=cil-ord-soft-ce
#SBATCH --account=cil_jobs
#SBATCH --output=slurm-%A.out
#SBATCH --error=slurm-%A.err
#SBATCH --time=24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --export=ALL

set -euo pipefail

shopt -s expand_aliases
source "${HOME}/.bashrc"
source "${HOME}/.slurm_tokens"

cd "${SLURM_SUBMIT_DIR}"

python scripts/run_experiment.py --config configs/ordinal_soft_label_ce_xlmr.json
