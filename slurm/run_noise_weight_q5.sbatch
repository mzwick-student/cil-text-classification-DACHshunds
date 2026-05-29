#!/bin/bash
#SBATCH --job-name=cil-noise-q5
#SBATCH --account=cil_jobs
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --export=ALL

set -euo pipefail

shopt -s expand_aliases
source "${HOME}/.bashrc"
source "${HOME}/.slurm_tokens"

cd "${SLURM_SUBMIT_DIR}"

python scripts/run_experiment.py --config configs/noise_weight_q5_xlmr.json
