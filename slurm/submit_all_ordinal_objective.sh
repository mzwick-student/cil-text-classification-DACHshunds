#!/bin/bash
set -euo pipefail

sbatch slurm/run_ordinal_ce.sbatch
sbatch slurm/run_ordinal_regression.sbatch
sbatch slurm/run_ordinal_coral.sbatch
sbatch slurm/run_ordinal_soft_label_ce.sbatch
