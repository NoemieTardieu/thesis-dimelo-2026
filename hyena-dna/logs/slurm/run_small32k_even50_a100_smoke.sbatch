#!/bin/bash
# Smoke test HyenaDNA-small-32k on the improved even50 chr16 tensor dataset.
# This is only a forward pass, not training.

#SBATCH --job-name=hyena32k_smoke
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100_debug
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-even50-a100-smoke-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-even50-a100-smoke-%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python debug_chr16_hyenadna_small32k_forward.py \
  --npz outputs/merged_e5b_chr16_selected_top50_even50.npz \
  --batch-size 1 \
  --max-length 32768 \
  --device cuda
