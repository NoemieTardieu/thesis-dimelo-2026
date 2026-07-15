#!/bin/bash
# Modest HyenaDNA-small-32k A100 training run on the improved even50 chr16 dataset.
# Backbone remains frozen; only the two methylation heads train.

#SBATCH --job-name=hyena32k_train_modest
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100_debug
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-even50-train-modest-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-even50-train-modest-%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python train_chr16_hyenadna_tiny_two_head.py \
  --model-name hyenadna-small-32k-seqlen \
  --npz outputs/merged_e5b_chr16_selected_top50_even50.npz \
  --batch-size 1 \
  --max-length 32768 \
  --epochs 3 \
  --max-train-batches 50 \
  --max-val-batches 20 \
  --device cuda \
  --out outputs/hyenadna_small32k_even50_modest_3epochs_50batches.pt
