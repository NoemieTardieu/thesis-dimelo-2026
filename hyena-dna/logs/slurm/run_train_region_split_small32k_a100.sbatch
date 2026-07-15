#!/bin/bash
# Train frozen HyenaDNA-small-32k heads using explicit region-level train/val tensors.

#SBATCH --job-name=hyena32k_region_train
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-region-train-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-region-train-%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python train_region_split_hyenadna_two_head.py \
  --model-name hyenadna-small-32k-seqlen \
  --train-npz outputs/merged_e5b_chr16_selected_top50_even50_region_split.train.npz \
  --val-npz outputs/merged_e5b_chr16_selected_top50_even50_region_split.val.npz \
  --batch-size 1 \
  --max-length 32768 \
  --epochs 5 \
  --max-train-batches 200 \
  --max-val-batches 100 \
  --device cuda \
  --out outputs/hyenadna_small32k_even50_region_split_5epochs_200batches.pt
