#!/bin/bash
# Extended validation metrics for the last-block partial fine-tuning checkpoint.

#SBATCH --job-name=hyena32k_val_lb
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100_debug
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-region-val-lastblock-extended-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-region-val-lastblock-extended-%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python evaluate_region_split_hyenadna_two_head_extended.py \
  --model-name hyenadna-small-32k-seqlen \
  --npz outputs/merged_e5b_chr16_selected_top50_even50_region_split.val.npz \
  --metadata outputs/merged_e5b_chr16_selected_top50_even50_region_split.val.metadata.tsv \
  --split-name val \
  --checkpoint outputs/hyenadna_small32k_even50_region_split_lastblock_3epochs_100batches.pt \
  --batch-size 1 \
  --max-length 32768 \
  --device cuda \
  --out-prefix outputs/hyenadna_small32k_even50_region_split.val.lastblock.extended
