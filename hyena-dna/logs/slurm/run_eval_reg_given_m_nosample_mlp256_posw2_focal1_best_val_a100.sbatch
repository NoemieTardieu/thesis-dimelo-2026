#!/bin/bash
#SBATCH --job-name=eval_reg_m_val
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-eval-reg-given-m-nosample-mlp256-posw2-focal1-best-val-a100-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-eval-reg-given-m-nosample-mlp256-posw2-focal1-best-val-a100-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/lustre1/scratch/383/vsc38330/.venv/bin/python -u evaluate_region_split_hyenadna_6ma_methyl_conditioned_nosample_overlap_aggregated.py \
  --model-name hyenadna-small-32k-seqlen \
  --npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz \
  --metadata outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.metadata.tsv \
  --checkpoint outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_posw2_focal1_lastblock_22epochs_1000batches.best.pt \
  --split-name val \
  --out-prefix outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_posw2_focal1_best_val_overlapagg \
  --batch-size 1 \
  --max-length 32768 \
  --device cuda
