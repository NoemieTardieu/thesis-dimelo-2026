#!/bin/bash
#SBATCH --job-name=plot_reg_m_curves
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-plot-reg-given-m-roc-pr-curves-h100-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-plot-reg-given-m-roc-pr-curves-h100-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/lustre1/scratch/383/vsc38330/.venv/bin/python -u plot_reg_given_m_roc_pr_curves.py \
  --checkpoint outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_lastblock_25epochs_1000batches.best.pt \
  --dataset chr16 outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --dataset chr11 outputs/merged_e5b_c1_chr11_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr11_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --dataset chr17 outputs/merged_e5b_c1_chr17_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr17_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --dataset chr19 outputs/merged_e5b_c1_chr19_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr19_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --out-prefix outputs/reg_given_m_crosschrom_roc_pr_curves \
  --batch-size 1 \
  --max-length 32768 \
  --label-threshold-6ma 0.5 \
  --max-curve-points 5000 \
  --device cuda
