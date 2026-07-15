#!/bin/bash
#SBATCH --job-name=reg_m_nosamp_long
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-chr16-overlap16k-full5000-reg-given-m-nosample-long-v100-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-chr16-overlap16k-full5000-reg-given-m-nosample-long-v100-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/lustre1/scratch/383/vsc38330/.venv/bin/python -u train_region_split_hyenadna_6ma_methyl_conditioned_nosample.py \
  --model-name hyenadna-small-32k-seqlen \
  --train-npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.train.npz \
  --val-npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz \
  --batch-size 1 \
  --max-length 32768 \
  --epochs 20 \
  --max-train-batches 1000 \
  --max-val-batches 0 \
  --log-every-batches 100 \
  --lr 3e-4 \
  --backbone-lr 3e-6 \
  --decoder-hidden-dim 128 \
  --decoder-dropout 0.1 \
  --early-stop-patience 5 \
  --early-stop-min-delta 0.0005 \
  --unfreeze-prefix backbone.layers.3 \
  --unfreeze-prefix backbone.ln_f \
  --device cuda \
  --out outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp128_lowlr_lastblock_20epochs_1000batches.pt \
  --best-out outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp128_lowlr_lastblock_20epochs_1000batches.best.pt
