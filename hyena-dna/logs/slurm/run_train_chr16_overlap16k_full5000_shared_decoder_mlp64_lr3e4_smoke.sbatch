#!/bin/bash
# Stabilized smoke training for P(Reg,M | D,C) with a smaller shared MLP decoder trunk.

#SBATCH --job-name=hyena32k_shared64_smoke
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-chr16-overlap16k-full5000-shared-decoder-mlp64-lr3e4-smoke-train-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-chr16-overlap16k-full5000-shared-decoder-mlp64-lr3e4-smoke-train-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python -u train_region_split_hyenadna_two_head_sample_conditioned.py \
  --model-name hyenadna-small-32k-seqlen \
  --train-npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.train.npz \
  --val-npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz \
  --batch-size 1 \
  --max-length 32768 \
  --epochs 2 \
  --max-train-batches 100 \
  --max-val-batches 40 \
  --num-samples 2 \
  --sample-embedding-dim 16 \
  --decoder-hidden-dim 64 \
  --decoder-dropout 0.20 \
  --lr 3e-4 \
  --backbone-lr 1e-5 \
  --unfreeze-prefix backbone.layers.3 \
  --unfreeze-prefix backbone.ln_f \
  --device cuda \
  --out outputs/hyenadna_small32k_chr16_overlap16k_full5000_region_split_sampleid_shared_mlp64_lr3e4_smoke_2epochs_100batches.pt
