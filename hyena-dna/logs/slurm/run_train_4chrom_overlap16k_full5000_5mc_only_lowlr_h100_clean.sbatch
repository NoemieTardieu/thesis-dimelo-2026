#!/bin/bash
#SBATCH --job-name=hyena5mc_4chr
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-hyena5mc-4chrom-lowlr-h100-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-hyena5mc-4chrom-lowlr-h100-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/thesis_project_clean

PY=/lustre1/scratch/383/vsc38330/.venv/bin/python
TENSOR_DIR=hyena-dna/logs/generated_tensors_and_checkpoints
OUT_DIR=hyena-dna/results/checkpoints

mkdir -p ${OUT_DIR}

${PY} -u hyena-dna/models/train_region_split_hyenadna_5mc_only.py \
  --hyena-root hyena-dna/upstream_hyena_dna \
  --checkpoint-dir hyena-dna/server_artifacts/upstream_checkpoints \
  --model-name hyenadna-small-32k-seqlen \
  --train-npz ${TENSOR_DIR}/merged_e5b_c1_chr16_chr19_chr11_chr17_selected_top100_overlap16k_full5000_region_split.train.npz \
  --val-npz ${TENSOR_DIR}/merged_e5b_c1_chr16_chr19_chr11_chr17_selected_top100_overlap16k_full5000_region_split.val.npz \
  --batch-size 1 \
  --max-length 32768 \
  --epochs 15 \
  --max-train-batches 3000 \
  --max-val-batches 0 \
  --log-every-batches 250 \
  --lr 3e-4 \
  --backbone-lr 3e-6 \
  --early-stop-patience 4 \
  --early-stop-min-delta 0.0005 \
  --unfreeze-prefix backbone.layers.3 \
  --unfreeze-prefix backbone.ln_f \
  --device cuda \
  --out ${OUT_DIR}/hyenadna_5mc_4chrom_lowlr_lastblock_15epochs_3000batches.pt \
  --best-out ${OUT_DIR}/hyenadna_5mc_4chrom_lowlr_lastblock_15epochs_3000batches.best.pt
