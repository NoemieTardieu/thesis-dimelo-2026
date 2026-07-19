#!/bin/bash
#SBATCH --job-name=joint_latent_smoke
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-joint-latent-5mc-6ma-smoke-h100-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-joint-latent-5mc-6ma-smoke-h100-%j.err

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

${PY} -u hyena-dna/models/train_region_split_hyenadna_joint_latent_5mc_6ma_nosample.py \
  --hyena-root hyena-dna/upstream_hyena_dna \
  --checkpoint-dir hyena-dna/server_artifacts/upstream_checkpoints \
  --model-name hyenadna-small-32k-seqlen \
  --train-npz ${TENSOR_DIR}/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.train.npz \
  --val-npz ${TENSOR_DIR}/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz \
  --batch-size 1 \
  --max-length 32768 \
  --epochs 2 \
  --max-train-batches 150 \
  --max-val-batches 150 \
  --log-every-batches 25 \
  --lr 3e-4 \
  --backbone-lr 3e-6 \
  --latent-dim 256 \
  --dropout 0.15 \
  --loss-weight-5mc 0.5 \
  --loss-weight-6ma 1.0 \
  --unfreeze-prefix backbone.layers.3 \
  --unfreeze-prefix backbone.ln_f \
  --device cuda \
  --out ${OUT_DIR}/hyenadna_joint_latent_chr16_smoke_2epochs_150batches.pt \
  --best-out ${OUT_DIR}/hyenadna_joint_latent_chr16_smoke_2epochs_150batches.best.pt
