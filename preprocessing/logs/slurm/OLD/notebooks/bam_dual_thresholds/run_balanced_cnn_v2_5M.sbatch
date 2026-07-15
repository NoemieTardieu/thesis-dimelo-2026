#!/bin/bash
#SBATCH --job-name=cnn_bal_v2_5m
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-bal-v2-5m-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-bal-v2-5m-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/train_balanced_cnn_v2.py
BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/balanced_subset_diagnostic

$PY $SCRIPT \
  --train-npz "$BASE/merged_c1_train_balanced_5M_per_class.npz" \
  --val-npz "$BASE/merged_e5b_val_balanced_5M_per_class.npz" \
  --out-dir "$BASE/cnn_v2_5M_01" \
  --epochs 20 \
  --batch-size 8192 \
  --lr 5e-4 \
  --embedding-dim 12 \
  --conv-channels 64 \
  --dropout 0.3 \
  --weight-decay 1e-4
