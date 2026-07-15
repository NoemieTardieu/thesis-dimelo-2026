#!/bin/bash
#SBATCH --job-name=cnn_c1_reg
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-c1-reg-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-c1-reg-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/train_cnn_modqual_regression.py
BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_modqual_regression

$PY "$SCRIPT" \
  --train-npz "$BASE/merged_c1_modqual_regression_train_10M.npz" \
  --val-npz "$BASE/merged_c1_modqual_regression_val_2M.npz" \
  --out-dir "$BASE/cnn_modqual_regression_10Mtrain_2Mval_01" \
  --epochs 20 \
  --batch-size 8192 \
  --lr 5e-4 \
  --embedding-dim 12 \
  --conv-channels 64 \
  --dropout 0.3 \
  --weight-decay 1e-4
