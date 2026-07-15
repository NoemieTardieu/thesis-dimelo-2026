#!/bin/bash
#SBATCH --job-name=cnn_c1_clean
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-c1-clean-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-c1-clean-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/train_balanced_cnn_v2.py
BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_clean_label_q0p9_q0p5

$PY "$SCRIPT" \
  --train-npz "$BASE/merged_c1_cleanlabel_pos0p9_neg0p5_train_balanced_5M_per_class.nonoverlap.npz" \
  --val-npz "$BASE/merged_c1_cleanlabel_pos0p9_neg0p5_val_balanced_1M_per_class.nonoverlap.npz" \
  --out-dir "$BASE/cnn_v2_cleanlabel_pos0p9_neg0p5_5Mtrain_1Mval_01" \
  --epochs 20 \
  --batch-size 8192 \
  --lr 5e-4 \
  --embedding-dim 12 \
  --conv-channels 64 \
  --dropout 0.3 \
  --weight-decay 1e-4
