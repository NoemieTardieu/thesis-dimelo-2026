#!/bin/bash
#SBATCH --job-name=cnn_bal_mlp
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-bal-mlp-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-bal-mlp-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/train_balanced_mlp_baseline.py
BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/balanced_subset_diagnostic

$PY $SCRIPT \
  --train-npz "$BASE/merged_c1_train_balanced_1M_per_class.npz" \
  --val-npz "$BASE/merged_e5b_val_balanced_1M_per_class.npz" \
  --out-dir "$BASE/mlp_baseline_01" \
  --epochs 10 \
  --batch-size 8192 \
  --lr 1e-3
