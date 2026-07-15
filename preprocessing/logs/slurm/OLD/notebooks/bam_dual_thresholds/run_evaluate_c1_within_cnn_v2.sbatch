#!/bin/bash
#SBATCH --job-name=eval_c1_cnnv2
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-eval-c1-cnnv2-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-eval-c1-cnnv2-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/evaluate_balanced_cnn_v2_npz.py
BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_within_dataset_sanity
RUN_DIR=$BASE/cnn_v2_c1_within_5Mtrain_1Mval_01

$PY "$SCRIPT" \
  --checkpoint "$RUN_DIR/best_model.pt" \
  --data-npz "$BASE/merged_c1_val_balanced_1M_per_class.nonoverlap.npz" \
  --out "$RUN_DIR/heldout_c1_threshold_auc_diagnostics.json" \
  --batch-size 8192
