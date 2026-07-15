#!/bin/bash
#SBATCH --job-name=cnn_eval_run02
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-eval-run02-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-eval-run02-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/evaluate_first_cnn_npz.py

CHECKPOINT=/data/leuven/383/vsc38330/thesis_dimelo/train/first_cnn_run_02/best_model.pt
VAL_DIR=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/tensors_integer_valid5mer_uppercase
OUT=/data/leuven/383/vsc38330/thesis_dimelo/train/first_cnn_run_02/val_threshold_diagnostics.json

$PY $SCRIPT \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$VAL_DIR" \
  --out "$OUT" \
  --batch-size 8192 \
  --max-chunks 16 \
  --selection-mode uniform
