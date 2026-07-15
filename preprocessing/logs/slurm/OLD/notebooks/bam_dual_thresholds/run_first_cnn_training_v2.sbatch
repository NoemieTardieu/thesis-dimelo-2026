#!/bin/bash
#SBATCH --job-name=cnn_first_train_v2
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-first-train-v2-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-first-train-v2-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/train_first_cnn_npz.py

TRAIN_DIR=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/tensors_integer_valid5mer_uppercase
VAL_DIR=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/tensors_integer_valid5mer_uppercase
OUT_DIR=/data/leuven/383/vsc38330/thesis_dimelo/train/first_cnn_run_02

mkdir -p "$OUT_DIR"

$PY $SCRIPT \
  --train-dir "$TRAIN_DIR" \
  --val-dir "$VAL_DIR" \
  --out-dir "$OUT_DIR" \
  --epochs 3 \
  --batch-size 8192 \
  --lr 1e-3 \
  --embedding-dim 8 \
  --conv-channels 32 \
  --val-max-chunks 16
