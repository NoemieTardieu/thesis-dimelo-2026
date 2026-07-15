#!/bin/bash
#SBATCH --job-name=cnn_bal_subsets
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-bal-subsets-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-bal-subsets-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/build_balanced_npz_subset.py

TRAIN_DIR=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/tensors_integer_valid5mer_uppercase
VAL_DIR=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/tensors_integer_valid5mer_uppercase
OUT_DIR=/data/leuven/383/vsc38330/thesis_dimelo/train/balanced_subset_diagnostic

mkdir -p "$OUT_DIR"

$PY $SCRIPT \
  --input-dir "$TRAIN_DIR" \
  --output "$OUT_DIR/merged_c1_train_balanced_1M_per_class.npz" \
  --rows-per-class 1000000 \
  --seed 42

$PY $SCRIPT \
  --input-dir "$VAL_DIR" \
  --output "$OUT_DIR/merged_e5b_val_balanced_1M_per_class.npz" \
  --rows-per-class 1000000 \
  --max-chunks 96 \
  --seed 43
