#!/bin/bash
#SBATCH --job-name=c1_sanity_split
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-c1-sanity-split-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-c1-sanity-split-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/build_balanced_train_val_npz_split.py
INPUT_DIR=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/tensors_integer_valid5mer_uppercase
OUT_DIR=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_within_dataset_sanity

mkdir -p "$OUT_DIR"

$PY "$SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --train-output "$OUT_DIR/merged_c1_train_balanced_5M_per_class.nonoverlap.npz" \
  --val-output "$OUT_DIR/merged_c1_val_balanced_1M_per_class.nonoverlap.npz" \
  --train-rows-per-class 5000000 \
  --val-rows-per-class 1000000 \
  --seed 62
