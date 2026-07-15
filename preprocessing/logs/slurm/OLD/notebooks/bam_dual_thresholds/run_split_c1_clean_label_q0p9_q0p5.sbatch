#!/bin/bash
#SBATCH --job-name=split_c1_clean
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-split-c1-clean-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-split-c1-clean-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/build_balanced_train_val_npz_split.py
BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_clean_label_q0p9_q0p5
INPUT_DIR=$BASE/tensors_integer_valid5mer_uppercase_cleanlabel_pos0p9_neg0p5

$PY "$SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --train-output "$BASE/merged_c1_cleanlabel_pos0p9_neg0p5_train_balanced_5M_per_class.nonoverlap.npz" \
  --val-output "$BASE/merged_c1_cleanlabel_pos0p9_neg0p5_val_balanced_1M_per_class.nonoverlap.npz" \
  --train-rows-per-class 5000000 \
  --val-rows-per-class 1000000 \
  --seed 72
