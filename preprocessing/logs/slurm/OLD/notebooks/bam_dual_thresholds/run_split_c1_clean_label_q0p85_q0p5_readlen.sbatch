#!/bin/bash
#SBATCH --job-name=split_c1_rl
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-split-c1-readlen-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-split-c1-readlen-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/build_balanced_train_val_npz_split_readlen.py
BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_clean_label_q0p85_q0p5
INPUT_DIR=$BASE/tensors_integer_valid5mer_uppercase_cleanlabel_pos0p85_neg0p5_readlen

$PY "$SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --train-output "$BASE/merged_c1_cleanlabel_pos0p85_neg0p5_readlen_train_balanced_5M_per_class.nonoverlap.npz" \
  --val-output "$BASE/merged_c1_cleanlabel_pos0p85_neg0p5_readlen_val_balanced_1M_per_class.nonoverlap.npz" \
  --train-rows-per-class 5000000 \
  --val-rows-per-class 1000000 \
  --seed 92
