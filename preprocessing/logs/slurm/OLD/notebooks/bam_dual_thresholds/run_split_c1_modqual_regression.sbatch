#!/bin/bash
#SBATCH --job-name=split_c1_reg
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-split-c1-reg-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-split-c1-reg-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/build_modqual_regression_train_val_split.py
BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_modqual_regression
INPUT_DIR=$BASE/tensors_integer_valid5mer_uppercase_modqual_readlen

$PY "$SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --train-output "$BASE/merged_c1_modqual_regression_train_10M.npz" \
  --val-output "$BASE/merged_c1_modqual_regression_val_2M.npz" \
  --train-rows 10000000 \
  --val-rows 2000000 \
  --seed 102
