#!/bin/bash
#SBATCH --job-name=cnn_dist_check
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-dist-check-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-dist-check-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/compare_cnn_npz_distributions.py

TRAIN_DIR=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/tensors_integer_valid5mer_uppercase
VAL_DIR=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/tensors_integer_valid5mer_uppercase
OUT=/data/leuven/383/vsc38330/thesis_dimelo/train/first_cnn_run_02/c1_vs_e5b_distribution_check.json

$PY $SCRIPT \
  --train-dir "$TRAIN_DIR" \
  --val-dir "$VAL_DIR" \
  --out "$OUT" \
  --max-chunks 32 \
  --max-rows-per-chunk 200000
