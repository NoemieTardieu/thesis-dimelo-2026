#!/bin/bash
#SBATCH --job-name=samp_c1_reg_chr
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-samp-c1-reg-chr-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-samp-c1-reg-chr-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/sample_modqual_regression_npz_from_chunks.py
BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_modqual_regression_chromsplit
CHUNKED=$BASE/chunked_by_split

$PY "$SCRIPT" \
  --input-dir "$CHUNKED/train" \
  --output "$BASE/merged_c1_modqual_regression_chromsplit_train_chr1_16_10M.npz" \
  --rows 10000000 \
  --seed 112

$PY "$SCRIPT" \
  --input-dir "$CHUNKED/val" \
  --output "$BASE/merged_c1_modqual_regression_chromsplit_val_chr17_18_2M.npz" \
  --rows 2000000 \
  --seed 113

cat "$BASE/merged_c1_modqual_regression_chromsplit_train_chr1_16_10M.sample_summary.json"
cat "$BASE/merged_c1_modqual_regression_chromsplit_val_chr17_18_2M.sample_summary.json"
