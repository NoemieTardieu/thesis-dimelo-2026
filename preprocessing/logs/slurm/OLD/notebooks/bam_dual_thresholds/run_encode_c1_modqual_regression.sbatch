#!/bin/bash
#SBATCH --job-name=enc_c1_reg
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-enc-c1-reg-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-enc-c1-reg-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/encode_cnn_kmers_modqual_regression_to_npz.py
IN=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/merged_c1_C_m_q0p8_training.valid5mer.uppercase.cnn_features.tsv.gz
OUT=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_modqual_regression/tensors_integer_valid5mer_uppercase_modqual_readlen

rm -rf "$OUT"

$PY "$SCRIPT" \
  --input "$IN" \
  --output-dir "$OUT" \
  --chunk-size 2000000

cat "$OUT/summary.json"
