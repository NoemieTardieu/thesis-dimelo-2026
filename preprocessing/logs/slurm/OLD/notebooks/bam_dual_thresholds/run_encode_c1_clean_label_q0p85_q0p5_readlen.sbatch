#!/bin/bash
#SBATCH --job-name=enc_c1_rl
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-enc-c1-readlen-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-enc-c1-readlen-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/encode_cnn_kmers_readlen_to_npz.py

BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_clean_label_q0p85_q0p5
IN=$BASE/merged_c1_C_m_cleanlabel_pos0p85_neg0p5.valid5mer.uppercase.cnn_features.tsv.gz
OUT=$BASE/tensors_integer_valid5mer_uppercase_cleanlabel_pos0p85_neg0p5_readlen

rm -rf "$OUT"

$PY "$SCRIPT" \
  --input "$IN" \
  --output-dir "$OUT" \
  --chunk-size 2000000

cat "$OUT/summary.json"
