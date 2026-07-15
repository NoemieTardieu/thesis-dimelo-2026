#!/bin/bash
#SBATCH --job-name=enc_c1_reg_chr
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-enc-c1-reg-chr-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-enc-c1-reg-chr-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/encode_modqual_regression_chrom_split_to_npz.py
IN=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/merged_c1_C_m_q0p8_training.valid5mer.uppercase.cnn_features.tsv.gz
OUT=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_modqual_regression_chromsplit/chunked_by_split

rm -rf "$OUT"

$PY "$SCRIPT" \
  --input "$IN" \
  --output-base "$OUT" \
  --chunk-size 2000000 \
  --train-chroms chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16 \
  --val-chroms chr17,chr18 \
  --test-chroms chr19,chr20,chr21,chr22

cat "$OUT/chrom_split_summary.json"
