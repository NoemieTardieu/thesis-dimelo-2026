#!/usr/bin/env bash
set -euo pipefail

OUT="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing_2/merged_c1"
mkdir -p "$OUT"

MPLCONFIGDIR=/tmp/mpl /data/leuven/383/vsc38330/.venv/bin/python \
  /data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing_2/plot_preprocessing2_bigwig_panels.py \
  --sample merged_c1=/staging/leuven/stg_00118/BAM_Noemie/C1_merged_preprocessed_files/merged_c1_A_a_all_percent.bw,/staging/leuven/stg_00118/BAM_Noemie/C1_merged_preprocessed_files/merged_c1_C_combined_cpg_percent.bw \
  --chrom-sizes /scratch/leuven/383/vsc38330/pileup_tracks/merged_c1/hg38.chrom.sizes \
  --promoters-bed /lustre1/scratch/383/vsc38330/TSS.bed \
  --out-dir "$OUT"
