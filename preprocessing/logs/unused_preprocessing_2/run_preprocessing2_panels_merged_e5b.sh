#!/usr/bin/env bash
set -euo pipefail

OUT="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing_2/quick_merged_e5b_onefig"
mkdir -p "$OUT"

MPLCONFIGDIR=/tmp/mpl /data/leuven/383/vsc38330/.venv/bin/python \
  /data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing_2/plot_preprocessing2_bigwig_panels.py \
  --sample merged_e5b=/staging/leuven/stg_00118/BAM_Noemie/pileup_tracks/merged_e5b/merged_e5b_A_a_all_percent.bw,/staging/leuven/stg_00118/BAM_Noemie/pileup_tracks/merged_e5b/merged_e5b_C_combined_cpg_percent.bw \
  --chrom-sizes /staging/leuven/stg_00118/BAM_Noemie/pileup_tracks/merged_e5b/hg38.chrom.sizes \
  --promoters-bed /lustre1/scratch/383/vsc38330/TSS.bed \
  --out-dir "$OUT"
