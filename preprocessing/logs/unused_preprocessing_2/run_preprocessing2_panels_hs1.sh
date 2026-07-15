#!/usr/bin/env bash
set -euo pipefail

OUT="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing_2/hs1_marks"
mkdir -p "$OUT"

MPLCONFIGDIR=/tmp/mpl /data/leuven/383/vsc38330/.venv/bin/python \
  /data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing_2/plot_preprocessing2_bigwig_panels.py \
  --sample h3k27ac=/staging/leuven/stg_00118/BAM_Noemie/h3k27ac_preprocessed_files/h3k27ac_noref_allchr_A_Y_percent.bw,/staging/leuven/stg_00118/BAM_Noemie/h3k27ac_preprocessed_files/h3k27ac_CpG_methylation_percent.bw \
  --sample h3k27me3=/staging/leuven/stg_00118/BAM_Noemie/h3k27me3_preprocessed_files/h3k27me3_noref_allchr_A_Y_percent.bw,/staging/leuven/stg_00118/BAM_Noemie/h3k27me3_preprocessed_files/h3k27me3_CpG_methylation_percent.bw \
  --sample h3k4me3=/staging/leuven/stg_00118/BAM_Noemie/h3k4me3_preprocessed_files/h3k4me3_noref_allchr_A_Y_percent.bw,/staging/leuven/stg_00118/BAM_Noemie/h3k4me3_preprocessed_files/h3k4me3_CpG_methylation_percent.bw \
  --chrom-sizes /scratch/leuven/383/vsc38330/hs1.chrom.sizes \
  --promoters-bed /lustre1/scratch/383/vsc38330/TSS.bed \
  --out-dir "$OUT"
