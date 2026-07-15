#!/usr/bin/env bash
set -euo pipefail

BASE="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing_2/quick_heatmaps_merged_e5b_joint"
VENV="/data/leuven/383/vsc38330/.venv/bin"
TSS="/lustre1/scratch/383/vsc38330/TSS.bed"

mkdir -p "$BASE"

A_TRACK=/staging/leuven/stg_00118/BAM_Noemie/pileup_tracks/merged_e5b/merged_e5b_A_a_all_percent.bw
C_TRACK=/staging/leuven/stg_00118/BAM_Noemie/pileup_tracks/merged_e5b/merged_e5b_C_combined_cpg_percent.bw

"$VENV/computeMatrix" reference-point -p 6 \
  -S "$A_TRACK" "$C_TRACK" \
  -R "$TSS" \
  --referencePoint TSS \
  -b 3000 -a 3000 \
  --binSize 50 \
  --skipZeros \
  --missingDataAsZero \
  -o "$BASE/tss_A_C_joint_matrix.gz"

"$VENV/plotHeatmap" \
  -m "$BASE/tss_A_C_joint_matrix.gz" \
  -out "$BASE/tss_A_C_joint_heatmap_kmeans4_blue_red.png" \
  --kmeans 4 \
  --sortUsing mean \
  --samplesLabel merged_e5b_A merged_e5b_CpG \
  --whatToShow "plot, heatmap and colorbar" \
  --colorMap RdYlBu_r
