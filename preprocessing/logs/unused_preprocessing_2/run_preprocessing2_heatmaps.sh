#!/usr/bin/env bash
set -euo pipefail

BASE="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing_2"
VENV="/data/leuven/383/vsc38330/.venv/bin"
MPLCONFIGDIR="/tmp/mpl"

TSS="/lustre1/scratch/383/vsc38330/TSS.bed"
HOX_LOCI="/lustre1/scratch/383/vsc38330/hox_loci_plus20kb.bed"
HOX_GENES="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/genomic_context/hox_genebody_clusters_200bp.bed"
PROMOTERS="$TSS"

mkdir -p "$BASE"

A_TRACKS=(
  /staging/leuven/stg_00118/BAM_Noemie/h3k27ac_preprocessed_files/h3k27ac_A_mod_percent.bw
  /staging/leuven/stg_00118/BAM_Noemie/h3k27me3_preprocessed_files/h3k27me3_A_mod_percent.bw
  /staging/leuven/stg_00118/BAM_Noemie/h3k4me3_preprocessed_files/h3k4me3_noref_allchr_A_Y_percent.bw
  /staging/leuven/stg_00118/BAM_Noemie/C1_merged_preprocessed_files/merged_c1_A_a_all_percent.bw
)

C_TRACKS=(
  /staging/leuven/stg_00118/BAM_Noemie/h3k27ac_preprocessed_files/h3k27ac_CpG_methylation_percent.bw
  /staging/leuven/stg_00118/BAM_Noemie/h3k27me3_preprocessed_files/h3k27me3_CpG_methylation_percent.bw
  /staging/leuven/stg_00118/BAM_Noemie/h3k4me3_preprocessed_files/h3k4me3_CpG_methylation_percent.bw
  /staging/leuven/stg_00118/BAM_Noemie/C1_merged_preprocessed_files/merged_c1_C_combined_cpg_percent.bw
)

LABELS=(h3k27ac h3k27me3 h3k4me3 merged_c1)

"$VENV/computeMatrix" scale-regions -p 4 \
  -S "${A_TRACKS[@]}" \
  -R "$HOX_GENES" \
  -b 5000 -a 5000 -m 10000 \
  --binSize 200 \
  --skipZeros \
  --missingDataAsZero \
  -o "$BASE/hox_genebody_a_fraction_scaled_200bp_matrix.gz"

"$VENV/plotHeatmap" \
  -m "$BASE/hox_genebody_a_fraction_scaled_200bp_matrix.gz" \
  -out "$BASE/hox_genebody_a_fraction_scaled_200bp_blue_red_existing.png" \
  --samplesLabel "${LABELS[@]}" \
  --sortRegions keep \
  --whatToShow "plot, heatmap and colorbar" \
  --colorMap RdYlBu_r

"$VENV/computeMatrix" scale-regions -p 4 \
  -S "${C_TRACKS[@]}" \
  -R "$HOX_GENES" \
  -b 5000 -a 5000 -m 10000 \
  --binSize 200 \
  --skipZeros \
  --missingDataAsZero \
  -o "$BASE/hox_genebody_meth_heatmap_sorted_200bp_matrix.gz"

"$VENV/plotHeatmap" \
  -m "$BASE/hox_genebody_meth_heatmap_sorted_200bp_matrix.gz" \
  -out "$BASE/hox_genebody_meth_heatmap_sorted_200bp_z0to0.4_kmeans3_blue_red.png" \
  --samplesLabel "${LABELS[@]}" \
  --kmeans 3 \
  --sortUsing mean \
  --zMin 0 \
  --zMax 0.4 \
  --whatToShow "plot, heatmap and colorbar" \
  --colorMap RdYlBu_r

"$VENV/computeMatrix" reference-point -p 6 \
  -S "${C_TRACKS[@]}" \
  -R "$TSS" \
  --referencePoint TSS \
  -b 3000 -a 3000 \
  --binSize 50 \
  --skipZeros \
  --missingDataAsZero \
  -o "$BASE/tss_c_meth_fraction_matrix.gz"

"$VENV/plotHeatmap" \
  -m "$BASE/tss_c_meth_fraction_matrix.gz" \
  -out "$BASE/tss_c_meth_fraction_heatmap_kmeans4_blue_red.png" \
  --kmeans 4 \
  --sortUsing mean \
  --samplesLabel "${LABELS[@]}" \
  --whatToShow "plot, heatmap and colorbar" \
  --colorMap RdYlBu_r

"$VENV/computeMatrix" reference-point -p 6 \
  -S "${A_TRACKS[@]}" \
  -R "$PROMOTERS" \
  --referencePoint center \
  -b 5000 -a 5000 \
  --binSize 50 \
  --skipZeros \
  --missingDataAsZero \
  -o "$BASE/gene_level_promoter_a_meth_frac_5k_clustered_summary_kmeans4_matrix.gz"

"$VENV/plotHeatmap" \
  -m "$BASE/gene_level_promoter_a_meth_frac_5k_clustered_summary_kmeans4_matrix.gz" \
  -out "$BASE/gene_level_promoter_a_meth_frac_5k_clustered_summary_kmeans4.png" \
  --kmeans 4 \
  --sortUsing mean \
  --samplesLabel "${LABELS[@]}" \
  --whatToShow "plot, heatmap and colorbar" \
  --colorMap RdYlBu_r

"$VENV/computeMatrix" scale-regions -p 4 \
  -S "${A_TRACKS[@]}" \
  -R "$HOX_GENES" \
  -b 5000 -a 5000 -m 10000 \
  --binSize 200 \
  --skipZeros \
  --missingDataAsZero \
  -o "$BASE/hox_genebody_a_meth_fraction_heatmap_sorted_200bp_matrix.gz"

"$VENV/plotHeatmap" \
  -m "$BASE/hox_genebody_a_meth_fraction_heatmap_sorted_200bp_matrix.gz" \
  -out "$BASE/hox_genebody_a_meth_fraction_heatmap_sorted_200bp.png" \
  --sortRegions keep \
  --samplesLabel "${LABELS[@]}" \
  --whatToShow "plot, heatmap and colorbar" \
  --colorMap RdYlBu_r
