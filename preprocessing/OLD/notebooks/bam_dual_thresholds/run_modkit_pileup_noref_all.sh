#!/usr/bin/env bash
set -euo pipefail

export PATH="/data/leuven/383/vsc38330/bin/dist_modkit_v0.6.1_481e3c9:$PATH"

outdir="/scratch/leuven/383/vsc38330/pileup_tracks"
mkdir -p "$outdir"

declare -A bams
bams[h3k27ac]="/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam"
bams[h3k27me3]="/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam"
bams[h3k4me3]="/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam"

chroms=(
  chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11
  chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX
)

for mark in h3k27ac h3k27me3 h3k4me3; do
  bam="${bams[$mark]}"

  for chrom in "${chroms[@]}"; do
    base="${outdir}/${mark}_noref_${chrom}"

    echo "=== Running ${mark} ${chrom} ==="

    modkit pileup \
      "$bam" \
      "${base}.bed.gz" \
      --region "$chrom" \
      --bgzf \
      --filter-threshold A:0.9 \
      --filter-threshold C:0.8 \
      --log-filepath "${base}.log"

    echo "=== Splitting ${mark} ${chrom} Y/A ==="
    gzip -cd "${base}.bed.gz" \
      | awk 'BEGIN{OFS="\t"} $4=="Y" {print $1,$2,$3,$11}' \
      > "${base}_A_Y_percent.bedGraph"

    echo "=== Splitting ${mark} ${chrom} Z/C ==="
    gzip -cd "${base}.bed.gz" \
      | awk 'BEGIN{OFS="\t"} $4=="Z" {print $1,$2,$3,$11}' \
      > "${base}_C_Z_percent.bedGraph"

    echo "=== Done ${mark} ${chrom} ==="
  done
done
