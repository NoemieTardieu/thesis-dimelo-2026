#!/usr/bin/env bash

set -euo pipefail

export PATH="/data/leuven/383/vsc38330/bin/dist_modkit_v0.6.1_481e3c9:$PATH"

outdir="/scratch/leuven/383/vsc38330/pileup_tracks"

run_mark() {
  local mark="$1"
  local bam="$2"
  shift 2
  local chrom

  for chrom in "$@"; do
    local base="${outdir}/${mark}_noref_${chrom}"
    local bed="${base}.bed.gz"
    local ay="${base}_A_Y_percent.bedGraph"
    local cz="${base}_C_Z_percent.bedGraph"
    local log="${base}.log"

    rm -f "$bed" "$ay" "$cz" "$log"

    echo "=== $(date '+%F %T') ${mark} ${chrom} ==="
    modkit pileup "$bam" "$bed" \
      --region "$chrom" \
      --bgzf \
      --filter-threshold A:0.9 \
      --filter-threshold C:0.8 \
      --log-filepath "$log"

    echo "=== $(date '+%F %T') ${mark} ${chrom} bedGraph extraction ==="
    zcat "$bed" | awk '$4=="Y" {print $1,$2,$3,$11}' OFS="\t" > "$ay"
    zcat "$bed" | awk '$4=="Z" {print $1,$2,$3,$11}' OFS="\t" > "$cz"
    echo "=== $(date '+%F %T') ${mark} ${chrom} done ==="
  done
}

run_mark \
  h3k27me3 \
  /scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam \
  chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX

run_mark \
  h3k4me3 \
  /scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam \
  chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX
