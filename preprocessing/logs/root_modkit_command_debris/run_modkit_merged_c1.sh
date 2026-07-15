#!/usr/bin/env bash

set -euo pipefail

export PATH="/data/leuven/383/vsc38330/bin/dist_modkit_v0.6.1_481e3c9:$PATH"

bam="/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam"
outdir="/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/merged_c1"

mkdir -p "$outdir"

modkit modbam check-tags "$bam" > "$outdir/check_tags.txt" 2> "$outdir/check_tags.log"

modkit summary "$bam" \
  > "$outdir/summary.tsv" \
  2> "$outdir/summary.log"

modkit sample-probs "$bam" \
  > "$outdir/sample_probs.tsv" \
  2> "$outdir/sample_probs.log"

modkit extract calls "$bam" "$outdir/extract_calls.tsv.gz" \
  --ignore-index \
  --bgzf \
  --filter-threshold A:0.9 \
  --filter-threshold C:0.8 \
  --log-filepath "$outdir/extract_calls.log"
