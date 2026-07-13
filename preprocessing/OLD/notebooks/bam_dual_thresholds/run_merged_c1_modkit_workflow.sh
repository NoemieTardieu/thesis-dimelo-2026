#!/usr/bin/env bash

set -euo pipefail

export PATH="/data/leuven/383/vsc38330/bin/dist_modkit_v0.6.1_481e3c9:$PATH"

bam="/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam"
ref="/omics/groups/OE0219/internal/Etienne/data/reference/GRCh38/GRCh38.d1.vd1.fa"

# Heavy intermediate work stays on scratch.
scratch_preproc="/scratch/leuven/383/vsc38330/modkit_preproc/merged_c1"
scratch_pileup="/scratch/leuven/383/vsc38330/pileup_tracks/merged_c1"

# Keep only selected final outputs in /data.
final_dir="/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/merged_c1"

mkdir -p "$scratch_preproc"
mkdir -p "$scratch_pileup"
mkdir -p "$final_dir"

if [[ ! -f "$bam" ]]; then
  echo "Missing BAM: $bam" >&2
  exit 1
fi

if [[ ! -f "$ref" ]]; then
  echo "Missing reference: $ref" >&2
  exit 1
fi

if [[ ! -f "${ref}.fai" ]]; then
  echo "Missing reference index: ${ref}.fai" >&2
  exit 1
fi

echo "Step 1: check modBAM tags"
modkit modbam check-tags "$bam" --head 1000 \
  > "$scratch_preproc/check_tags.stdout.txt" \
  2> "$scratch_preproc/check_tags.stderr.txt"

echo "Step 2: optional summary and sample-probs"
modkit summary "$bam" \
  > "$scratch_preproc/summary.tsv" \
  2> "$scratch_preproc/summary.log"

modkit sample-probs "$bam" \
  > "$scratch_preproc/sample_probs.tsv" \
  2> "$scratch_preproc/sample_probs.log"

echo "Step 3: extract calls for all A:a"
modkit extract calls "$bam" "$scratch_preproc/extract_calls_A_all.tsv.gz" \
  --ignore-index \
  --bgzf \
  --modified-bases A:a \
  --filter-threshold A:0.9 \
  --filter-threshold C:0.8 \
  --log-filepath "$scratch_preproc/extract_calls_A_all.log"

echo "Step 4: extract calls for combined CpG-only C"
modkit extract calls "$bam" "$scratch_preproc/extract_calls_C_combined_cpg.tsv.gz" \
  --ignore-index \
  --bgzf \
  --modified-bases C \
  --combine-mods \
  --cpg \
  --reference "$ref" \
  --filter-threshold A:0.9 \
  --filter-threshold C:0.8 \
  --log-filepath "$scratch_preproc/extract_calls_C_combined_cpg.log"

echo "Step 5: pileup for all A:a"
modkit pileup "$bam" "$scratch_pileup/merged_c1_A_a_all.bed.gz" \
  --modified-bases A:a \
  --bgzf \
  --filter-threshold A:0.9 \
  --filter-threshold C:0.8 \
  --log-filepath "$scratch_pileup/merged_c1_A_a_all.log"

echo "Step 6: pileup for combined CpG-only C"
modkit pileup "$bam" "$scratch_pileup/merged_c1_C_combined_cpg.bed.gz" \
  --modified-bases C \
  --combine-mods \
  --cpg \
  --reference "$ref" \
  --bgzf \
  --filter-threshold A:0.9 \
  --filter-threshold C:0.8 \
  --log-filepath "$scratch_pileup/merged_c1_C_combined_cpg.log"

echo "Step 7: copy selected final outputs back to /data"
cp "$scratch_preproc/check_tags.stderr.txt" "$final_dir/" 2>/dev/null || true
cp "$scratch_preproc/summary.tsv" "$final_dir/" 2>/dev/null || true
cp "$scratch_preproc/sample_probs.tsv" "$final_dir/" 2>/dev/null || true
cp "$scratch_preproc/extract_calls_A_all.tsv.gz" "$final_dir/" 2>/dev/null || true
cp "$scratch_preproc/extract_calls_C_combined_cpg.tsv.gz" "$final_dir/" 2>/dev/null || true

echo "Done."
echo "Scratch intermediates:"
echo "  $scratch_preproc"
echo "  $scratch_pileup"
echo "Selected final outputs copied to:"
echo "  $final_dir"
