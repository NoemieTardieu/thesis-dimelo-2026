#!/bin/bash
#SBATCH --job-name=build_c1_cl085
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-build-c1-clean085-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-build-c1-clean085-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
ENCODER=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/encode_cnn_kmers_to_npz.py

SRC_BASE=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1
IN=$SRC_BASE/merged_c1_C_m_q0p8_training.valid5mer.uppercase.cnn_features.tsv.gz

OUT_BASE=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_clean_label_q0p85_q0p5
FEATURES=$OUT_BASE/merged_c1_C_m_cleanlabel_pos0p85_neg0p5.valid5mer.uppercase.cnn_features.tsv.gz
TENSOR_DIR=$OUT_BASE/tensors_integer_valid5mer_uppercase_cleanlabel_pos0p85_neg0p5

mkdir -p "$OUT_BASE"
rm -rf "$TENSOR_DIR"

gzip -cd "$IN" \
| awk -F'\t' '
BEGIN{OFS="\t"}
NR==1 {
  for (i=1; i<=NF; i++) {
    if ($i=="chrom") chrom=i
    if ($i=="ref_position") refpos=i
    if ($i=="mod_qual") modqual=i
    if ($i=="ref_kmer") refk=i
    if ($i=="query_kmer") qk=i
    if ($i=="forward_read_position") frp=i
    if ($i=="read_length") rl=i
    if ($i=="ref_strand") rs=i
    if ($i=="ref_mod_strand") rms=i
  }
  print "chrom","ref_position","mod_qual","ref_kmer","query_kmer","forward_read_position","read_length","ref_strand","ref_mod_strand","label"
  next
}
length($refk)==5 && length($qk)==5 {
  q=$modqual+0
  if (q >= 0.85) {
    print $chrom,$refpos,$modqual,$refk,$qk,$frp,$rl,$rs,$rms,1
  } else if (q <= 0.5) {
    print $chrom,$refpos,$modqual,$refk,$qk,$frp,$rl,$rs,$rms,0
  }
}' \
| gzip > "$FEATURES"

$PY "$ENCODER" \
  --input "$FEATURES" \
  --output-dir "$TENSOR_DIR" \
  --chunk-size 2000000

cat "$TENSOR_DIR/summary.json"
