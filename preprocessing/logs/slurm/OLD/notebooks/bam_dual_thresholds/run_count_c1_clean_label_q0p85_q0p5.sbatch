#!/bin/bash
#SBATCH --job-name=count_c1_cl085
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-count-c1-clean085-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-count-c1-clean085-%j.err

set -euo pipefail

BASE=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1
IN=$BASE/merged_c1_C_m_q0p8_training.valid5mer.uppercase.cnn_features.tsv.gz
OUT=/data/leuven/383/vsc38330/thesis_dimelo/train/c1_clean_label_q0p85_q0p5/c1_clean_label_counts_q0p85_q0p5.json

mkdir -p "$(dirname "$OUT")"

gzip -cd "$IN" \
| awk -F'\t' '
BEGIN{OFS="\t"; pos=0; neg=0; middle=0; malformed=0}
NR==1 {
  for (i=1; i<=NF; i++) {
    if ($i=="mod_qual") modqual=i
    if ($i=="ref_kmer") refk=i
    if ($i=="query_kmer") qk=i
  }
  next
}
{
  if (length($refk)!=5 || length($qk)!=5) {
    malformed++
    next
  }
  q=$modqual+0
  if (q >= 0.85) pos++
  else if (q <= 0.5) neg++
  else middle++
}
END{
  print "{"
  print "  \"positive_mod_qual_ge_0p85\": " pos ","
  print "  \"negative_mod_qual_le_0p5\": " neg ","
  print "  \"discarded_middle_0p5_to_0p85\": " middle ","
  print "  \"malformed_kmer_rows\": " malformed
  print "}"
}' > "$OUT"

cat "$OUT"
