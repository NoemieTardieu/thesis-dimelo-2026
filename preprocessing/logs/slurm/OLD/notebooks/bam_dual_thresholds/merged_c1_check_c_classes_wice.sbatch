#!/bin/bash
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc38330
#SBATCH --partition=batch
#SBATCH --job-name=merged_c1_checkC
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/check_c_classes-%j.out
#SBATCH --error=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/check_c_classes-%j.err

set -euo pipefail

gzip -cd /staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/extract_full_merged_c1_complete.tsv.gz \
| awk -F'\t' '
NR==1 {
  for (i=1; i<=NF; i++) {
    if ($i=="canonical_base") cb=i
    if ($i=="mod_code") mc=i
  }
  next
}
$cb=="C" {count[$mc]++}
END {
  for (k in count) print "[" k "]", count[k]
}' \
| sort
