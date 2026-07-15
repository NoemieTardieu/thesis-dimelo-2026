#!/bin/bash
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc38330
#SBATCH --partition=batch
#SBATCH --job-name=merged_c1_modcount
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/modcount-%j.out
#SBATCH --error=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/modcount-%j.err

set -euo pipefail

IN=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/extract_full_merged_c1_complete.tsv.gz
OUT=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/c_mod_code_counts.txt

gzip -cd "$IN" | awk -F'\t' '
NR==1 {
  for (i=1; i<=NF; i++) {
    if ($i=="canonical_base") cb=i
    if ($i=="mod_code") mc=i
  }
  next
}
$cb=="C" { c[$mc]++ }
END {
  for (k in c) print k, c[k]
}' | sort > "$OUT"
