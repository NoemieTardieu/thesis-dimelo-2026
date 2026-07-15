#!/bin/bash
# First-pass merged_c1 extract-full rerun: one job per chromosome.
# This is the successful pattern that was used before the heavy chromosomes
# were rescued by chunks.

#SBATCH --job-name=merged_c1_full_chr
#SBATCH -M genius
#SBATCH -A lccbe
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=180000M
#SBATCH --array=1-23%2
#SBATCH --output=/staging/leuven/stg_00118/BAM_Noemie/C1_merged_preprocessed_files/slurm-c1-full-chr-%A_%a.out
#SBATCH --error=/staging/leuven/stg_00118/BAM_Noemie/C1_merged_preprocessed_files/slurm-c1-full-chr-%A_%a.err

set -euo pipefail

source /data/leuven/383/vsc38330/.venv/bin/activate
export PATH="/data/leuven/383/vsc38330/bin/dist_modkit_v0.6.1_481e3c9:$PATH"

BAM=/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam
REF=/data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa
CHROMS=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/merged_c1_chromosomes_all.txt
OUTDIR=/staging/leuven/stg_00118/BAM_Noemie/C1_merged_preprocessed_files/by_chrom

REGION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CHROMS")
mkdir -p "$OUTDIR"

modkit extract full \
  "$BAM" \
  "$OUTDIR/extract_full_${REGION}.tsv.gz" \
  --bgzf \
  --region "$REGION" \
  --ref "$REF"
