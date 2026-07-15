#!/bin/bash
#SBATCH --job-name=chr11_c1_variance
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-full-chr11-c1-locus-variance-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-full-chr11-c1-locus-variance-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/lustre1/scratch/383/vsc38330/.venv/bin/python -u analyze_full_chromosome_locus_variance.py \
  --extract-full merged_c1=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/by_chrom/extract_full_chr11.tsv.gz \
  --chrom chr11 \
  --chrom-length 135086622 \
  --out-prefix outputs/full_chr11_c1_locus_variance \
  --min-reads 5 \
  --threshold 0.5 \
  --batch-size 1000000
