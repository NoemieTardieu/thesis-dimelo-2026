#!/bin/bash
#SBATCH --job-name=full_chr16_variance
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-full-chr16-locus-variance-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-full-chr16-locus-variance-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${VSC_SCRATCH:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python -u analyze_full_chromosome_locus_variance.py \
  --extract-full merged_e5b=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/by_chrom/extract_full_chr16.tsv.gz \
  --extract-full merged_c1=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/by_chrom/extract_full_chr16.tsv.gz \
  --chrom chr16 \
  --chrom-length 90338345 \
  --out-prefix outputs/full_chr16_locus_variance \
  --min-reads 5 \
  --threshold 0.5 \
  --batch-size 1000000
