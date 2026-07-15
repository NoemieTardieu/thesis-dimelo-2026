#!/bin/bash
#SBATCH --job-name=verify_var_tail
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-verify-chr16-c1-locus-variance-thresholds-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-verify-chr16-c1-locus-variance-thresholds-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/lustre1/scratch/383/vsc38330/.venv/bin/python -u verify_locus_variance_thresholds.py \
  --per-locus outputs/full_chr16_c1_locus_variance.per_locus_variance.tsv.gz \
  --summary-json outputs/full_chr16_c1_locus_variance.summary.json \
  --extract-full /staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/by_chrom/extract_full_chr16.tsv.gz \
  --group merged_c1 \
  --chrom chr16 \
  --chrom-length 90338345 \
  --min-reads 5 \
  --threshold-line 0.02 \
  --batch-size 1000000 \
  --out-prefix outputs/full_chr16_c1_locus_variance_threshold_audit
