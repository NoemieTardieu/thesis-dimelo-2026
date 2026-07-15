#!/bin/bash
#SBATCH --job-name=chr16_c1_var_cigar
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-chr16-c1-variance-cigar-dna-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-chr16-c1-variance-cigar-dna-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${VSC_SCRATCH:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/venvs/thesis_env/bin/python -u analyze_variance_loci_cigar_dna.py \
  --variance-tsv outputs/full_chr16_c1_locus_variance.per_locus_variance.tsv.gz \
  --bam /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam \
  --reference-fasta /data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa \
  --out-prefix outputs/full_chr16_c1_variance_cigar_dna_top500_bottom500 \
  --group merged_c1 \
  --chrom chr16 \
  --marks 5mC 6mA \
  --top-n 500 \
  --bottom-n 500 \
  --min-reads 10 \
  --max-reads-per-locus 250 \
  --local-window 20 \
  --insertion-window 5
