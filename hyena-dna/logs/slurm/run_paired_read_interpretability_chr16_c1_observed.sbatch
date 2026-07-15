#!/bin/bash
#SBATCH --job-name=paired_interp_c1
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-paired-read-interpretability-chr16-c1-observed-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-paired-read-interpretability-chr16-c1-observed-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/lustre1/scratch/383/vsc38330/.venv/bin/python -u paired_read_interpretability_analysis.py \
  --dataset outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.metadata.tsv \
  --dataset outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --bam /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam \
  --sample merged_c1 \
  --chrom chr16 \
  --locus-variance outputs/full_chr16_c1_locus_variance.per_locus_variance.tsv.gz \
  --reference-fasta /data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa \
  --top-loci 500 \
  --max-candidate-loci 120 \
  --max-reads-per-locus 25 \
  --window-size 1000 \
  --bin-size 50 \
  --min-reads-locus 8 \
  --min-window-coverage-fraction 0.60 \
  --min-pair-comparable-fraction 0.60 \
  --min-valid-6ma-positions 10 \
  --min-valid-5mc-positions 1 \
  --max-missing-signal-fraction 0.95 \
  --min-valid-per-bin 1 \
  --min-shared-bins 2 \
  --low-quantile 0.10 \
  --high-quantile 0.90 \
  --top-n-per-case 3 \
  --seed 7 \
  --out-prefix outputs/chr16_c1_paired_read_interpretability_observed
