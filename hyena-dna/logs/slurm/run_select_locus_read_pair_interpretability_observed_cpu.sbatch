#!/bin/bash
#SBATCH --job-name=read_pair_interp
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-read-pair-interpretability-observed-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-read-pair-interpretability-observed-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/lustre1/scratch/383/vsc38330/.venv/bin/python -u select_locus_read_pair_interpretability.py \
  --dataset outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.metadata.tsv \
  --dataset outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --bam /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam \
  --sample merged_c1 \
  --chrom chr16 \
  --locus-variance outputs/full_chr16_c1_locus_variance.per_locus_variance.tsv.gz \
  --top-loci 1000 \
  --min-reads 8 \
  --local-window 80 \
  --similar-6ma-delta 0.05 \
  --max-candidate-loci 200 \
  --device cpu \
  --out-prefix outputs/chr16_c1_read_pair_interpretability_observed
