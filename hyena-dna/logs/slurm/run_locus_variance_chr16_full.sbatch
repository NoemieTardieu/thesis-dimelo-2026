#!/bin/bash
#SBATCH --job-name=chr16_locus_variance
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-chr16-locus-variance-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-chr16-locus-variance-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${VSC_SCRATCH:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python -u analyze_locus_variance.py \
  --dataset \
    outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.train.npz \
    outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.train.metadata.tsv \
  --dataset \
    outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz \
    outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.metadata.tsv \
  --dataset \
    outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz \
    outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --bam merged_e5b=/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam \
  --bam merged_c1=/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam \
  --out-prefix outputs/chr16_overlap16k_full5000_locus_variance \
  --threshold 0.5 \
  --min-reads 3 \
  --collapse-cpg-dyads
