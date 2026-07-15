#!/bin/bash
#SBATCH --job-name=m_reg_4chrom
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-m-reg-4chrom-test-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-m-reg-4chrom-test-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/venvs/thesis_env/bin/python -u analyze_final_m_reg_relationship.py \
  --dataset outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv chr16_test \
  --dataset outputs/merged_e5b_c1_chr11_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr11_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv chr11_test \
  --dataset outputs/merged_e5b_c1_chr17_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr17_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv chr17_test \
  --dataset outputs/merged_e5b_c1_chr19_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr19_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv chr19_test \
  --out-prefix outputs/final_m_reg_relationship_4chrom_test_overlapagg \
  --cooccurrence-bins 5
