#!/bin/bash
#SBATCH --job-name=m_reg_chr16
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-m-reg-chr16-val-test-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-m-reg-chr16-val-test-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/venvs/thesis_env/bin/python -u analyze_final_m_reg_relationship.py \
  --dataset outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.metadata.tsv chr16_val \
  --dataset outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv chr16_test \
  --out-prefix outputs/final_m_reg_relationship_chr16_val_test_overlapagg \
  --cooccurrence-bins 5
