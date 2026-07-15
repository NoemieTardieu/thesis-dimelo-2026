#!/bin/bash
# Tensorize selected chr16 regions with an even cap of 20 reads per selected region.

#SBATCH --job-name=chr16_even20
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-chr16-even20-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-chr16-even20-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
  --regions-bed regions/merged_e5b_chr16_selected_100kb_top50.bed \
  --max-reads-per-region 20 \
  --max-reads 1000 \
  --out-prefix outputs/merged_e5b_chr16_selected_top50_even20
