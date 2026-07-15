#!/bin/bash
# Tensorize selected chr16 regions with an even cap of 50 reads per selected region.
# This version uses top100 selected regions, giving up to about 5000 reads.

#SBATCH --job-name=chr16_top100_e50
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-chr16-top100-even50-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-chr16-top100-even50-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
  --regions-bed regions/merged_e5b_chr16_selected_100kb_top100.bed \
  --max-reads-per-region 50 \
  --max-reads 5000 \
  --out-prefix outputs/merged_e5b_chr16_selected_top100_even50
