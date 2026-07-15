#!/bin/bash
#SBATCH --job-name=dimelo_val_128bp
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-dimelo-validation-128bp-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-dimelo-validation-128bp-%j.err

set -euo pipefail

BENCHMARK_DIR=/data/leuven/383/vsc38330/alphagenome
TENSOR_DIR=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/outputs
PYTHON=/scratch/leuven/383/vsc38330/.venv/bin/python

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${BENCHMARK_DIR}"

"${PYTHON}" -c \
  "import numpy, pandas, pysam; print('DiMeLo validation environment ready')"

"${PYTHON}" -u build_dimelo_128bp_tracks.py \
  --split val \
  --regions outputs/4chrom_val_regions.tsv \
  --outputs-dir "${TENSOR_DIR}" \
  --out outputs/dimelo_val_128bp.tsv

"${PYTHON}" -u fit_validation_thresholds.py \
  --validation-track outputs/dimelo_val_128bp.tsv \
  --out outputs/validation_thresholds.json

test -s outputs/dimelo_val_128bp.tsv
test -s outputs/validation_thresholds.json

"${PYTHON}" -c \
  "import json, pandas as pd; d=pd.read_csv('outputs/dimelo_val_128bp.tsv', sep='\t'); c=d[['chrom','region_id']].drop_duplicates().groupby('chrom').size().to_dict(); assert c == {'chr11': 15, 'chr16': 15, 'chr17': 15, 'chr19': 15}, c; t=json.load(open('outputs/validation_thresholds.json')); print('regions_per_chromosome=', c); print('thresholds=', t['thresholds'])"

echo "DiMeLo validation tracks and frozen thresholds completed."
