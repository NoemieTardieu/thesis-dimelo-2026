#!/bin/bash
#SBATCH --job-name=dimelo_test_128bp
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-dimelo-test-128bp-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-dimelo-test-128bp-%j.err

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
  "import numpy, pandas, pysam; print('DiMeLo test environment ready')"

"${PYTHON}" -u build_dimelo_128bp_tracks.py \
  --split test \
  --regions outputs/4chrom_test_regions.tsv \
  --outputs-dir "${TENSOR_DIR}" \
  --grid outputs/alphagenome_test_128bp.tsv \
  --out outputs/dimelo_test_128bp.tsv

test -s outputs/dimelo_test_128bp.tsv

"${PYTHON}" -c \
  "import pandas as pd; a=pd.read_csv('outputs/alphagenome_test_128bp.tsv', sep='\t'); d=pd.read_csv('outputs/dimelo_test_128bp.tsv', sep='\t'); keys=['chrom','region_id','region_start','region_end','bin_start','bin_end']; expected=len(a); counts=d.groupby('sample').size().to_dict(); assert set(counts) == {'merged_c1','merged_e5b','pooled'}, counts; assert all(value == expected for value in counts.values()), (expected, counts); assert len(d[keys].drop_duplicates()) == expected; print('alphagenome_bins=', expected); print('dimelo_rows_per_sample=', counts)"

echo "DiMeLo test track completed on the exact AlphaGenome grid."
