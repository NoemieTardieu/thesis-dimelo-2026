#!/bin/bash
#SBATCH --job-name=ag_encode_random_sensitivity
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_project_clean/alphagenome/logs/slurm/slurm-ag-encode-random-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_project_clean/alphagenome/logs/slurm/slurm-ag-encode-random-%j.err

set -euo pipefail

BENCHMARK_DIR=/data/leuven/383/vsc38330/thesis_project_clean/alphagenome
PYTHON=/scratch/leuven/383/vsc38330/.venv/bin/python
KEY_FILE=/user/leuven/383/vsc38330/.config/alphagenome/env
FAI=/staging/leuven/stg_00118/antonella/reference/GRCh38.p13.genome.fa.fai

source "${KEY_FILE}"
: "${ALPHAGENOME_API_KEY:?ALPHAGENOME_API_KEY was not exported by ${KEY_FILE}}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${BENCHMARK_DIR}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${VSC_SCRATCH:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

cd "${BENCHMARK_DIR}"

"${PYTHON}" -u scripts/make_alphagenome_random_regions.py \
  --fasta-index "${FAI}" \
  --out metadata/regions/4chrom_random_sensitivity_regions.tsv

"${PYTHON}" -u scripts/run_alphagenome_4chrom.py \
  --regions metadata/regions/4chrom_random_sensitivity_regions.tsv \
  --selected-tracks metadata/selected_a549_h3k4me3_tracks.tsv \
  --cache-dir logs/cache/random_encode_sensitivity_cache \
  --fasta-index "${FAI}" \
  --retries 5

"${PYTHON}" -u scripts/evaluate_alphagenome_encode_sensitivity.py \
  --regions metadata/regions/4chrom_random_sensitivity_regions.tsv \
  --cache-dir logs/cache/random_encode_sensitivity_cache \
  --label random_regions \
  --out-dir results/alphagenome_encode_sensitivity

test -s results/alphagenome_encode_sensitivity/random_regions.metrics.tsv
test -s results/alphagenome_encode_sensitivity/random_regions.summary.md

echo "AlphaGenome-ENCODE random-region sensitivity check complete."
