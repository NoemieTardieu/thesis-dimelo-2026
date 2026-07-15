#!/bin/bash
#SBATCH --job-name=alphagenome_200bp_followup
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-200bp-followup-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-200bp-followup-%j.err

set -euo pipefail

BENCHMARK_DIR=/data/leuven/383/vsc38330/alphagenome
PYTHON=/scratch/leuven/383/vsc38330/.venv/bin/python

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${VSC_SCRATCH:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

cd "${BENCHMARK_DIR}"

"${PYTHON}" -u run_200bp_followup.py \
  --bin-size 200 \
  --bootstrap-replicates 2000

test -s outputs/benchmark_200bp/benchmark_200bp.summary.tsv
test -s outputs/benchmark_200bp/thresholds_200bp.json
plot_count=$(find outputs/benchmark_200bp/plots -maxdepth 1 -name '*.png' | wc -l)
test "${plot_count}" -eq 12
echo "200 bp follow-up plots=${plot_count}"
