#!/bin/bash
#SBATCH --job-name=alphagenome_4chr
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-alphagenome-4chrom-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-alphagenome-4chrom-%j.err

set -euo pipefail

BENCHMARK_DIR=/data/leuven/383/vsc38330/alphagenome
PYTHON=/scratch/leuven/383/vsc38330/.venv/bin/python
KEY_FILE=/user/leuven/383/vsc38330/.config/alphagenome/env

source "${KEY_FILE}"
: "${ALPHAGENOME_API_KEY:?ALPHAGENOME_API_KEY was not exported by ${KEY_FILE}}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${VSC_SCRATCH:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

cd "${BENCHMARK_DIR}"

"${PYTHON}" -c \
  "import alphagenome, os; assert os.environ.get('ALPHAGENOME_API_KEY'); print('AlphaGenome environment ready')"

"${PYTHON}" -u run_alphagenome_4chrom.py
"${PYTHON}" -u export_alphagenome_128bp_tracks.py

echo "AlphaGenome four-chromosome query and export completed."
