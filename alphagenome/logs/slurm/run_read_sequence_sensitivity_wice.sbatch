#!/bin/bash
#SBATCH --job-name=alphagenome_readseq_sensitivity
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-readseq-sensitivity-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-readseq-sensitivity-%j.err

set -euo pipefail

BENCHMARK_DIR=/data/leuven/383/vsc38330/alphagenome
PYTHON=/scratch/leuven/383/vsc38330/.venv/bin/python

if [ -f /user/leuven/383/vsc38330/.config/alphagenome/env ]; then
  set -a
  source /user/leuven/383/vsc38330/.config/alphagenome/env
  set +a
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${BENCHMARK_DIR}"

"${PYTHON}" -u alphagenome_read_sequence_sensitivity.py \
  --sequence-length 16384 \
  --max-reads-per-sample 6 \
  --retries 4

test -s outputs/read_sequence_sensitivity/read_sequence_vs_reference_summary.tsv
echo "Read-sequence AlphaGenome sensitivity completed."
