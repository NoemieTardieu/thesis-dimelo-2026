#!/bin/bash
#SBATCH --job-name=alphagenome_readseq_aggregate
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-readseq-aggregate-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-readseq-aggregate-%j.err

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

"${PYTHON}" -u build_alphagenome_readseq_aggregate.py \
  --bin-size 200 \
  --sequence-length 131072 \
  --max-reads-per-region-per-sample 2 \
  --out outputs/alphagenome_readseq_200bp.tsv \
  --summary outputs/alphagenome_readseq_200bp.summary.json

test -s outputs/alphagenome_readseq_200bp.tsv

"${PYTHON}" -u evaluate_readseq_alphagenome_200bp.py \
  --bootstrap-replicates 0 \
  --out-prefix outputs/alphagenome_readseq_200bp_benchmark

test -s outputs/alphagenome_readseq_200bp_benchmark.summary.tsv

"${PYTHON}" -c \
  "import pandas as pd; df=pd.read_csv('outputs/alphagenome_readseq_200bp_benchmark.summary.tsv', sep='\t'); print(df[['model','experimental_sample','number_of_bins','pearson','spearman','auroc','auprc']].to_string(index=False))"

echo "AlphaGenome read-sequence aggregation completed."
