#!/bin/bash
#SBATCH --job-name=ag_followup
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-promoter-followup-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-promoter-followup-%j.err

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

"${PYTHON}" -u promoter_followup_analysis.py

test -s outputs/promoter_followup/alphagenome_vs_hyenadna_primary_metrics.tsv
test -s outputs/promoter_followup/bin_size_sensitivity.tsv
plot_count=$(find outputs/promoter_followup/normalized_minmax_plots -maxdepth 1 -name '*.png' | wc -l)
test "${plot_count}" -eq 12

"${PYTHON}" -c \
  "import pandas as pd; m=pd.read_csv('outputs/promoter_followup/alphagenome_vs_hyenadna_primary_metrics.tsv', sep='\t'); b=pd.read_csv('outputs/promoter_followup/bin_size_sensitivity.tsv', sep='\t'); print('primary_metrics_rows=', len(m)); print('bin_sensitivity_rows=', len(b)); print(b[(b.bin_size.isin([128,1024])) & (b.experimental_sample=='pooled')][['bin_size','model','number_of_bins','pearson','spearman','auroc','auprc','positive_fraction']].to_string(index=False))"

echo "Promoter follow-up analysis completed."
