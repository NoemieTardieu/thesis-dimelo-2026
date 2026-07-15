#!/bin/bash
#SBATCH --job-name=alphagenome_eval
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-alphagenome-final-eval-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-alphagenome-final-eval-%j.err

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

"${PYTHON}" -c \
  "import matplotlib, numpy, pandas, scipy, sklearn; print('Final benchmark environment ready')"

"${PYTHON}" -u evaluate_alphagenome_vs_dimelo.py \
  --alphagenome outputs/alphagenome_test_128bp.tsv \
  --dimelo outputs/dimelo_test_128bp.tsv \
  --hyenadna outputs/hyenadna_test_128bp.tsv \
  --thresholds outputs/validation_thresholds.json \
  --out-prefix outputs/benchmark \
  --bootstrap-replicates 2000 \
  --seed 7

"${PYTHON}" -u plot_alphagenome_comparison.py \
  --alphagenome outputs/alphagenome_test_128bp.tsv \
  --dimelo outputs/dimelo_test_128bp.tsv \
  --hyenadna outputs/hyenadna_test_128bp.tsv \
  --out-dir outputs/plots/comparison \
  --max-regions 12

test -s outputs/benchmark.summary.tsv
test -s outputs/benchmark.per_region.tsv
test -s outputs/benchmark.summary.json

"${PYTHON}" -c \
  "import pandas as pd; s=pd.read_csv('outputs/benchmark.summary.tsv', sep='\t'); primary=s[s.threshold_name.eq('top_10_percent_primary')]; assert set(primary.model) == {'AlphaGenome','HyenaDNA'}; assert set(primary.experimental_sample) == {'merged_c1','merged_e5b','pooled'}; assert set(primary.scope) == {'pooled','chr11','chr16','chr17','chr19'}; assert primary.number_of_bins.min() > 0; print('summary_rows=', len(s)); print('primary_rows=', len(primary)); print(primary[(primary.scope=='pooled') & (primary.track.isin(['A549_H3K4me3_fixed_mean','HyenaDNA']))][['model','track','experimental_sample','number_of_bins','pearson','spearman','auroc','auprc','positive_fraction','auprc_enrichment']].to_string(index=False))"

plot_count=$(find outputs/plots/comparison -maxdepth 1 -name '*.png' | wc -l)
test "${plot_count}" -eq 12
echo "comparison_plots=${plot_count}"
echo "Final AlphaGenome/DiMeLo/HyenaDNA benchmark completed."
