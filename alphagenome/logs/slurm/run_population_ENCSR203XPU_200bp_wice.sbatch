#!/bin/bash
#SBATCH --job-name=ENCSR203XPU_200bp_benchmark
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-ENCSR203XPU-200bp-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-ENCSR203XPU-200bp-%j.err

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

"${PYTHON}" -u population_track_benchmark.py \
  --external-bigwig external_tracks/ENCSR203XPU/ENCFF074PND_ENCSR203XPU_A549_H3K4me3_fold_change_GRCh38.bigWig \
  --out-dir outputs/population_track_benchmark_ENCSR203XPU_200bp_final \
  --bin-size 200 \
  --bootstrap-replicates 2000 \
  --external-cell-type A549 \
  --external-assay-type "ENCODE ENCSR203XPU H3K4me3 ChIP-seq fold change over control" \
  --external-value-type "fold change over control" \
  --external-value-transform "ENCODE processed fold-change bigWig, combined biological replicates 1,2,3" \
  --alphagenome-cell-type A549 \
  --max-stacked-regions 12

test -s outputs/population_track_benchmark_ENCSR203XPU_200bp_final/pairwise_metrics.tsv.gz
test -s outputs/population_track_benchmark_ENCSR203XPU_200bp_final/canonical_raw_bins.tsv.gz
test -s outputs/population_track_benchmark_ENCSR203XPU_200bp_final/text_summary.md
stacked_count=$(find outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/stacked_selected_regions -maxdepth 1 -name '*.png' | wc -l)
test "${stacked_count}" -eq 12
echo "ENCSR203XPU 200 bp benchmark complete; stacked plots=${stacked_count}"
