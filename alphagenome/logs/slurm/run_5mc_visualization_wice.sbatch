#!/bin/bash
#SBATCH --job-name=hyena_5mc_plot
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-5mc-visualization-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-5mc-visualization-%j.err

set -euo pipefail

BENCHMARK_DIR=/data/leuven/383/vsc38330/alphagenome
PYTHON=/scratch/leuven/383/vsc38330/.venv/bin/python

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export MPLCONFIGDIR="${VSC_SCRATCH:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

cd "${BENCHMARK_DIR}"

"${PYTHON}" -c \
  "import torch, pysam, matplotlib; assert torch.cuda.is_available(); print('GPU=', torch.cuda.get_device_name(0)); print('5mC visualization environment ready')"

"${PYTHON}" -u build_plot_5mc_hyenadna.py \
  --split test \
  --regions outputs/4chrom_test_regions.tsv \
  --bin-size 1000 \
  --max-regions 12 \
  --device cuda

test -s outputs/5mc_visual/dimelo_5mc_test_1000bp.tsv
test -s outputs/5mc_visual/hyenadna_5mc_test_1000bp.tsv
plot_count=$(find outputs/5mc_visual/plots_5mc_1000bp -maxdepth 1 -name '*.png' | wc -l)
test "${plot_count}" -eq 12
echo "5mC plots=${plot_count}"
