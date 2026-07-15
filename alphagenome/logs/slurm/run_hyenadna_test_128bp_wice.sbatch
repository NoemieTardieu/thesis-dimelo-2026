#!/bin/bash
#SBATCH --job-name=hyena_test_128bp
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/alphagenome/slurm-hyenadna-test-128bp-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/alphagenome/slurm-hyenadna-test-128bp-%j.err

set -euo pipefail

BENCHMARK_DIR=/data/leuven/383/vsc38330/alphagenome
HYENA_ROOT=/data/leuven/383/vsc38330/hyena-dna-main
TENSOR_DIR="${HYENA_ROOT}/preprocessing_chr16_merged_e5b/outputs"
CHECKPOINT_DIR="${HYENA_ROOT}/checkpoints"
CHECKPOINT="${TENSOR_DIR}/hyenadna_small32k_4chrom_overlap16k_full5000_region_split_nosample_short_2epochs_1000batches.pt"
PYTHON=/scratch/leuven/383/vsc38330/.venv/bin/python

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

cd "${BENCHMARK_DIR}"

"${PYTHON}" -c \
  "import torch, pysam; assert torch.cuda.is_available(); print('GPU=', torch.cuda.get_device_name(0)); print('HyenaDNA test environment ready')"

"${PYTHON}" -u build_hyenadna_128bp_tracks.py \
  --split test \
  --regions outputs/4chrom_test_regions.tsv \
  --outputs-dir "${TENSOR_DIR}" \
  --hyena-root "${HYENA_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --model-name hyenadna-small-32k-seqlen \
  --grid outputs/alphagenome_test_128bp.tsv \
  --device cuda \
  --out outputs/hyenadna_test_128bp.tsv

test -s outputs/hyenadna_test_128bp.tsv

"${PYTHON}" -c \
  "import pandas as pd; a=pd.read_csv('outputs/alphagenome_test_128bp.tsv', sep='\t'); h=pd.read_csv('outputs/hyenadna_test_128bp.tsv', sep='\t'); keys=['chrom','region_id','region_start','region_end','bin_start','bin_end']; expected=len(a); counts=h.groupby('sample').size().to_dict(); covered=h.groupby('sample')['mean_signal'].count().to_dict(); assert set(counts) == {'merged_c1','merged_e5b','pooled'}, counts; assert all(value == expected for value in counts.values()), (expected, counts); assert len(h[keys].drop_duplicates()) == expected; assert all(value > 0 for value in covered.values()), covered; print('alphagenome_bins=', expected); print('hyenadna_rows_per_sample=', counts); print('hyenadna_covered_bins=', covered)"

echo "HyenaDNA test track completed on the exact AlphaGenome grid."
