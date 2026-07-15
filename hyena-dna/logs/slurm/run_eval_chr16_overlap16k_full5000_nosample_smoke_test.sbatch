#!/bin/bash
#SBATCH --job-name=eval_nosample_smoke
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-chr16-overlap16k-full5000-nosample-smoke-test-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-chr16-overlap16k-full5000-nosample-smoke-test-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python -u evaluate_region_split_hyenadna_two_head_extended.py \
  --model-name hyenadna-small-32k-seqlen \
  --npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz \
  --metadata outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --checkpoint outputs/hyenadna_small32k_chr16_overlap16k_full5000_region_split_nosample_smoke_2epochs_100batches.pt \
  --split-name test \
  --out-prefix outputs/hyenadna_small32k_chr16_overlap16k_full5000_nosample_smoke.test \
  --batch-size 1 \
  --max-length 32768 \
  --device cuda
