#!/bin/bash
#SBATCH --job-name=eval_reg_m_nosamp
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-eval-reg-given-m-nosample-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-eval-reg-given-m-nosample-%j.err

set -euo pipefail

: "${NPZ:?Need NPZ=path/to/split.npz}"
: "${METADATA:?Need METADATA=path/to/split.metadata.tsv}"
: "${CHECKPOINT:?Need CHECKPOINT=path/to/model.best.pt}"
: "${SPLIT_NAME:?Need SPLIT_NAME=val/test/etc}"
: "${OUT_PREFIX:?Need OUT_PREFIX=outputs/name}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/lustre1/scratch/383/vsc38330/.venv/bin/python -u evaluate_region_split_hyenadna_6ma_methyl_conditioned_nosample_overlap_aggregated.py \
  --model-name hyenadna-small-32k-seqlen \
  --npz "${NPZ}" \
  --metadata "${METADATA}" \
  --checkpoint "${CHECKPOINT}" \
  --split-name "${SPLIT_NAME}" \
  --out-prefix "${OUT_PREFIX}" \
  --batch-size 1 \
  --max-length 32768 \
  --device cuda
