#!/bin/bash
# Evaluate the 4-chromosome sample-conditioned checkpoint on a chosen split/dataset.
#
# Required sbatch exports:
#   SPLIT      val or test
#   LABEL      short label for output filenames, e.g. chr19 or 4chrom
#   NPZ        tensor npz path
#   METADATA   metadata tsv path
#
# Optional sbatch exports:
#   MAX_BATCHES  maximum number of batches/reads to evaluate
#   CHECKPOINT   checkpoint path. Defaults to the 4-chromosome checkpoint.
#   MODEL_LABEL  output filename label. Defaults to 4chrom_checkpoint.

#SBATCH --job-name=hyena32k_eval_ds
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-4chr-ckpt-eval-%x-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-4chr-ckpt-eval-%x-%j.err

set -euo pipefail

: "${SPLIT:?SPLIT is required}"
: "${LABEL:?LABEL is required}"
: "${NPZ:?NPZ is required}"
: "${METADATA:?METADATA is required}"
CHECKPOINT="${CHECKPOINT:-outputs/hyenadna_small32k_4chrom_combined_top100_even50_region_split_sampleid_short_2epochs_150batches.pt}"
MODEL_LABEL="${MODEL_LABEL:-4chrom_checkpoint}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

MAX_BATCH_ARGS=()
if [[ -n "${MAX_BATCHES:-}" ]]; then
  MAX_BATCH_ARGS=(--max-batches "${MAX_BATCHES}" --progress-every 250)
else
  MAX_BATCH_ARGS=(--progress-every 250)
fi

/data/leuven/383/vsc38330/.venv/bin/python -u evaluate_region_split_hyenadna_two_head_sample_conditioned_extended.py \
  --model-name hyenadna-small-32k-seqlen \
  --npz "${NPZ}" \
  --metadata "${METADATA}" \
  --split-name "${SPLIT}" \
  --checkpoint "${CHECKPOINT}" \
  --batch-size 1 \
  --max-length 32768 \
  "${MAX_BATCH_ARGS[@]}" \
  --device cuda \
  --out-prefix "outputs/hyenadna_small32k_${MODEL_LABEL}_on_${LABEL}.${SPLIT}.sampleid.extended"
