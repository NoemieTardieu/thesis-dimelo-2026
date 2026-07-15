#!/bin/bash
#SBATCH --job-name=eval_4chrom_onechr
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-4chrom-nosample-short-on-%x-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-4chrom-nosample-short-on-%x-%j.err

set -euo pipefail

: "${CHR:?Need CHR=chr16/chr19/chr11/chr17}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python -u evaluate_region_split_hyenadna_two_head_overlap_aggregated.py \
  --model-name hyenadna-small-32k-seqlen \
  --npz outputs/merged_e5b_c1_${CHR}_selected_top100_overlap16k_full5000_region_split.test.npz \
  --metadata outputs/merged_e5b_c1_${CHR}_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --checkpoint outputs/hyenadna_small32k_4chrom_overlap16k_full5000_region_split_nosample_short_2epochs_1000batches.pt \
  --split-name test \
  --out-prefix outputs/hyenadna_small32k_4chrom_overlap16k_full5000_nosample_short_on_${CHR}_overlapagg_test \
  --batch-size 1 \
  --max-length 32768 \
  --device cuda
