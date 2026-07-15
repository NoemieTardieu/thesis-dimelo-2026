#!/bin/bash
# Partial test eval for the 4-chromosome short sample-conditioned checkpoint.
# This evaluates 1000 reads first, so we get metrics without risking another walltime timeout.

#SBATCH --job-name=hyena32k_4chr_test_p1k
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-4chr-combined-sampleid-short-test-extended-partial1000-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-4chr-combined-sampleid-short-test-extended-partial1000-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python -u evaluate_region_split_hyenadna_two_head_sample_conditioned_extended.py \
  --model-name hyenadna-small-32k-seqlen \
  --npz outputs/merged_e5b_c1_chr16_chr19_chr11_chr17_selected_top100_even50_region_split.test.npz \
  --metadata outputs/merged_e5b_c1_chr16_chr19_chr11_chr17_selected_top100_even50_region_split.test.metadata.tsv \
  --split-name test \
  --checkpoint outputs/hyenadna_small32k_4chrom_combined_top100_even50_region_split_sampleid_short_2epochs_150batches.pt \
  --batch-size 1 \
  --max-length 32768 \
  --max-batches 1000 \
  --progress-every 250 \
  --device cuda \
  --out-prefix outputs/hyenadna_small32k_4chrom_combined_top100_even50_region_split.test.sampleid.short.partial1000.extended
