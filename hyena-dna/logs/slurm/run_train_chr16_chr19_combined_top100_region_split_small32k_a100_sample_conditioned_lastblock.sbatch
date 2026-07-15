#!/bin/bash
# Combined chr16+chr19, merged_e5b+merged_c1:
# sample-conditioned last-block fine-tuning of HyenaDNA-small-32k.

#SBATCH --job-name=hyena32k_c16c19_sid
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-chr16-chr19-combined-sampleid-lastblock-train-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-hyena32k-chr16-chr19-combined-sampleid-lastblock-train-%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python train_region_split_hyenadna_two_head_sample_conditioned.py \
  --model-name hyenadna-small-32k-seqlen \
  --train-npz outputs/merged_e5b_c1_chr16_chr19_selected_top100_even50_region_split.train.npz \
  --val-npz outputs/merged_e5b_c1_chr16_chr19_selected_top100_even50_region_split.val.npz \
  --batch-size 1 \
  --max-length 32768 \
  --epochs 3 \
  --max-train-batches 500 \
  --max-val-batches 250 \
  --num-samples 2 \
  --sample-embedding-dim 16 \
  --lr 1e-3 \
  --backbone-lr 1e-5 \
  --unfreeze-prefix backbone.layers.3 \
  --unfreeze-prefix backbone.ln_f \
  --device cuda \
  --out outputs/hyenadna_small32k_chr16_chr19_combined_top100_even50_region_split_sampleid_lastblock_3epochs_500batches.pt
