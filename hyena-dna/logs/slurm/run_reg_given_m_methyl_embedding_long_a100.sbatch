#!/bin/bash
#SBATCH --job-name=reg_m_embed_long
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-reg-given-m-methyl-embedding-long-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-reg-given-m-methyl-embedding-long-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/thesis_project_clean
mkdir -p hyena-dna/results/checkpoints

/lustre1/scratch/383/vsc38330/.venv/bin/python -u hyena-dna/models/train_region_split_hyenadna_6ma_methyl_embedding_conditioned_nosample.py \
  --hyena-root /data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/upstream_hyena_dna \
  --checkpoint-dir /data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/server_artifacts/upstream_checkpoints \
  --train-npz hyena-dna/logs/generated_tensors_and_checkpoints/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.train.npz \
  --val-npz hyena-dna/logs/generated_tensors_and_checkpoints/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz \
  --out hyena-dna/results/checkpoints/hyenadna_reg_given_m_methyl_embedding_long.pt \
  --best-out hyena-dna/results/checkpoints/hyenadna_reg_given_m_methyl_embedding_long.best.pt \
  --epochs 15 \
  --max-train-batches 1000 \
  --max-val-batches 0 \
  --log-every-batches 10 \
  --eval-before-training \
  --lr 1e-4 \
  --methyl-embedding-lr 1e-4 \
  --backbone-lr 1e-6 \
  --decoder-hidden-dim 256 \
  --decoder-dropout 0.15 \
  --unknown-methylation-value 0.5 \
  --c-token-id 8 \
  --g-token-id 9 \
  --unfreeze-prefix backbone.embeddings \
  --unfreeze-prefix backbone.layers.3 \
  --unfreeze-prefix backbone.ln_f \
  --device cuda
