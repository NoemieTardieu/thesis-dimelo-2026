#!/bin/bash
#SBATCH --job-name=eval_membed_chr11
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-eval-reg-given-m-methyl-embedding-long-best-chr11-test-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-eval-reg-given-m-methyl-embedding-long-best-chr11-test-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/thesis_project_clean

/lustre1/scratch/383/vsc38330/.venv/bin/python -u hyena-dna/evaluation/evaluate_region_split_hyenadna_6ma_methyl_embedding_conditioned_nosample_overlap_aggregated.py \
  --hyena-root /data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/upstream_hyena_dna \
  --checkpoint-dir /data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/server_artifacts/upstream_checkpoints \
  --npz hyena-dna/logs/generated_tensors_and_checkpoints/merged_e5b_c1_chr11_selected_top100_overlap16k_full5000_region_split.test.npz \
  --metadata hyena-dna/logs/generated_tensors_and_checkpoints/merged_e5b_c1_chr11_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --checkpoint hyena-dna/results/checkpoints/hyenadna_reg_given_m_methyl_embedding_long.best.pt \
  --split-name chr11_test \
  --out-prefix hyena-dna/results/hyenadna_reg_given_m_methyl_embedding_long_best_chr11_test_overlapagg \
  --batch-size 1 \
  --max-length 32768 \
  --device cuda
