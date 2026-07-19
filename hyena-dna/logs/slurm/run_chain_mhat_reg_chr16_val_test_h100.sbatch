#!/bin/bash
#SBATCH --job-name=chain_mhat_reg16
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-chain-mhat-reg-chr16-val-test-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/logs/slurm/slurm-chain-mhat-reg-chr16-val-test-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/thesis_project_clean

PY=/lustre1/scratch/383/vsc38330/.venv/bin/python
TENSOR_DIR=hyena-dna/logs/generated_tensors_and_checkpoints
RESULT_DIR=hyena-dna/results
HYENA_ROOT=hyena-dna/upstream_hyena_dna
UPSTREAM_CKPT=hyena-dna/server_artifacts/upstream_checkpoints

FIVEMC_CKPT=${TENSOR_DIR}/hyenadna_small32k_chr16_overlap16k_full5000_5mc_only_lowlr_lastblock_20epochs_1000batches.best.pt
REG_CKPT=${RESULT_DIR}/checkpoints/hyenadna_reg_given_m_methyl_embedding_long.best.pt

for SPLIT in val test; do
  IN_NPZ=${TENSOR_DIR}/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.${SPLIT}.npz
  META=${TENSOR_DIR}/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.${SPLIT}.metadata.tsv
  MHAT_NPZ=${TENSOR_DIR}/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.${SPLIT}.mhat_5mc_from_pmd.all_cpg.npz

  $PY -u hyena-dna/evaluation/create_predicted_5mc_conditioned_npz.py \
    --hyena-root ${HYENA_ROOT} \
    --checkpoint-dir ${UPSTREAM_CKPT} \
    --npz ${IN_NPZ} \
    --checkpoint-5mc ${FIVEMC_CKPT} \
    --out-npz ${MHAT_NPZ} \
    --predicted-mask-mode all-cpg \
    --batch-size 1 \
    --max-length 32768 \
    --device cuda

  $PY -u hyena-dna/evaluation/evaluate_region_split_hyenadna_6ma_methyl_embedding_conditioned_nosample_overlap_aggregated.py \
    --hyena-root ${HYENA_ROOT} \
    --checkpoint-dir ${UPSTREAM_CKPT} \
    --npz ${MHAT_NPZ} \
    --metadata ${META} \
    --checkpoint ${REG_CKPT} \
    --split-name chr16_${SPLIT}_mhat \
    --out-prefix ${RESULT_DIR}/hyenadna_chain_pmd_to_reg_methyl_embedding_chr16_${SPLIT}_overlapagg \
    --batch-size 1 \
    --max-length 32768 \
    --device cuda
done
