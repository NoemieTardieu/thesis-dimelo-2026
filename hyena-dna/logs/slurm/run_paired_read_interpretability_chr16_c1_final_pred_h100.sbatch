#!/bin/bash
#SBATCH --job-name=paired_chr16_final
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-paired-chr16-final-pred-h100-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-paired-chr16-final-pred-h100-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/lustre1/scratch/383/vsc38330/.venv/bin/python -u paired_read_interpretability_analysis.py \
  --dataset outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --bam /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam \
  --sample merged_c1 \
  --chrom chr16 \
  --locus-variance outputs/full_chr16_c1_locus_variance.per_locus_variance.tsv.gz \
  --reference-fasta /data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa \
  --reg-checkpoint outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_lastblock_25epochs_1000batches.best.pt \
  --device cuda \
  --top-loci 500 \
  --max-candidate-loci 80 \
  --max-reads-per-locus 18 \
  --max-pairs-per-locus 80 \
  --window-size 500 \
  --bin-size 50 \
  --min-reads-locus 8 \
  --min-window-coverage-fraction 0.45 \
  --min-pair-comparable-fraction 0.45 \
  --min-valid-6ma-positions 6 \
  --min-valid-5mc-positions 1 \
  --max-missing-signal-fraction 0.98 \
  --min-valid-per-bin 1 \
  --min-shared-bins 1 \
  --low-quantile 0.10 \
  --high-quantile 0.90 \
  --top-n-per-case 3 \
  --flag-substitution-distance 0.10 \
  --seed 7 \
  --out-prefix outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256
