#!/bin/bash
#SBATCH --job-name=pair_chr16_c1
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch_sapphirerapids_long
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/to_delete/slurm/slurm-paired-chr16-c1-expanded-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/to_delete/slurm/slurm-paired-chr16-c1-expanded-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}

cd /data/leuven/383/vsc38330/thesis_project_clean

/lustre1/scratch/383/vsc38330/.venv/bin/python -u hyena-dna/analysis/paired_read_interpretability_analysis.py \
  --dataset hyena-dna/to_delete/generated_tensors_and_checkpoints/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz hyena-dna/OLD/outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv \
  --bam /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam \
  --sample merged_c1 \
  --chrom chr16 \
  --locus-variance hyena-dna/OLD/outputs/full_chr16_c1_locus_variance.per_locus_variance.tsv.gz \
  --reference-fasta /data/leuven/383/vsc38330/thesis_project_clean/preprocessing/server_artifacts/original_thesis_dimelo_residual/src/data/hg38.fa \
  --out-prefix hyena-dna/results/chr16_c1_paired_read_interpretability_expanded \
  --top-loci 1500 \
  --max-candidate-loci 250 \
  --max-reads-per-locus 12 \
  --max-pairs-per-locus 100 \
  --window-size 500 \
  --bin-size 50 \
  --min-reads-locus 8 \
  --min-window-coverage-fraction 0.5 \
  --min-pair-comparable-fraction 0.5 \
  --min-valid-6ma-positions 8 \
  --min-valid-5mc-positions 1 \
  --max-missing-signal-fraction 0.97 \
  --min-shared-bins 2 \
  --top-n-per-case 3
