#!/bin/bash
# Tensorize merged_e5b on selected chr17 top100 regions.
# Requires extract_full_chr17.tsv.gz in merged_e5b/by_chunk_merged_per_chrom.

#SBATCH --job-name=chr17_e5b_t100_e50
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-wice-chr17-e5b-top100-even50-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-wice-chr17-e5b-top100-even50-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
  --bam /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam \
  --extract-full /staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/by_chunk_merged_per_chrom/extract_full_chr17.tsv.gz \
  --chrom chr17 \
  --regions-bed regions/merged_e5b_chr17_selected_100kb_top100.bed \
  --max-reads-per-region 50 \
  --max-reads 5000 \
  --out-prefix outputs/merged_e5b_chr17_selected_top100_even50
