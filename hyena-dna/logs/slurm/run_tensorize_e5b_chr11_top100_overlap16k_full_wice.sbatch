#!/bin/bash
# Full overlap-window tensorization for merged_e5b chr11 top100 regions.

#SBATCH --job-name=chr11_e5b_ovlp_full
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-wice-chr11-e5b-top100-overlap16k-full-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-wice-chr11-e5b-top100-overlap16k-full-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
  --bam /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam \
  --extract-full /staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/by_chunk_merged_per_chrom/extract_full_chr11.tsv.gz \
  --chrom chr11 \
  --regions-bed regions/merged_e5b_chr11_selected_100kb_top100.bed \
  --max-length 32768 \
  --long-read-mode overlap \
  --window-stride 16384 \
  --max-reads-per-region 50 \
  --max-reads 5000 \
  --out-prefix outputs/merged_e5b_chr11_selected_top100_overlap16k_full5000
