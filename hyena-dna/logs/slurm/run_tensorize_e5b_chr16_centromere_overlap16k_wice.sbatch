#!/bin/bash
# Tensorize chr16 centromere negative-control region for merged_e5b.
# Uses overlapping windows for reads >32 kb.

#SBATCH --job-name=chr16_cent_e5b_tensor
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-wice-chr16-centromere-e5b-overlap16k-tensor-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-wice-chr16-centromere-e5b-overlap16k-tensor-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
  --bam /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam \
  --extract-full /staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/by_chrom/extract_full_chr16.tsv.gz \
  --chrom chr16 \
  --regions-bed regions/chr16_centromere_36300000_46400000.bed \
  --max-length 32768 \
  --long-read-mode overlap \
  --window-stride 16384 \
  --max-reads 1000 \
  --out-prefix outputs/merged_e5b_chr16_centromere_overlap16k_first1000
