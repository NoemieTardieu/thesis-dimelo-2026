#!/bin/bash
# Check whether the chr16 centromere interval has reads and/or modkit signal in merged_e5b.
# First pass uses a read cap to avoid a very long full extract scan.

#SBATCH --job-name=chr16_cent_e5b
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-wice-chr16-centromere-e5b-signal-check-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-wice-chr16-centromere-e5b-signal-check-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

/data/leuven/383/vsc38330/.venv/bin/python check_region_read_signal.py \
  --bam /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam \
  --extract-full /staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/by_chrom/extract_full_chr16.tsv.gz \
  --chrom chr16 \
  --start 36300000 \
  --end 46400000 \
  --min-mapq 20 \
  --max-reads 5000 \
  --out-json outputs/merged_e5b_chr16_centromere_36300000_46400000_signal_check.max5000.json
