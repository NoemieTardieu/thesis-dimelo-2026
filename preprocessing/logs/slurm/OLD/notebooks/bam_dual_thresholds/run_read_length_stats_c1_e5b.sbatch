#!/bin/bash
# Full-BAM read-length statistics for deciding how much data fits a 32k HyenaDNA window.

#SBATCH --job-name=readlen_32k
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-2
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-readlen-32k-%A_%a.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-readlen-32k-%A_%a.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/compute_bam_read_length_stats.py
OUTDIR=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/read_length_stats

mkdir -p "$OUTDIR"

if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  SAMPLE=merged_c1
  BAM=/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam
elif [[ "$SLURM_ARRAY_TASK_ID" == "2" ]]; then
  SAMPLE=merged_e5b
  BAM=/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam
else
  echo "Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

"$PY" "$SCRIPT" \
  --bam "$BAM" \
  --out-prefix "$OUTDIR/${SAMPLE}.full" \
  --length-cutoff 32000
