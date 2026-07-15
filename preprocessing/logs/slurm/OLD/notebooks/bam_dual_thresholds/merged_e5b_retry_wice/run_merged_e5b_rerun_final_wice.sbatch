#!/bin/bash
#SBATCH --job-name=merged_e5b_rerun_final
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc38330
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-16%2
#SBATCH --output=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/slurm-rerun-final-%A_%a.out
#SBATCH --error=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/slurm-rerun-final-%A_%a.err

set -euo pipefail
source /data/leuven/383/vsc38330/.venv/bin/activate
export PATH="/data/leuven/383/vsc38330/bin/dist_modkit_v0.6.1_481e3c9:$PATH"

BAM="/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam"
REF="/data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa"
INTERVALS="/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/merged_e5b_retry_wice/rerun_regions_final.txt"
OUTDIR="/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/by_chunk"

REGION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$INTERVALS")
SAFE_REGION=$(echo "$REGION" | tr ':-' '__')
mkdir -p "$OUTDIR"

/data/leuven/383/vsc38330/bin/dist_modkit_v0.6.1_481e3c9/modkit extract full   "$BAM"   "$OUTDIR/extract_full_${SAFE_REGION}.tsv.gz"   --bgzf   --region "$REGION"   --ref "$REF"
