#!/bin/bash
# deepTools coverage QC for merged_c1 and merged_e5b.
# This checks whether the two BAMs have enough coverage/breadth for modeling.

#SBATCH --job-name=plotcov_c1_e5b
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-plotcov-c1-e5b-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-plotcov-c1-e5b-%j.err

set -euo pipefail

PLOTCOVERAGE=/data/leuven/383/vsc38330/.venv/bin/plotCoverage
OUTDIR=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/coverage_qc

BAM_C1=/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam
BAM_E5B=/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam

mkdir -p "$OUTDIR"

"$PLOTCOVERAGE" \
  -b "$BAM_C1" "$BAM_E5B" \
  --labels merged_c1 merged_e5b \
  --minMappingQuality 20 \
  --numberOfSamples 1000000 \
  --coverageThresholds 1 \
  --coverageThresholds 5 \
  --coverageThresholds 10 \
  --coverageThresholds 20 \
  --coverageThresholds 30 \
  --outCoverageMetrics "$OUTDIR/merged_c1_e5b.coverage_metrics.tsv" \
  --outRawCounts "$OUTDIR/merged_c1_e5b.coverage_raw_counts.tsv" \
  -o "$OUTDIR/merged_c1_e5b.coverage.png" \
  -p 4
