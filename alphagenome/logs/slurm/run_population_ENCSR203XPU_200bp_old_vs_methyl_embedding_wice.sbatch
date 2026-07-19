#!/bin/bash
#SBATCH --job-name=ENCSR203XPU_old_vs_membed
#SBATCH --clusters=wice
#SBATCH --account=lccbe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_project_clean/alphagenome/logs/slurm/slurm-ENCSR203XPU-old-vs-membed-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_project_clean/alphagenome/logs/slurm/slurm-ENCSR203XPU-old-vs-membed-%j.err

set -euo pipefail

PROJECT_DIR=/data/leuven/383/vsc38330/thesis_project_clean
ALPHA_DIR="${PROJECT_DIR}/alphagenome"
PYTHON_GPU=/scratch/leuven/383/vsc38330/.venv/bin/python
PYTHON_BENCH="${PROJECT_DIR}/server_artifacts/root_environment/venvs/thesis_env/bin/python"

EXTERNAL_BIGWIG="${ALPHA_DIR}/server_artifacts/external_tracks/ENCSR203XPU/ENCFF074PND_ENCSR203XPU_A549_H3K4me3_fold_change_GRCh38.bigWig"
REGIONS="${ALPHA_DIR}/metadata/regions/4chrom_test_regions.tsv"
ARCHIVE_200BP="${ALPHA_DIR}/archive/phase05_200bp_pre_encode_followup/benchmark_200bp"
ALPHAGENOME_200BP="${ARCHIVE_200BP}/alphagenome_test_200bp.tsv"
DIMELO_200BP="${ARCHIVE_200BP}/dimelo_test_200bp.tsv"
OLD_HYENA_200BP="${ARCHIVE_200BP}/hyenadna_test_200bp.tsv"
NEW_HYENA_200BP="${ALPHA_DIR}/results/model_comparison_inputs/hyenadna_methyl_embedding_test_200bp.tsv"

OUT_PARENT="${ALPHA_DIR}/results/population_track_benchmark_ENCSR203XPU_200bp_hyenadna_model_comparison"
OLD_OUT="${OUT_PARENT}/old_hyenadna"
NEW_OUT="${OUT_PARENT}/dna_plus_5mc_hyenadna"
COMPARE_OUT="${OUT_PARENT}/old_vs_dna_plus_5mc_hyenadna"

export PYTHONUNBUFFERED=1
export BASE_PYTHONPATH="${PROJECT_DIR}/alphagenome/src:${PYTHONPATH:-}"
export PYSAM_PYTHONPATH="${PROJECT_DIR}/server_artifacts/root_environment/venvs/thesis_env/lib/python3.13/site-packages:${BASE_PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${VSC_SCRATCH:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${OUT_PARENT}" "${ALPHA_DIR}/results/model_comparison_inputs"

cd "${PROJECT_DIR}"

echo "[1/4] Building final DNA+5mC HyenaDNA 200 bp population track"
if [[ -s "${NEW_HYENA_200BP}" ]]; then
  echo "Reusing existing ${NEW_HYENA_200BP}"
else
  env PYTHONPATH="${PYSAM_PYTHONPATH}" \
  "${PYTHON_GPU}" -u alphagenome/scripts/build_hyenadna_methyl_embedding_200bp_tracks.py \
    --split test \
    --regions "${REGIONS}" \
    --out "${NEW_HYENA_200BP}" \
    --device cuda
fi

test -s "${NEW_HYENA_200BP}"

echo "[2/4] Re-running population benchmark with old HyenaDNA track"
export PYTHONPATH="${BASE_PYTHONPATH}"
"${PYTHON_BENCH}" -u alphagenome/src/population_track_benchmark.py \
  --external-bigwig "${EXTERNAL_BIGWIG}" \
  --alphagenome "${ALPHAGENOME_200BP}" \
  --hyena "${OLD_HYENA_200BP}" \
  --dimelo "${DIMELO_200BP}" \
  --regions "${REGIONS}" \
  --out-dir "${OLD_OUT}" \
  --bin-size 200 \
  --bootstrap-replicates 2000 \
  --external-cell-type A549 \
  --external-assay-type "ENCODE ENCSR203XPU H3K4me3 ChIP-seq fold change over control" \
  --external-value-type "fold change over control" \
  --external-value-transform "ENCODE processed fold-change bigWig, combined biological replicates 1,2,3" \
  --alphagenome-cell-type A549 \
  --max-stacked-regions 12

echo "[3/4] Running population benchmark with final DNA+5mC HyenaDNA track"
"${PYTHON_BENCH}" -u alphagenome/src/population_track_benchmark.py \
  --external-bigwig "${EXTERNAL_BIGWIG}" \
  --alphagenome "${ALPHAGENOME_200BP}" \
  --hyena "${NEW_HYENA_200BP}" \
  --dimelo "${DIMELO_200BP}" \
  --regions "${REGIONS}" \
  --out-dir "${NEW_OUT}" \
  --bin-size 200 \
  --bootstrap-replicates 2000 \
  --external-cell-type A549 \
  --external-assay-type "ENCODE ENCSR203XPU H3K4me3 ChIP-seq fold change over control" \
  --external-value-type "fold change over control" \
  --external-value-transform "ENCODE processed fold-change bigWig, combined biological replicates 1,2,3" \
  --alphagenome-cell-type A549 \
  --max-stacked-regions 12

echo "[4/4] Creating old-vs-final HyenaDNA comparison tables and plots"
"${PYTHON_BENCH}" -u alphagenome/scripts/compare_hyenadna_population_models.py \
  --old-dir "${OLD_OUT}" \
  --new-dir "${NEW_OUT}" \
  --out-dir "${COMPARE_OUT}" \
  --max-regions 12

test -s "${OLD_OUT}/pairwise_metrics.tsv.gz"
test -s "${NEW_OUT}/pairwise_metrics.tsv.gz"
test -s "${COMPARE_OUT}/hyenadna_model_metric_comparison.tsv"
test -s "${COMPARE_OUT}/hyenadna_model_metric_deltas.tsv"

echo "Population benchmark old-vs-DNA+5mC HyenaDNA comparison complete:"
echo "  ${OUT_PARENT}"
