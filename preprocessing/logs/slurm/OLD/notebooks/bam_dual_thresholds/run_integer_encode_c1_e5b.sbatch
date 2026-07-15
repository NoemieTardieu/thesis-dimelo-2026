#!/bin/bash
#SBATCH --job-name=cnn_int_encode
#SBATCH --clusters=genius
#SBATCH --account=lccbe
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-int-encode-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds/slurm-cnn-int-encode-%j.err

set -euo pipefail

PY=/data/leuven/383/vsc38330/.venv/bin/python
SCRIPT=/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/scripts/encode_cnn_kmers_to_npz.py

TRAIN_IN=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/merged_c1_C_m_q0p8_training.valid5mer.uppercase.cnn_features.tsv.gz
VAL_IN=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/merged_e5b_C_m_q0p8_validation.valid5mer.uppercase.cnn_features.tsv.gz

TRAIN_OUT=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/tensors_integer_valid5mer_uppercase
VAL_OUT=/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/tensors_integer_valid5mer_uppercase

$PY $SCRIPT --input "$TRAIN_IN" --output-dir "$TRAIN_OUT" --chunk-size 2000000
$PY $SCRIPT --input "$VAL_IN" --output-dir "$VAL_OUT" --chunk-size 2000000
