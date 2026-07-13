#!/usr/bin/env bash
set -euo pipefail

source /data/leuven/383/vsc38330/venvs/thesis_env/bin/activate

base="/data/leuven/383/vsc38330/thesis_dimelo/modkit_visualization"
mkdir -p "$base/read_level" "$base/profiles" "$base/stratifications"

python "$base/plot_modkit_read_level_figures.py" \
  --outdir "$base/read_level"

python "$base/plot_bigwig_average_profile.py" \
  --regions-bed /scratch/leuven/383/vsc38330/promoters_2kb_clean.bed \
  --track "h3k27ac mCpG raw=/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/genomic_context/signal_tracks_200bp/h3k27ac.meth_c_meth_frac.bw" \
  --track "h3k27ac CpG-only=/data/leuven/383/vsc38330/thesis_dimelo/igv_subsets/h3k27ac_CpG_methylation_percent.bw" \
  --out "$base/profiles/h3k27ac_raw_vs_cpg_promoters2kb.png" \
  --title "h3k27ac: raw C-channel vs CpG-only methylation around promoters" \
  --ylabel "Fraction methylated" \
  --upstream 5000 --downstream 5000 --bin-size 200 --max-regions 20000

python "$base/plot_bigwig_average_profile.py" \
  --regions-bed /scratch/leuven/383/vsc38330/TSS.bed \
  --track "h3k27ac mCpG CpG-only=/data/leuven/383/vsc38330/thesis_dimelo/igv_subsets/h3k27ac_CpG_methylation_percent.bw" \
  --out "$base/profiles/h3k27ac_cpg_tss.png" \
  --title "h3k27ac CpG methylation around TSS" \
  --ylabel "Fraction methylated" \
  --upstream 5000 --downstream 5000 --bin-size 200 --max-regions 20000

python "$base/plot_bigwig_average_profile.py" \
  --regions-bed /scratch/leuven/383/vsc38330/TSS.bed \
  --track "h3k27ac=/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/genomic_context/signal_tracks_200bp_amod_frac/h3k27ac.a_meth_frac.bw" \
  --track "h3k27me3=/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/genomic_context/signal_tracks_200bp_amod_frac/h3k27me3.a_meth_frac.bw" \
  --track "h3k4me3=/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/genomic_context/signal_tracks_200bp_amod_frac/h3k4me3.a_meth_frac.bw" \
  --out "$base/profiles/cross_mark_A_signal_tss.png" \
  --title "Cross-mark A-signal around TSS" \
  --ylabel "A modified fraction" \
  --upstream 2000 --downstream 2000 --bin-size 200 --max-regions 20000

python "$base/plot_bigwig_profile_by_region_sets.py" \
  --bigwig /data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/genomic_context/signal_tracks_200bp_amod_frac/h3k27ac.a_meth_frac.bw \
  --regions "CGI-overlap promoters=/scratch/leuven/383/vsc38330/promoters_2kb_CGI_overlap.bed" \
  --regions "CGI-nonoverlap promoters=/scratch/leuven/383/vsc38330/promoters_2kb_CGI_nonoverlap.bed" \
  --out "$base/stratifications/h3k27ac_a_signal_cgi_vs_noncgi_promoters.png" \
  --title "h3k27ac A-signal: CGI vs non-CGI promoters" \
  --ylabel "A modified fraction" \
  --upstream 5000 --downstream 5000 --bin-size 200 --max-regions 20000

python "$base/plot_bigwig_profile_by_region_sets.py" \
  --bigwig /data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/genomic_context/signal_tracks_200bp_amod_frac/h3k27ac.a_meth_frac.bw \
  --regions "All clean promoters=/scratch/leuven/383/vsc38330/promoters_2kb_clean.bed" \
  --regions "HOX loci=/scratch/leuven/383/vsc38330/hox_loci_plus20kb.bed" \
  --regions "HOX gene bodies=/scratch/leuven/383/vsc38330/hox_gene_bodies.bed" \
  --out "$base/stratifications/h3k27ac_a_signal_promoters_vs_hox.png" \
  --title "h3k27ac A-signal across promoter and HOX contexts" \
  --ylabel "A modified fraction" \
  --upstream 5000 --downstream 5000 --bin-size 200 --max-regions 20000
