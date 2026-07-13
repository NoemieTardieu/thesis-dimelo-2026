#!/usr/bin/env python3
from __future__ import annotations

import csv
import fnmatch
import os
import shutil
from pathlib import Path


ROOT = Path("/data/leuven/383/vsc38330")
CLEAN = ROOT / "thesis_project_clean"
PRE = ROOT / "thesis_dimelo"
HYENA = ROOT / "hyena-dna-main"
HYENA_WORK = HYENA / "preprocessing_chr16_merged_e5b"
ALPHA = ROOT / "alphagenome"

HEAVY_EXTS = {
    ".bam",
    ".bai",
    ".cram",
    ".crai",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".bigWig",
    ".bw",
}

rows_moved: list[dict[str, str]] = []
rows_old: list[dict[str, str]] = []
rows_delete: list[dict[str, str]] = []


def ensure_dirs() -> None:
    dirs = [
        CLEAN / "preprocessing" / "OLD",
        CLEAN / "preprocessing" / "to_delete",
        CLEAN / "hyena-dna" / "OLD",
        CLEAN / "hyena-dna" / "to_delete",
        CLEAN / "alphagenome" / "OLD",
        CLEAN / "alphagenome" / "to_delete",
        CLEAN / "docs",
        CLEAN / "configs",
        CLEAN / "metadata",
        CLEAN / "inventories",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}.moved{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def classify_manifest(dest: Path) -> list[dict[str, str]]:
    parts = dest.parts
    if "to_delete" in parts:
        return rows_delete
    if "OLD" in parts:
        return rows_old
    return rows_moved


def move_path(src: Path, dest: Path, reason: str, category: str) -> bool:
    if not src.exists():
        return False
    dest = unique_dest(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    row = {
        "source": str(src),
        "destination": str(dest),
        "category": category,
        "reason": reason,
    }
    classify_manifest(dest).append(row)
    return True


def move_file(src: Path, dest_dir: Path, reason: str, category: str | None = None) -> bool:
    category = category or dest_dir.name
    return move_path(src, dest_dir / src.name, reason, category)


def move_tree(src: Path, dest: Path, reason: str, category: str) -> bool:
    return move_path(src, dest, reason, category)


def move_glob(base: Path, pattern: str, dest_dir: Path, reason: str, category: str) -> None:
    for src in sorted(base.glob(pattern)):
        if src.exists():
            move_file(src, dest_dir, reason, category)


def move_matching_files(
    base: Path,
    dest_base: Path,
    include: list[str],
    reason: str,
    category: str,
    exclude: list[str] | None = None,
) -> None:
    exclude = exclude or []
    if not base.exists():
        return
    for src in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = src.relative_to(base)
        rel_s = rel.as_posix()
        if any(fnmatch.fnmatch(rel_s, pat) for pat in exclude):
            continue
        if any(fnmatch.fnmatch(rel_s, pat) for pat in include):
            move_path(src, dest_base / rel, reason, category)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "destination", "category", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def move_preprocessing() -> None:
    pillar = CLEAN / "preprocessing"
    active = pillar / "src"
    docs = pillar / "docs"
    metadata = pillar / "metadata"
    results = pillar / "results"

    for name in [
        "PREPROCESSING_RESTRUCTURING_CONTEXT.md",
        "THESIS_WORKFLOW_UP_TO_PHASE2.txt",
        "THESIS_WORKFLOW_REVISED_AFTER_FEEDBACK.txt",
        "PHASE1_QC_AND_VISUAL_VALIDATION_SUBSECTION_DRAFT.md",
        "PHASE2_LONG_CONTEXT_SUBSECTION_DRAFT.md",
        "THESIS_WORKFLOW_REMAINING_PHASES_PLANNING.txt",
        "PHASE3_PREPARATION_CHECKLIST.txt",
        "THESIS_WORKFLOW_GENOMIC_CONTEXT_STRATIFICATION.txt",
        "THESIS_WORKFLOW_HISTONE_MARK_INTERPLAY_AND_CPG_CONTEXT.txt",
        "THESIS_WORKFLOW_EXTRA_DEPENDENCE_VALIDATION_FIGURES.txt",
    ]:
        move_file(PRE / name, docs, "preprocessing documentation and thesis notes", "active_docs")

    phase1_dir = PRE / "src" / "preprocessing"
    move_file(phase1_dir / "phase1_dimelo_qc.py", active / "phase1_qc", "final Phase 1 QC script", "active_code")
    move_file(phase1_dir / "phase1_qc_summary.py", active / "phase1_qc", "final Phase 1 QC summary script", "active_code")
    move_tree(
        phase1_dir / "phase_1_visualization",
        active / "visualization" / "phase_1_visualization",
        "Phase 1 exploratory visualizations kept as reproducibility code/results",
        "active_code",
    )
    move_tree(
        phase1_dir / "phase1_output" / "summary_report",
        results / "phase1_summary_report",
        "lightweight Phase 1 summary report outputs",
        "active_results",
    )

    phase2 = phase1_dir / "phase2_long_context"
    for name in [
        "generate_long_context_intervals.py",
        "build_methylation_interval_backend.py",
        "methylation_interval_store.py",
        "run_backend_all_marks.py",
        "summarize_backend_manifests.py",
        "validate_backend_integrity.py",
        "windowed_oe_enrichment_heatmap.py",
        "phase2_joint_overlay_plots.py",
        "plot_cpg_methylation_by_mark.py",
        "plot_cpg_fraction_vs_cut_distance.py",
        "export_igv_tracks.py",
        "README.md",
        "intervals_long_context.tsv",
    ]:
        dest = active / "phase2_long_context"
        if name.endswith(".tsv"):
            dest = metadata
        move_file(phase2 / name, dest, "final Phase 2 long-context backend code or metadata", "active_code")

    move_matching_files(
        phase2 / "backends_all_marks",
        metadata / "phase2_backends_all_marks",
        ["backend_*_C/manifest_*_C.tsv", "summary/*.tsv", "summary/*.md"],
        "lightweight Phase 2 backend manifests and QC summaries",
        "active_metadata",
    )
    move_matching_files(
        phase2,
        results / "phase2_figures_and_summaries",
        [
            "windowed_enrichment_*/*.svg",
            "windowed_enrichment_*/*.tsv",
            "windowed_enrichment_*/*.md",
            "windowed_joint_plots_*/*.svg",
            "windowed_joint_plots_*/*.md",
            "cpg_methylation_by_mark/*.tsv",
            "cpg_methylation_by_mark/*.svg",
            "cpg_methylation_by_mark/*.png",
            "cpg_fraction_vs_cut_distance/*.tsv",
            "cpg_fraction_vs_cut_distance/*.svg",
            "cpg_fraction_vs_cut_distance/*.png",
        ],
        "lightweight Phase 2 interpretation figures and summaries",
        "active_results",
        exclude=["**/*.npz"],
    )

    phase3 = phase1_dir / "phase3_hyena"
    move_file(phase3 / "build_phase3_training_index.py", active / "phase3_hyena", "final Phase 3 HyenaDNA index builder", "active_code")
    move_file(phase3 / "phase3_training_index.tsv", metadata, "final Phase 3 training index", "active_metadata")

    move_tree(PRE / "train", pillar / "to_delete" / "cnn_legacy" / "train", "CNN/5-mer training path not used in final thesis", "to_delete")
    move_tree(PRE / "src" / "train", pillar / "to_delete" / "cnn_legacy" / "src_train", "CNN/5-mer training code not used in final thesis", "to_delete")
    move_tree(PRE / "src" / "preprocessing_2", pillar / "to_delete" / "unused_preprocessing_2", "old preprocessing route not used in final thesis", "to_delete")
    move_tree(PRE / "modkit_preproc", pillar / "to_delete" / "local_modkit_preproc_outputs", "local old generated modkit preprocessing outputs; external extract-full files remain on server", "to_delete")
    move_tree(PRE / "modkit_visualization", pillar / "OLD" / "modkit_visualization", "historical modkit visualization code kept separate from final pipeline", "OLD")
    move_tree(PRE / "notebooks", pillar / "OLD" / "notebooks", "exploratory notebooks and helper scripts", "OLD")
    move_tree(PRE / "src" / "modeling_prep", pillar / "OLD" / "modeling_prep", "earlier modeling-prep experiments kept for reference", "OLD")
    move_tree(PRE / "src" / "eval", pillar / "OLD" / "eval", "earlier evaluation code kept for reference", "OLD")
    move_tree(PRE / "src" / "models", pillar / "OLD" / "models", "earlier model code kept for reference", "OLD")

    move_glob(PRE, "*.sbatch", pillar / "to_delete" / "slurm", "SLURM job file removed from active repository", "to_delete")
    move_glob(PRE, "slurm-*.out", pillar / "to_delete" / "slurm", "SLURM output log removed from active repository", "to_delete")
    move_glob(PRE, "slurm-*.err", pillar / "to_delete" / "slurm", "SLURM error log removed from active repository", "to_delete")


def move_hyena() -> None:
    pillar = CLEAN / "hyena-dna"
    if (HYENA / "HYENADNA_RESTRUCTURING_CONTEXT.md").exists():
        move_file(HYENA / "HYENADNA_RESTRUCTURING_CONTEXT.md", pillar / "docs", "HyenaDNA restructuring context", "active_docs")
    if (HYENA_WORK / "README.txt").exists():
        move_file(HYENA_WORK / "README.txt", pillar / "docs", "original HyenaDNA thesis work README", "active_docs")

    groups = {
        "preprocessing": [
            "make_chr16_dimelo_tensors.py",
            "create_region_split_tensors.py",
            "combine_sample_split_tensors.py",
            "combine_chrom_split_tensors.py",
            "select_chr16_training_regions.py",
        ],
        "models": [
            "train_region_split_hyenadna_5mc_only.py",
            "train_region_split_hyenadna_6ma_methyl_conditioned_nosample.py",
        ],
        "evaluation": [
            "evaluate_region_split_hyenadna_5mc_only_overlap_aggregated.py",
            "evaluate_region_split_hyenadna_6ma_methyl_conditioned_nosample_overlap_aggregated.py",
            "evaluate_reg_given_m_threshold_metrics.py",
        ],
        "analysis": [
            "analyze_full_chromosome_locus_variance.py",
            "verify_locus_variance_thresholds.py",
            "analyze_variance_loci_cigar_dna.py",
            "analyze_final_m_reg_relationship.py",
            "select_locus_read_pair_interpretability.py",
            "paired_read_interpretability_analysis.py",
            "summarize_paired_read_interpretability_runs.py",
            "check_region_read_signal.py",
        ],
        "plotting": [
            "plot_hyenadna_epoch_losses.py",
            "plot_reg_given_m_roc_pr_curves.py",
            "plot_reg_given_m_crosschrom_auroc_auprc.py",
        ],
    }
    for subdir, names in groups.items():
        for name in names:
            move_file(HYENA_WORK / name, pillar / subdir, "final HyenaDNA thesis code", "active_code")

    # Superseded or exploratory scripts.
    for src in sorted(HYENA_WORK.glob("*.py")):
        name = src.name
        lower = name.lower()
        if any(token in lower for token in ["debug", "tiny", "sample_conditioned", "two_head", "smoke"]):
            move_file(src, pillar / "OLD" / "scripts", "superseded/debug/sample-conditioned HyenaDNA script", "OLD")
        elif "6ma_methyl_conditioned.py" in lower or "extended" in lower:
            move_file(src, pillar / "OLD" / "scripts", "superseded HyenaDNA model/evaluation script", "OLD")
        else:
            move_file(src, pillar / "OLD" / "scripts", "unclassified HyenaDNA thesis script kept for review", "OLD")

    move_glob(HYENA_WORK, "*.sbatch", pillar / "to_delete" / "slurm", "SLURM job file removed from active repository", "to_delete")
    move_glob(HYENA_WORK, "slurm-*.out", pillar / "to_delete" / "slurm", "SLURM output log removed from active repository", "to_delete")
    move_glob(HYENA_WORK, "slurm-*.err", pillar / "to_delete" / "slurm", "SLURM error log removed from active repository", "to_delete")
    move_glob(HYENA_WORK, "*.out", pillar / "to_delete" / "slurm", "cluster output log removed from active repository", "to_delete")
    move_glob(HYENA_WORK, "*.err", pillar / "to_delete" / "slurm", "cluster error log removed from active repository", "to_delete")

    outputs = HYENA_WORK / "outputs"
    move_matching_files(
        outputs,
        pillar / "results",
        [
            "reg_given_m_crosschrom_*",
            "hyenadna_small32k_chr16trained_reg_given_m_nosample_mlp256_lowlr_on_chr*_test_overlapagg.summary.json",
            "hyenadna_small32k_chr16trained_reg_given_m_nosample_mlp256_lowlr_on_chr*_test_overlapagg.per_region_metrics.tsv",
            "hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_best_*_overlapagg.summary.json",
            "hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_best_*_overlapagg.per_region_metrics.tsv",
            "hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_threshold_metrics.*",
            "final_m_reg_relationship_chr16_val_test_overlapagg.*summary.tsv",
            "final_m_reg_relationship_chr16_val_test_overlapagg.*cooccurrence.tsv",
            "final_m_reg_relationship_chr16_val_test_overlapagg.hexbin.*",
            "final_m_reg_relationship_4chrom_test_overlapagg.*summary.tsv",
            "final_m_reg_relationship_4chrom_test_overlapagg.*cooccurrence.tsv",
            "chr16_c1_read_pair_interpretability_with_reg_predictions_min6ma001.*",
            "chr16_c1_paired_read_interpretability_observed_medium.selected_pairs.tsv",
            "chr16_c1_paired_read_interpretability_observed_medium.report.md",
            "chr16_c1_paired_read_interpretability_observed_medium.summary.json",
            "chr16_c1_paired_read_interpretability_observed_medium.params.json",
            "chr16_c1_paired_read_interpretability_observed_medium.plot*.png",
            "chr16_c1_paired_read_interpretability_observed_medium.plot*.svg",
            "chr16_c1_paired_read_interpretability_observed_medium.pair_*.png",
            "chr16_c1_paired_read_interpretability_observed_medium.pair_*.svg",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.selected_pairs.tsv",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.report.md",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.summary.json",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.params.json",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.within_locus_associations.tsv",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.selected_pair_position_audit.tsv",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.plot*.png",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.plot*.svg",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.pair_*.png",
            "chr16_c1_paired_read_interpretability_final_pred_mlp256.pair_*.svg",
            "full_chr*_c1_locus_variance.summary.json",
            "full_chr*_c1_locus_variance.variance_plots.png",
            "full_chr16_c1_locus_variance_threshold_audit.*.tsv",
            "full_chr16_c1_locus_variance_threshold_audit.*.png",
            "full_chr16_c1_locus_variance_threshold_audit.*.svg",
        ],
        "final lightweight HyenaDNA results and thesis figures",
        "active_results",
        exclude=["*.pt", "*.npz"],
    )
    move_matching_files(
        outputs,
        pillar / "results" / "checkpoints",
        [
            "hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_lastblock_25epochs_1000batches.best.pt",
            "hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_posw2_focal1_lastblock_22epochs_1000batches.best.pt",
        ],
        "final large checkpoints retained on server and ignored by Git",
        "active_large_result",
    )
    move_matching_files(outputs, pillar / "to_delete" / "generated_tensors_and_checkpoints", ["*.npz", "*.pt"], "large generated tensors/checkpoints excluded from GitHub active tree", "to_delete")
    move_matching_files(outputs, pillar / "OLD" / "outputs", ["*"], "remaining HyenaDNA outputs kept for review", "OLD")
    move_tree(HYENA_WORK / "regions", pillar / "metadata" / "regions", "HyenaDNA selected region metadata", "active_metadata")


def move_alphagenome() -> None:
    pillar = CLEAN / "alphagenome"
    src_dir = pillar / "src"
    scripts_dir = pillar / "scripts"

    core = [
        "__init__.py",
        "benchmark_utils.py",
        "alphagenome_query.py",
        "reference_tracks.py",
        "population_track_benchmark.py",
        "rebin_alphagenome_to_200bp.py",
        "external_bigwig_benchmark.py",
    ]
    scripts = [
        "make_4chrom_test_regions.py",
        "inspect_alphagenome_h3k4me3_tracks.py",
        "run_alphagenome_4chrom.py",
        "export_alphagenome_128bp_tracks.py",
        "build_dimelo_128bp_tracks.py",
        "build_hyenadna_128bp_tracks.py",
        "build_alphagenome_readseq_aggregate.py",
        "evaluate_readseq_alphagenome_200bp.py",
        "evaluate_alphagenome_readlevel.py",
        "diagnose_alphagenome_readlevel_failure.py",
        "make_readlevel_locus_example.py",
        "plot_readseq_alphagenome_normalized.py",
        "add_peak_agreement_example.py",
        "make_thesis_correlation_heatmap_panel.py",
        "make_thesis_density_three_panel.py",
    ]
    old_scripts = [
        "run_alphagenome_smoke.py",
        "fit_validation_thresholds.py",
        "evaluate_alphagenome_vs_dimelo.py",
        "plot_alphagenome_comparison.py",
        "build_plot_5mc_hyenadna.py",
        "promoter_followup_analysis.py",
        "summarize_promoter_followup_metrics.py",
        "run_200bp_followup.py",
        "alphagenome_read_sequence_sensitivity.py",
    ]
    for name in core:
        move_file(ALPHA / name, src_dir, "final AlphaGenome reusable module", "active_code")
    for name in scripts:
        move_file(ALPHA / name, scripts_dir, "final AlphaGenome workflow/figure script", "active_code")
    for name in old_scripts:
        move_file(ALPHA / name, pillar / "OLD" / "scripts", "older/superseded AlphaGenome script kept for review", "OLD")

    for name in ["README.md", "ALPHAGENOME_RESTRUCTURING_CONTEXT.md", "requirements.txt"]:
        dest = pillar / "docs" if name.endswith(".md") else pillar
        move_file(ALPHA / name, dest, "AlphaGenome documentation or requirements", "active_docs")

    move_tree(ALPHA / "tests", pillar / "tests", "AlphaGenome lightweight tests", "active_tests")
    move_glob(ALPHA, "*.sbatch", pillar / "to_delete" / "slurm", "SLURM job file removed from active repository", "to_delete")
    move_glob(ALPHA, "slurm-*.out", pillar / "to_delete" / "slurm", "SLURM output log removed from active repository", "to_delete")
    move_glob(ALPHA, "slurm-*.err", pillar / "to_delete" / "slurm", "SLURM error log removed from active repository", "to_delete")

    move_tree(ALPHA / "cache", pillar / "to_delete" / "cache" / "cache", "AlphaGenome cache excluded from GitHub", "to_delete")
    move_tree(ALPHA / "cache_readseq", pillar / "to_delete" / "cache" / "cache_readseq", "AlphaGenome read sequence cache excluded from GitHub", "to_delete")
    move_tree(ALPHA / "cache_readseq_10reads", pillar / "to_delete" / "cache" / "cache_readseq_10reads", "AlphaGenome read sequence cache excluded from GitHub", "to_delete")
    move_tree(ALPHA / "__pycache__", pillar / "to_delete" / "python_cache" / "__pycache__", "Python cache excluded from GitHub", "to_delete")

    move_matching_files(
        ALPHA / "outputs" / "metadata",
        pillar / "metadata",
        ["selected_a549_h3k4me3_tracks.tsv"],
        "selected AlphaGenome track metadata",
        "active_metadata",
    )
    final_out = ALPHA / "outputs" / "population_track_benchmark_ENCSR203XPU_200bp_final"
    move_matching_files(
        final_out,
        pillar / "results" / "population_track_benchmark_ENCSR203XPU_200bp_final",
        [
            "text_summary.md",
            "config.json",
            "pairwise_metrics.tsv.gz",
            "chromosome_specific_metrics.tsv.gz",
            "coverage_sensitivity.tsv.gz",
            "weighted_correlations.tsv.gz",
            "normalization_parameters.tsv",
            "validated_track_metadata.tsv",
            "representative_loci.tsv.gz",
            "figures/**/*.png",
            "figures/**/*.svg",
            "figures/**/*.pdf",
        ],
        "final ENCSR203XPU 200 bp benchmark summaries and figures",
        "active_results",
        exclude=["canonical_raw_bins.tsv.gz", "canonical_normalized_bins.tsv.gz"],
    )
    read_example = ALPHA / "outputs" / "alphagenome_readlevel_locus_example"
    move_matching_files(
        read_example,
        pillar / "results" / "alphagenome_readlevel_locus_example",
        ["*.png", "*.md", "*.tsv", "*.json"],
        "final same-locus read-level AlphaGenome example",
        "active_results",
    )
    move_matching_files(
        ALPHA / "outputs",
        pillar / "results" / "readseq_summaries",
        [
            "alphagenome_readseq_200bp_10reads.summary.json",
            "alphagenome_readseq_200bp_10reads_benchmark.summary.tsv",
            "alphagenome_readlevel_10reads*",
        ],
        "AlphaGenome ONT-read sequence summary outputs",
        "active_results",
        exclude=["*.tsv"],
    )

    move_tree(ALPHA / "outputs" / "population_track_benchmark_GSM2421502_1kb", pillar / "OLD" / "outputs" / "population_track_benchmark_GSM2421502_1kb", "older 1 kb external benchmark kept for reference", "OLD")
    for d in [
        "benchmark_200bp",
        "benchmark_200bp_smoke",
        "external_bigwig_benchmark_GSE91218",
        "external_bigwig_benchmark_GSM2421502",
        "population_track_benchmark_ENCSR203XPU_200bp",
        "population_track_benchmark_ENCSR203XPU_200bp_quick",
        "population_track_benchmark_GSM2421502_dryrun",
        "promoter_followup",
        "read_sequence_sensitivity",
        "plots",
        "5mc_visual",
        "alphagenome_readlevel_diagnostics",
        "alphagenome_readseq_200bp_10reads_normalized_plots",
    ]:
        move_tree(ALPHA / "outputs" / d, pillar / "OLD" / "outputs" / d, "older/exploratory AlphaGenome output kept for review", "OLD")
    move_matching_files(ALPHA / "outputs", pillar / "OLD" / "outputs", ["*"], "remaining AlphaGenome outputs kept for review", "OLD")
    move_tree(ALPHA / "external_tracks", pillar / "metadata" / "external_tracks", "external track metadata and files kept on server; large BigWigs ignored by Git", "active_metadata")


def write_project_files() -> None:
    gitignore = """# secrets
.env
env
*.key
*API_KEY*

# raw/genomic data and generated arrays
*.bam
*.bai
*.cram
*.crai
*.bigWig
*.bw
*.npz
*.npy

# model checkpoints
*.pt
*.pth
checkpoints/

# generated heavy tables
canonical_raw_bins.tsv.gz
canonical_normalized_bins.tsv.gz
all_read_bin_values.tsv
alphagenome_readseq_*.tsv
*.per_locus_variance.tsv.gz
*.all_pairs.tsv

# AlphaGenome and runtime caches
cache*/
**/cache*/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ipynb_checkpoints/

# SLURM / cluster logs
*.sbatch
slurm-*.out
slurm-*.err
*.out
*.err
"""
    write_text(CLEAN / ".gitignore", gitignore)
    write_text(
        CLEAN / "README.md",
        """# Thesis DiMeLo Project

Clean GitHub-ready thesis workspace organized into three pillars:

- `preprocessing/`: final DiMeLo preprocessing, long-context interval backends, and HyenaDNA index preparation.
- `hyena-dna/`: thesis-specific HyenaDNA tensor generation, modeling, evaluation, and interpretation.
- `alphagenome/`: AlphaGenome reference/read-sequence analyses and population-level benchmark.

Large data, caches, tensors, BAMs, BigWigs, and checkpoints are retained on the server but ignored for Git. See `metadata/external_data_registry.tsv` and the pillar READMEs.
""",
    )
    write_text(
        CLEAN / "preprocessing" / "README.md",
        """# Preprocessing Pillar

Active workflow:

1. Phase 1 QC parses DiMeLo modBAM modified-base tags and summarizes read-level/binned signal.
2. Phase 2 builds 1 Mb long-context interval backends and mark-specific manifests.
3. Phase 3 merges backend manifests into the HyenaDNA-facing training index.

`OLD/` contains exploratory code kept for reference. `to_delete/` is a quarantine for unused old preprocessing and CNN-era work; nothing has been permanently deleted.
""",
    )
    write_text(
        CLEAN / "hyena-dna" / "README.md",
        """# HyenaDNA Pillar

Active thesis direction:

- `P(M | D)` predicts CpG 5mC from sequence.
- `P(Reg | D, M)` predicts DiMeLo 6mA regulatory signal from DNA plus methylation context.

Final evaluation uses overlap aggregation by sample/read/read-position and includes cross-chromosome generalization, threshold metrics, variance analysis, and paired-read interpretability.

Large tensors and checkpoints are retained on the server and ignored by Git.
""",
    )
    write_text(
        CLEAN / "alphagenome" / "README.md",
        """# AlphaGenome Pillar

Active final analysis is the ENCSR203XPU A549 H3K4me3 200 bp population-track benchmark, plus complementary ONT-read-sequence/read-level AlphaGenome checks.

Keep scripts, configs, metadata, small summaries, and thesis-ready figures in the active tree. Caches, older benchmarks, smoke runs, SLURM files, and exploratory outputs are isolated in `OLD/` or `to_delete/`.
""",
    )
    write_text(
        CLEAN / "metadata" / "external_data_registry.tsv",
        """pillar\trole\tpath\tnote
preprocessing\traw mark-specific modBAM\t/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam\tserver-resident input
preprocessing\traw mark-specific modBAM\t/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam\tserver-resident input
preprocessing\traw mark-specific modBAM\t/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam\tserver-resident input
hyena-dna\tmerged C1 BAM\t/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam\tserver-resident input
hyena-dna\tmerged E5B BAM\t/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam\tserver-resident input
hyena-dna\tmodkit extract-full C1 by chromosome\t/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/by_chrom\tlarge server-resident labels
hyena-dna\tmodkit extract-full E5B by chromosome\t/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/by_chrom\tlarge server-resident labels
alphagenome\texternal ENCODE H3K4me3 BigWig\t/data/leuven/383/vsc38330/thesis_project_clean/alphagenome/metadata/external_tracks/ENCSR203XPU\tlarge BigWig ignored by Git
""",
    )
    write_text(
        CLEAN / "docs" / "superseded_and_failed_files_log.md",
        """# Superseded And Failed Files Log

This restructure uses manifests as the detailed log of historical files:

- `inventories/to_delete_manifest.tsv`: unused, failed, generated, SLURM, CNN-era, cache, and other quarantined files.
- `inventories/old_manifest.tsv`: historically useful or uncertain files kept outside the active workflow.
- `inventories/moved_files.tsv`: active files moved into the clean project tree.

Main decisions:

- CNN/5-mer workflows were not used in the final thesis and were quarantined in pillar-local `to_delete/`.
- Sample-conditioned, tiny/debug/smoke, and early HyenaDNA iterations were moved to `hyena-dna/OLD/`.
- AlphaGenome quick/smoke/dry-run/cache outputs were moved out of the active tree.
- Large external data remain on server paths and are documented in `metadata/external_data_registry.tsv`.
""",
    )


def main() -> None:
    if CLEAN.exists():
        raise SystemExit(f"Refusing to run because {CLEAN} already exists")
    ensure_dirs()
    move_preprocessing()
    move_hyena()
    move_alphagenome()
    write_project_files()
    write_manifest(CLEAN / "inventories" / "moved_files.tsv", rows_moved)
    write_manifest(CLEAN / "inventories" / "old_manifest.tsv", rows_old)
    write_manifest(CLEAN / "inventories" / "to_delete_manifest.tsv", rows_delete)
    print(f"Created {CLEAN}")
    print(f"Active moves: {len(rows_moved)}")
    print(f"OLD moves: {len(rows_old)}")
    print(f"to_delete moves: {len(rows_delete)}")


if __name__ == "__main__":
    main()
