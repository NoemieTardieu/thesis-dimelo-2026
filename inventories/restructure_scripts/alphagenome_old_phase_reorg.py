#!/usr/bin/env python3
from __future__ import annotations

import csv
import fnmatch
import shutil
from pathlib import Path


ALPHA = Path("/data/leuven/383/vsc38330/thesis_project_clean/alphagenome")
OLD = ALPHA / "OLD"
OUT = OLD / "outputs"
SCRIPTS = OLD / "scripts"
ARCHIVE = ALPHA / "archive"
SERVER = ALPHA / "server_artifacts" / "alphagenome_archive_large_tables"
MANIFEST = ALPHA.parent / "inventories" / "alphagenome_old_reorg_manifest.tsv"

FIELDS = ["source", "destination", "phase", "status", "reason"]
rows: list[dict[str, str]] = []


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    i = 2
    while True:
        candidate = dest.with_name(f"{dest.stem}.moved{i}{dest.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def move_path(src: Path, dest: Path, phase: str, status: str, reason: str) -> bool:
    if not src.exists():
        return False
    dest = unique_dest(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = str(src)
    shutil.move(str(src), str(dest))
    rows.append(
        {
            "source": old,
            "destination": str(dest),
            "phase": phase,
            "status": status,
            "reason": reason,
        }
    )
    return True


def move_file_to(src: Path, dest_dir: Path, phase: str, status: str, reason: str) -> bool:
    return move_path(src, dest_dir / src.name, phase, status, reason)


def move_tree_to(src: Path, dest: Path, phase: str, status: str, reason: str) -> bool:
    return move_path(src, dest, phase, status, reason)


def move_globs(
    base: Path,
    patterns: list[str],
    dest_base: Path,
    phase: str,
    status: str,
    reason: str,
) -> None:
    if not base.exists():
        return
    for src in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = src.relative_to(base)
        rel_s = rel.as_posix()
        if any(fnmatch.fnmatch(rel_s, pat) for pat in patterns):
            move_path(src, dest_base / rel, phase, status, reason)


def write_manifest() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def update_gitignore() -> None:
    gitignore = ALPHA.parent / ".gitignore"
    text = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    additions = [
        "alphagenome_test_128bp.tsv",
        "dimelo_*_128bp*.tsv",
        "hyenadna_*_128bp*.tsv",
        "alphagenome_readseq_*.tsv",
        "canonical_raw_bins.tsv.gz",
        "canonical_normalized_bins.tsv.gz",
        "*.per_read.tsv",
    ]
    missing = [line for line in additions if line not in text]
    if missing:
        gitignore.write_text(text.rstrip() + "\n\n# AlphaGenome archived large generated tables\n" + "\n".join(missing) + "\n", encoding="utf-8")


def write_docs() -> None:
    index = """# AlphaGenome Archive Index

`OLD/` has been reorganized into phase-specific archive folders. `OLD/README.md` is now only a pointer.

## Active / Thesis-Relevant Locations

- `metadata/regions/`: held-out four-chromosome validation/test region definitions.
- `results/population_track_benchmark_ENCSR203XPU_200bp_final/`: primary final ENCSR203XPU 200 bp benchmark.
- `results/secondary_external_track_GSM2421502_1kb/`: selected secondary GSM2421502 1 kb summaries and thesis density figure.
- `results/readseq_summaries/`: small summaries for AlphaGenome on ONT read sequences.
- `results/alphagenome_readlevel_locus_example/`: same-locus read-level example.

## Archive Phases

- `archive/phase00_setup_smoke/`: setup, API smoke test, and metadata discovery.
- `archive/phase02_initial_128bp_reference_benchmark/`: first 128 bp reference-coordinate comparison summaries.
- `archive/phase03_validation_thresholds/`: validation-derived thresholds and early classification metrics.
- `archive/phase04_visual_promoter_followup/`: early visual/promoter follow-up plots and scripts.
- `archive/phase05_200bp_pre_encode_followup/`: pre-ENCODE 200 bp follow-up benchmark.
- `archive/phase06_read_sequence_sensitivity/`: AlphaGenome `predict_sequence()` sensitivity outputs.
- `archive/phase07_readlevel_diagnostics/`: direct read-level diagnostic summaries and plots.
- `archive/phase09_external_bigwig_exploratory/`: exploratory external BigWig benchmarks.
- `archive/phase10_encode_benchmark_dev_runs/`: non-final ENCSR203XPU quick/dev/final-duplicate local artifacts.
- `archive/phase12_5mc_visual_check/`: auxiliary 5mC visual checks, not central AlphaGenome evidence.

Large generated tables and bulky intermediates are under `server_artifacts/alphagenome_archive_large_tables/` and ignored by Git.
"""
    (ALPHA / "docs" / "ALPHAGENOME_ARCHIVE_INDEX.md").write_text(index, encoding="utf-8")
    (OLD / "README.md").write_text(
        """# OLD Reorganized

The previous broad `OLD/` contents were reorganized into:

- `../archive/` for phase-specific scientific history.
- `../metadata/` for reproducibility metadata.
- `../results/` for thesis-relevant final or secondary outputs.
- `../server_artifacts/alphagenome_archive_large_tables/` for bulky generated local tables.

See `../docs/ALPHAGENOME_ARCHIVE_INDEX.md` and `../../inventories/alphagenome_old_reorg_manifest.tsv`.
""",
        encoding="utf-8",
    )


def main() -> None:
    if not OLD.exists():
        raise SystemExit(f"Missing OLD folder: {OLD}")

    # Phase 1 reproducibility metadata promoted out of OLD.
    for name in ["4chrom_test_regions.tsv", "4chrom_val_regions.tsv"]:
        move_file_to(
            OUT / name,
            ALPHA / "metadata" / "regions",
            "phase01_region_definition",
            "active_metadata",
            "held-out region definition promoted from OLD",
        )

    # Phase 0 setup and metadata discovery.
    move_file_to(SCRIPTS / "run_alphagenome_smoke.py", ARCHIVE / "phase00_setup_smoke" / "scripts", "phase00_setup_smoke", "archive", "setup smoke script")
    move_tree_to(OUT / "metadata", ARCHIVE / "phase00_setup_smoke" / "metadata_discovery", "phase00_setup_smoke", "archive", "AlphaGenome track metadata discovery outputs")
    move_globs(OUT / "plots", ["alphagenome_smoke.*"], ARCHIVE / "phase00_setup_smoke" / "plots", "phase00_setup_smoke", "archive", "smoke test plot")

    # Phase 2: initial 128 bp benchmark. Keep summaries in archive, bulky per-bin tables in server artifacts.
    for name in ["benchmark.summary.tsv", "benchmark.summary.json", "benchmark.per_region.tsv"]:
        move_file_to(OUT / name, ARCHIVE / "phase02_initial_128bp_reference_benchmark" / "summaries", "phase02_initial_128bp_reference_benchmark", "archive", "initial 128 bp benchmark summary")
    move_globs(
        OUT,
        [
            "alphagenome_test_128bp.tsv",
            "dimelo_*_128bp*.tsv",
            "hyenadna_*_128bp*.tsv",
        ],
        SERVER / "phase02_initial_128bp_reference_benchmark",
        "phase02_initial_128bp_reference_benchmark",
        "server_artifact",
        "large 128 bp per-bin/generated track table",
    )
    move_globs(
        OUT,
        [
            "dimelo_*_128bp*.summary.json",
            "hyenadna_*_128bp*.summary.json",
        ],
        ARCHIVE / "phase02_initial_128bp_reference_benchmark" / "track_summaries",
        "phase02_initial_128bp_reference_benchmark",
        "archive",
        "small 128 bp track summary JSON",
    )

    # Phase 3 validation thresholds.
    for name in ["fit_validation_thresholds.py", "evaluate_alphagenome_vs_dimelo.py"]:
        move_file_to(SCRIPTS / name, ARCHIVE / "phase03_validation_thresholds" / "scripts", "phase03_validation_thresholds", "archive", "validation/classification script")
    move_file_to(OUT / "validation_thresholds.json", ARCHIVE / "phase03_validation_thresholds" / "outputs", "phase03_validation_thresholds", "archive", "validation-derived threshold file")

    # Phase 4 early visual/promoter follow-up.
    for name in ["plot_alphagenome_comparison.py", "promoter_followup_analysis.py", "summarize_promoter_followup_metrics.py"]:
        move_file_to(SCRIPTS / name, ARCHIVE / "phase04_visual_promoter_followup" / "scripts", "phase04_visual_promoter_followup", "archive", "early visual/promoter follow-up script")
    move_tree_to(OUT / "promoter_followup", ARCHIVE / "phase04_visual_promoter_followup" / "promoter_followup", "phase04_visual_promoter_followup", "archive", "promoter follow-up outputs")
    move_tree_to(OUT / "plots", ARCHIVE / "phase04_visual_promoter_followup" / "plots", "phase04_visual_promoter_followup", "archive", "early comparison plots")

    # Phase 5 200 bp pre-ENCODE follow-up.
    move_file_to(SCRIPTS / "run_200bp_followup.py", ARCHIVE / "phase05_200bp_pre_encode_followup" / "scripts", "phase05_200bp_pre_encode_followup", "archive", "pre-ENCODE 200 bp follow-up script")
    move_tree_to(OUT / "benchmark_200bp", ARCHIVE / "phase05_200bp_pre_encode_followup" / "benchmark_200bp", "phase05_200bp_pre_encode_followup", "archive", "pre-ENCODE 200 bp benchmark outputs")
    move_tree_to(OUT / "benchmark_200bp_smoke", ARCHIVE / "phase05_200bp_pre_encode_followup" / "benchmark_200bp_smoke", "phase05_200bp_pre_encode_followup", "archive", "pre-ENCODE 200 bp smoke outputs")

    # Phase 6 read-sequence sensitivity. Small summaries in results, large TSVs in server artifacts.
    move_file_to(SCRIPTS / "alphagenome_read_sequence_sensitivity.py", ARCHIVE / "phase06_read_sequence_sensitivity" / "scripts", "phase06_read_sequence_sensitivity", "archive", "read-sequence sensitivity script")
    move_tree_to(OUT / "read_sequence_sensitivity", ARCHIVE / "phase06_read_sequence_sensitivity" / "read_sequence_sensitivity", "phase06_read_sequence_sensitivity", "archive", "read sequence sensitivity output folder")
    for name in [
        "alphagenome_readseq_200bp.summary.json",
        "alphagenome_readseq_200bp_benchmark.summary.json",
        "alphagenome_readseq_200bp_benchmark.summary.tsv",
        "alphagenome_readseq_200bp_benchmark.same_bins.tsv",
        "alphagenome_readseq_200bp_10reads_benchmark.summary.json",
        "alphagenome_readseq_200bp_10reads_benchmark.summary.tsv",
    ]:
        move_file_to(OUT / name, ALPHA / "results" / "readseq_summaries", "phase06_read_sequence_sensitivity", "active_results", "small read-sequence summary promoted to results")
    for name in ["alphagenome_readseq_200bp.tsv", "alphagenome_readseq_200bp_10reads.tsv"]:
        move_file_to(OUT / name, SERVER / "phase06_read_sequence_sensitivity", "phase06_read_sequence_sensitivity", "server_artifact", "large read-sequence generated table")
    move_tree_to(OUT / "alphagenome_readseq_200bp_10reads_normalized_plots", ARCHIVE / "phase06_read_sequence_sensitivity" / "normalized_plots_10reads", "phase06_read_sequence_sensitivity", "archive", "read-sequence normalized plots")

    # Phase 7 read-level diagnostics.
    move_file_to(OUT / "alphagenome_readlevel_10reads.summary.tsv", ARCHIVE / "phase07_readlevel_diagnostics" / "summaries", "phase07_readlevel_diagnostics", "archive", "read-level evaluation summary")
    move_file_to(OUT / "alphagenome_readlevel_10reads.per_read.tsv", SERVER / "phase07_readlevel_diagnostics", "phase07_readlevel_diagnostics", "server_artifact", "large per-read read-level table")
    move_tree_to(OUT / "alphagenome_readlevel_diagnostics", ARCHIVE / "phase07_readlevel_diagnostics" / "diagnostics", "phase07_readlevel_diagnostics", "archive", "read-level diagnostic plots and summaries")

    # Phase 9 exploratory external BigWig benchmarks and secondary GSM2421502 result.
    gsm = OUT / "population_track_benchmark_GSM2421502_1kb"
    for name in ["text_summary.md", "config.json", "validated_track_metadata.tsv", "normalization_parameters.tsv", "pairwise_metrics.tsv.gz", "chromosome_specific_metrics.tsv.gz", "coverage_sensitivity.tsv.gz", "weighted_correlations.tsv.gz"]:
        move_file_to(gsm / name, ALPHA / "results" / "secondary_external_track_GSM2421502_1kb", "phase09_external_bigwig_exploratory", "active_secondary_result", "selected secondary GSM2421502 1 kb result")
    move_globs(gsm / "figures", ["thesis_density_three_panel.*", "density/*", "*.png", "*.svg", "*.pdf"], ALPHA / "results" / "secondary_external_track_GSM2421502_1kb" / "figures", "phase09_external_bigwig_exploratory", "active_secondary_result", "selected secondary GSM2421502 figures")
    move_tree_to(gsm, ARCHIVE / "phase09_external_bigwig_exploratory" / "population_track_benchmark_GSM2421502_1kb_remaining", "phase09_external_bigwig_exploratory", "archive", "remaining GSM2421502 benchmark artifacts")
    for d in ["external_bigwig_benchmark_GSE91218", "external_bigwig_benchmark_GSM2421502", "population_track_benchmark_GSM2421502_dryrun"]:
        move_tree_to(OUT / d, ARCHIVE / "phase09_external_bigwig_exploratory" / d, "phase09_external_bigwig_exploratory", "archive", "exploratory external BigWig benchmark")

    # Phase 10 ENCSR dev/non-final/final duplicate local artifacts.
    for d in ["population_track_benchmark_ENCSR203XPU_200bp", "population_track_benchmark_ENCSR203XPU_200bp_quick"]:
        move_tree_to(OUT / d, ARCHIVE / "phase10_encode_benchmark_dev_runs" / d, "phase10_encode_benchmark_dev_runs", "archive", "non-final ENCSR203XPU dev/quick run")
    final_dup = OUT / "population_track_benchmark_ENCSR203XPU_200bp_final"
    move_globs(
        final_dup,
        ["canonical_raw_bins.tsv.gz", "canonical_normalized_bins.tsv.gz", "file_inventory.tsv.gz", "proposed_deletion_manifest.tsv.gz", "processing_log.txt"],
        SERVER / "phase10_encode_final_local_artifacts",
        "phase10_encode_benchmark_dev_runs",
        "server_artifact",
        "bulky/log local final benchmark artifact already represented in clean results",
    )
    move_tree_to(final_dup, ARCHIVE / "phase10_encode_benchmark_dev_runs" / "population_track_benchmark_ENCSR203XPU_200bp_final_remaining_duplicate", "phase10_encode_benchmark_dev_runs", "archive", "remaining duplicate final benchmark files; primary copy is in results")

    # Phase 12 5mC visual check.
    move_file_to(SCRIPTS / "build_plot_5mc_hyenadna.py", ARCHIVE / "phase12_5mc_visual_check" / "scripts", "phase12_5mc_visual_check", "archive", "auxiliary 5mC visual check script")
    move_tree_to(OUT / "5mc_visual", ARCHIVE / "phase12_5mc_visual_check" / "5mc_visual", "phase12_5mc_visual_check", "archive", "auxiliary 5mC visual check outputs")

    # Any truly residual files/folders under OLD go to a catch-all archive instead of being lost.
    if SCRIPTS.exists():
        move_tree_to(SCRIPTS, ARCHIVE / "unassigned_old_residual" / "scripts", "unassigned", "archive", "unassigned residual OLD scripts")
    if OUT.exists():
        move_tree_to(OUT, ARCHIVE / "unassigned_old_residual" / "outputs", "unassigned", "archive", "unassigned residual OLD outputs")

    write_docs()
    update_gitignore()
    write_manifest()
    print(f"AlphaGenome OLD reorg moves: {len(rows)}")


if __name__ == "__main__":
    main()
