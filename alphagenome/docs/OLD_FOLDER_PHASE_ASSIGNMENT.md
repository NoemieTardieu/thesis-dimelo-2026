# OLD Folder Phase Assignment

This document maps the contents of:

```text
/data/leuven/383/vsc38330/thesis_project_clean/alphagenome/OLD
```

to the AlphaGenome project phases. The goal is to avoid treating everything in `OLD/` as useless. Many files in `OLD/` are still scientifically meaningful, but they are earlier-phase outputs, intermediate products, or files that should be moved into cleaner locations.

## Summary Decision

`OLD/` should not mean “bad” or “unused.” It currently contains four different kinds of material:

1. **Misplaced reproducibility metadata**  
   Important small files that should move to `metadata/`.

2. **Earlier-phase scientific outputs**  
   Useful for thesis history, but superseded by the final benchmark.

3. **Final or near-final generated outputs duplicated from the old working tree**  
   These should either already exist in `results/` or be copied there intentionally.

4. **Large generated/intermediate files**  
   Useful locally, but not for GitHub.

## Phase 0: Setup, Smoke Test, and Track Discovery

### OLD files/folders

```text
OLD/scripts/run_alphagenome_smoke.py
OLD/outputs/metadata/chip_histone_all.tsv
OLD/outputs/metadata/chip_histone_h3k4me3.tsv
OLD/outputs/plots/alphagenome_smoke.png
```

### Meaning

These belong to the first AlphaGenome setup and sanity-check phase.

### Suggested action

Keep only if you want a reproducibility trail for the initial API/metadata check.

Suggested destination:

```text
docs/archive/setup_smoke/
```

or leave in:

```text
OLD/setup_smoke/
```

The selected final metadata already exists outside `OLD`:

```text
metadata/selected_a549_h3k4me3_tracks.tsv
```

## Phase 1: Held-Out Region Definition

### OLD files/folders

```text
OLD/outputs/4chrom_test_regions.tsv
OLD/outputs/4chrom_val_regions.tsv
```

### Meaning

These are not really old. They define the held-out genomic regions used in the AlphaGenome, DiMeLo, and HyenaDNA analyses.

### Suggested action

Move or copy into:

```text
metadata/regions/4chrom_test_regions.tsv
metadata/regions/4chrom_val_regions.tsv
```

These are small and should be kept for GitHub/reproducibility.

## Phase 2: First Reference-Based AlphaGenome Benchmark at 128 bp

### OLD files/folders

```text
OLD/outputs/alphagenome_test_128bp.tsv
OLD/outputs/dimelo_test_128bp.tsv
OLD/outputs/dimelo_test_128bp.chr*.tsv
OLD/outputs/dimelo_test_128bp.chr*.summary.json
OLD/outputs/dimelo_val_128bp.tsv
OLD/outputs/dimelo_val_128bp.chr*.tsv
OLD/outputs/dimelo_val_128bp.chr*.summary.json
OLD/outputs/hyenadna_test_128bp.tsv
OLD/outputs/hyenadna_test_128bp.chr*.tsv
OLD/outputs/hyenadna_test_128bp.chr*.summary.json
OLD/outputs/benchmark.summary.tsv
OLD/outputs/benchmark.summary.json
OLD/outputs/benchmark.per_region.tsv
```

### Meaning

This was the first working AlphaGenome-vs-DiMeLo-vs-HyenaDNA benchmark at 128 bp.

### Suggested action

Scientifically useful, but superseded by the final 200 bp population benchmark.

Keep small summaries if cited:

```text
results/archive/initial_128bp_benchmark/benchmark.summary.tsv
results/archive/initial_128bp_benchmark/benchmark.summary.json
```

Large per-bin TSVs should stay local/archive-only and should not go to GitHub.

## Phase 3: Validation Thresholds and Classification Metrics

### OLD files/folders

```text
OLD/scripts/fit_validation_thresholds.py
OLD/scripts/evaluate_alphagenome_vs_dimelo.py
OLD/outputs/validation_thresholds.json
OLD/outputs/benchmark.summary.tsv
OLD/outputs/benchmark.per_region.tsv
```

### Meaning

This phase computed validation-derived thresholds and early AUROC/AUPRC-style metrics.

### Suggested action

Archive as earlier analysis. Keep only if you mention the early AUROC/AUPRC result in the thesis.

Suggested destination:

```text
results/archive/validation_threshold_benchmark/
```

## Phase 4: Early Visual Region Plots and Promoter Follow-Up

### OLD files/folders

```text
OLD/scripts/plot_alphagenome_comparison.py
OLD/scripts/promoter_followup_analysis.py
OLD/scripts/summarize_promoter_followup_metrics.py
OLD/outputs/promoter_followup/
OLD/outputs/plots/comparison/
```

### Meaning

This phase generated early normalized plots and promoter-facing summaries comparing AlphaGenome, DiMeLo, and HyenaDNA.

### Suggested action

Mostly superseded by final benchmark plots.

If any old figure was used in a meeting or draft, keep the figure in:

```text
results/archive/promoter_followup/
```

Otherwise, keep only locally.

## Phase 5: 200 bp AlphaGenome/DiMeLo/HyenaDNA Follow-Up

### OLD files/folders

```text
OLD/scripts/run_200bp_followup.py
OLD/outputs/benchmark_200bp/
OLD/outputs/benchmark_200bp_smoke/
```

### Meaning

This phase rebinned the original AlphaGenome/DiMeLo/HyenaDNA comparison to 200 bp.

It answered the promoter’s question about moving away from 128 bp toward a more biologically interpretable nucleosome-scale bin.

### Suggested action

Keep summary-level results if used in the thesis narrative.

Suggested destination:

```text
results/archive/benchmark_200bp_pre_encode/
```

The smoke output can be left in `OLD` or moved to:

```text
results/archive/smoke_tests/
```

Large per-bin TSVs and many region plots should not go to GitHub unless selected as final figures.

## Phase 6: AlphaGenome on Original ONT Read Sequences

### OLD files/folders

```text
OLD/scripts/alphagenome_read_sequence_sensitivity.py
OLD/outputs/read_sequence_sensitivity/
OLD/outputs/alphagenome_readseq_200bp.tsv
OLD/outputs/alphagenome_readseq_200bp.summary.json
OLD/outputs/alphagenome_readseq_200bp_benchmark.same_bins.tsv
OLD/outputs/alphagenome_readseq_200bp_benchmark.summary.json
OLD/outputs/alphagenome_readseq_200bp_benchmark.summary.tsv
OLD/outputs/alphagenome_readseq_200bp_10reads.tsv
OLD/outputs/alphagenome_readseq_200bp_10reads_benchmark.summary.json
OLD/outputs/alphagenome_readseq_200bp_10reads_benchmark.summary.tsv
OLD/outputs/alphagenome_readseq_200bp_10reads_normalized_plots/
```

### Meaning

This phase used original ONT `query_sequence` as input to AlphaGenome through `predict_sequence()`.

It tested whether read-specific sequence differences changed AlphaGenome predictions and allowed comparison with HyenaDNA’s read-sequence input style.

### Suggested action

Keep final small summaries:

```text
results/readseq_summaries/
```

The clean folder already has:

```text
results/readseq_summaries/alphagenome_readseq_200bp_10reads.summary.json
results/readseq_summaries/alphagenome_readlevel_10reads.summary.json
```

Large readseq `.tsv` and `.npz` outputs should remain local/archive-only, not GitHub.

## Phase 7: AlphaGenome Read-Level Evaluation

### OLD files/folders

```text
OLD/outputs/alphagenome_readlevel_10reads.per_read.tsv
OLD/outputs/alphagenome_readlevel_10reads.summary.tsv
OLD/outputs/alphagenome_readlevel_diagnostics/
```

### Meaning

This phase evaluated AlphaGenome predictions directly at read level against observed DiMeLo signal.

It showed that AlphaGenome also performs weakly at read level.

### Suggested action

Keep small summaries and diagnostic plots if useful:

```text
results/archive/readlevel_diagnostics/
```

or keep only the final same-locus example in:

```text
results/alphagenome_readlevel_locus_example/
```

Large per-read tables should not go to GitHub.

## Phase 8: Same-Locus Read-Level Example

### OLD files/folders

The final version is already outside `OLD`:

```text
results/alphagenome_readlevel_locus_example/
```

### Meaning

This is a thesis-relevant final result, not old.

It shows that multiple reads at the same genomic locus get similar AlphaGenome scores but variable DiMeLo 6mA signal.

### Suggested action

Keep in `results/`.

Do not bury this in `OLD`.

## Phase 9: External BigWig / Population-Level Benchmark Development

### OLD files/folders

```text
OLD/outputs/external_bigwig_benchmark_GSE91218/
OLD/outputs/external_bigwig_benchmark_GSM2421502/
OLD/outputs/population_track_benchmark_GSM2421502_dryrun/
OLD/outputs/population_track_benchmark_GSM2421502_1kb/
```

### Meaning

This phase explored adding external population-level BigWig tracks before the final matched ENCSR203XPU benchmark.

`GSM2421502_1kb` may still be useful as a secondary/exploratory external-track benchmark, especially if thesis figures were made from it.

### Suggested action

Separate these from the final ENCSR203XPU analysis.

Suggested destination:

```text
results/archive/external_bigwig_exploratory/
```

If `GSM2421502_1kb` is used in the thesis, move selected summary/figure files to:

```text
results/secondary_external_track_GSM2421502_1kb/
```

Do not mix these with the final ENCSR203XPU benchmark.

## Phase 10: Final ENCSR203XPU Population-Level Benchmark

### OLD files/folders

```text
OLD/outputs/population_track_benchmark_ENCSR203XPU_200bp/
OLD/outputs/population_track_benchmark_ENCSR203XPU_200bp_quick/
OLD/outputs/population_track_benchmark_ENCSR203XPU_200bp_final/
```

### Meaning

This is the final matched population-level benchmark:

```text
external ENCSR203XPU A549 H3K4me3 ChIP-seq
vs AlphaGenome A549 H3K4me3
vs aggregated HyenaDNA 6mA
vs aggregated DiMeLo 6mA
```

### Suggested action

The clean final output should live in:

```text
results/population_track_benchmark_ENCSR203XPU_200bp_final/
```

The clean folder already has the main final summary tables there.

Inside `OLD`, only keep:

- raw/canonical bin tables if you need local reproducibility
- processing logs
- file inventories

Do not keep duplicate final outputs in both `OLD` and `results` unless there is a clear reason.

## Phase 11: Final Thesis Figures

### OLD files/folders

Final figure outputs may exist under:

```text
OLD/outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/
OLD/outputs/population_track_benchmark_GSM2421502_1kb/figures/
```

Clean figure scripts:

```text
scripts/make_thesis_correlation_heatmap_panel.py
scripts/make_thesis_density_three_panel.py
scripts/add_peak_agreement_example.py
```

### Meaning

These are final presentation/thesis figures.

### Suggested action

Move/copy selected final figures into a clean folder such as:

```text
results/thesis_figures/
```

or keep them inside each relevant result directory:

```text
results/population_track_benchmark_ENCSR203XPU_200bp_final/figures/
results/secondary_external_track_GSM2421502_1kb/figures/
```

Do not leave final thesis figures only inside `OLD`.

## Phase 12: 5mC Visual Check

### OLD files/folders

```text
OLD/scripts/build_plot_5mc_hyenadna.py
OLD/outputs/5mc_visual/
```

### Meaning

This was a visual check for DiMeLo/HyenaDNA 5mC-related signal.

It is not central to AlphaGenome because AlphaGenome was not used in that 5mC comparison.

### Suggested action

Move out of AlphaGenome if possible:

```text
../hyenadna/results/5mc_visual/
```

or:

```text
results/archive/5mc_visual_check/
```

Do not make it part of the main AlphaGenome benchmark story.

## Recommended Replacement for OLD

Instead of one broad `OLD/`, use something like:

```text
archive/
  setup_smoke/
  initial_128bp_benchmark/
  validation_threshold_benchmark/
  promoter_followup_plots/
  benchmark_200bp_pre_encode/
  read_sequence_sensitivity/
  readlevel_diagnostics/
  external_bigwig_exploratory/
  smoke_runs/
```

And move truly important files into clean project folders:

```text
metadata/regions/
results/population_track_benchmark_ENCSR203XPU_200bp_final/
results/alphagenome_readlevel_locus_example/
results/readseq_summaries/
results/thesis_figures/
```

## Priority Moves

Highest priority:

```text
OLD/outputs/4chrom_test_regions.tsv
OLD/outputs/4chrom_val_regions.tsv
    -> metadata/regions/
```

Important if not already copied:

```text
OLD/outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/
    -> results/population_track_benchmark_ENCSR203XPU_200bp_final/figures/
```

Optional but useful:

```text
OLD/outputs/alphagenome_readlevel_diagnostics/
    -> results/archive/readlevel_diagnostics/
```

Should likely stay archive-only:

```text
OLD/outputs/alphagenome_readseq_200bp_10reads.tsv
OLD/outputs/alphagenome_readseq_200bp.tsv
OLD/outputs/read_sequence_sensitivity/*.npz
OLD/outputs/*/canonical_raw_bins.tsv.gz
OLD/outputs/*/canonical_normalized_bins.tsv.gz
```

