# AlphaGenome Phase Map for Code Reorganization

This document explains the different AlphaGenome phases that happened during the project. It is meant to help decide what belongs in the clean GitHub version, what should remain as reproducible final output, and what can stay archived in `OLD/`.

## Current Clean Directory Shape

The cleaned project already has a reasonable top-level structure:

```text
alphagenome/
  docs/
  metadata/
  results/
  scripts/
  src/
  tests/
  OLD/
  to_delete/
  server_artifacts/
```

The confusing part is `OLD/`. It does not mean that everything inside is scientifically wrong. It mostly contains earlier development phases and intermediate outputs that were superseded by cleaner scripts and final results.

## Phase 0: Setup, Smoke Test, and Track Discovery

Purpose:

- Install AlphaGenome.
- Confirm the API key worked.
- Inspect AlphaGenome metadata.
- Identify the correct A549 H3K4me3 output track.
- Run one successful AlphaGenome smoke query.

Scientific role:

- This established that AlphaGenome could be queried and that the selected track was truly A549 H3K4me3.

Important retained files:

```text
metadata/selected_a549_h3k4me3_tracks.tsv
scripts/inspect_alphagenome_h3k4me3_tracks.py
src/alphagenome_query.py
```

Mostly historical files in `OLD/`:

```text
OLD/scripts/run_alphagenome_smoke.py
```

Keep status:

- Keep the metadata and reusable query code.
- The smoke script can stay archived unless you want a lightweight API test in `scripts/`.

## Phase 1: Define Held-Out Four-Chromosome Regions

Purpose:

- Create/record the held-out evaluation regions used for the AlphaGenome, DiMeLo, and HyenaDNA comparisons.

Main region files:

```text
OLD/outputs/4chrom_test_regions.tsv
OLD/outputs/4chrom_val_regions.tsv
```

Scientific role:

- These files define the held-out regions.
- Final analyses used test regions from chromosomes 11, 16, 17, and 19.
- There were 60 test regions total, 15 per chromosome.

Clean project script:

```text
scripts/make_4chrom_test_regions.py
```

Keep status:

- These region files are not really “old” conceptually.
- They should probably move out of `OLD/outputs/` into a clean location such as:

```text
metadata/regions/4chrom_test_regions.tsv
metadata/regions/4chrom_val_regions.tsv
```

or:

```text
results/region_definitions/
```

Do not lose these; they are part of reproducibility.

## Phase 2: First Reference-Based AlphaGenome Benchmark at 128 bp

Purpose:

- Query AlphaGenome on GRCh38 reference sequence.
- Export A549 H3K4me3 predictions.
- Build 128 bp DiMeLo and HyenaDNA genomic tracks.
- Compare AlphaGenome, DiMeLo, and HyenaDNA on the same reference-coordinate bins.

Scientific role:

- This was the first proof that DiMeLo signal showed biologically meaningful agreement with AlphaGenome.
- It answered the initial question: “is my DiMeLo data broadly consistent with a strong external regulatory model?”

Important old outputs:

```text
OLD/outputs/alphagenome_test_128bp.tsv
OLD/outputs/dimelo_test_128bp.tsv
OLD/outputs/hyenadna_test_128bp.tsv
OLD/outputs/benchmark.summary.tsv
OLD/outputs/benchmark.per_region.tsv
OLD/outputs/benchmark.summary.json
```

Relevant clean/reusable scripts:

```text
scripts/run_alphagenome_4chrom.py
scripts/export_alphagenome_128bp_tracks.py
scripts/build_dimelo_128bp_tracks.py
scripts/build_hyenadna_128bp_tracks.py
src/benchmark_utils.py
src/reference_tracks.py
```

Keep status:

- This phase is scientifically useful background, but it is superseded by the final ENCSR203XPU population benchmark.
- Keep scripts if they are still used by final workflows.
- Keep only small summary outputs if needed for thesis history.
- Large per-bin TSVs can stay archived or be regenerated.

## Phase 3: Validation Thresholds and Classification Metrics

Purpose:

- Use validation regions to define positive-bin cutoffs.
- Calculate AUROC/AUPRC and related metrics for AlphaGenome, DiMeLo, and HyenaDNA.

Scientific role:

- Useful for early interpretation and sensitivity checks.
- Less central than the final population-level correlation benchmark.

Files in `OLD/`:

```text
OLD/scripts/fit_validation_thresholds.py
OLD/scripts/evaluate_alphagenome_vs_dimelo.py
OLD/outputs/validation_thresholds.json
OLD/outputs/benchmark.summary.tsv
```

Keep status:

- Archive as methodological history.
- Not necessary as the main final thesis benchmark unless you still cite the AUROC/AUPRC result.

Key early result:

```text
AlphaGenome vs pooled DiMeLo 6mA:
Pearson  = 0.477
Spearman = 0.326
AUROC    = 0.809
AUPRC    = 0.517
Random AUPRC baseline = 0.123
```

## Phase 4: Plotting and Visual Comparison of AlphaGenome, DiMeLo, and HyenaDNA

Purpose:

- Create browser-style region plots showing:
  - AlphaGenome A549 H3K4me3
  - DiMeLo 6mA for C1/E5B
  - HyenaDNA predictions for C1/E5B
- Later normalize tracks to the same visual scale.

Scientific role:

- These plots made the signal visually interpretable.
- They showed that aggregation into genomic bins reveals strong regional agreement.

Historical scripts:

```text
OLD/scripts/plot_alphagenome_comparison.py
OLD/scripts/promoter_followup_analysis.py
OLD/scripts/summarize_promoter_followup_metrics.py
```

Keep status:

- These are mostly superseded by the final plotting scripts and final benchmark figures.
- If any figure was used in the thesis, preserve the final image, not every intermediate plotting script.

## Phase 5: 200 bp Follow-Up Benchmark

Purpose:

- Rebin signals from 128 bp/native representations to common 200 bp genomic bins.
- Use a more biologically interpretable scale, approximately nucleosome-scale.
- Compute averaged-signal HyenaDNA metrics at the same scale as AlphaGenome.
- Normalize tracks globally, not per locus.

Scientific role:

- This answered the promoter’s question about 128 bp versus a more biologically meaningful bin size.
- It made the AlphaGenome/HyenaDNA/DiMeLo comparison fairer at the population-track level.

Historical script:

```text
OLD/scripts/run_200bp_followup.py
```

Clean/reusable code:

```text
src/rebin_alphagenome_to_200bp.py
src/population_track_benchmark.py
```

Keep status:

- The concept is central.
- The old standalone script may be superseded if `population_track_benchmark.py` now performs the final workflow.

## Phase 6: AlphaGenome With Original ONT Read Sequences

Purpose:

- Test promoter suggestion: prompt AlphaGenome with original ONT read sequences rather than only GRCh38 reference sequence.
- Use `predict_sequence()` on BAM `query_sequence`.
- Project read-coordinate predictions back to GRCh38 using the BAM alignment/CIGAR.
- Aggregate predictions into 200 bp genomic bins.

Scientific role:

- This tested whether read-specific sequence differences, insertions, deletions, substitutions, soft clipping, or basecalling differences changed AlphaGenome predictions.
- It made AlphaGenome more comparable to HyenaDNA in the sense that both were run from original read sequences, even though AlphaGenome remains a sequence-to-functional-track model.

Clean scripts:

```text
scripts/build_alphagenome_readseq_aggregate.py
scripts/evaluate_readseq_alphagenome_200bp.py
scripts/plot_readseq_alphagenome_normalized.py
```

Historical files:

```text
OLD/scripts/alphagenome_read_sequence_sensitivity.py
OLD/outputs/alphagenome_readseq_200bp.tsv
OLD/outputs/alphagenome_readseq_200bp.summary.json
OLD/outputs/alphagenome_readseq_200bp_10reads.tsv
OLD/outputs/alphagenome_readseq_200bp_10reads.summary.json
OLD/outputs/alphagenome_readseq_200bp_10reads_benchmark.summary.tsv
```

Final retained summaries:

```text
results/readseq_summaries/alphagenome_readseq_200bp_10reads.summary.json
results/readseq_summaries/alphagenome_readlevel_10reads.summary.json
```

Keep status:

- Keep the final scripts and small summaries.
- Large readseq TSVs should not go to GitHub; they are generated data.

Key run:

```text
1,200 selected ONT reads attempted:
600 C1 reads
600 E5B reads
10 reads per region per sample across 60 held-out regions
1,198 predictions retained
2 reads excluded because they exceeded 131,072 bp
```

## Phase 7: AlphaGenome Read-Level Evaluation

Purpose:

- Evaluate AlphaGenome predictions from original ONT read sequences directly against observed DiMeLo values at read level.

Scientific role:

- This addressed the promoter’s question: does AlphaGenome also suffer from low read-level accuracy?
- Answer: yes, AlphaGenome also performs weakly at read level.
- This supports the conclusion that read-level DiMeLo variation is not explained well by DNA sequence alone.

Clean scripts:

```text
scripts/evaluate_alphagenome_readlevel.py
scripts/diagnose_alphagenome_readlevel_failure.py
scripts/make_readlevel_locus_example.py
```

Important retained result:

```text
results/readseq_summaries/alphagenome_readlevel_10reads.summary.json
```

Read-level performance:

```text
C1:
Pearson  = 0.1129
Spearman = -0.0418
AUROC    = 0.5503
AUPRC    = 0.0757

E5B:
Pearson  = 0.1087
Spearman = -0.0609
AUROC    = 0.5426
AUPRC    = 0.0784

Pooled:
Pearson  = 0.1107
Spearman = -0.0511
AUROC    = 0.5463
AUPRC    = 0.0770
```

Keep status:

- This is final and scientifically important.
- Keep the summary and same-locus example.
- Do not keep huge read-level intermediate tables in GitHub.

## Phase 8: Same-Locus Read-Level Example

Purpose:

- Make a qualitative figure showing why sequence-only read-level prediction fails.
- Select multiple reads covering the same genomic locus.
- Show that AlphaGenome predictions are similar because the sequence context is similar, while observed DiMeLo varies across molecules.

Final retained outputs:

```text
results/alphagenome_readlevel_locus_example/same_locus_read_example.png
results/alphagenome_readlevel_locus_example/same_locus_read_table.tsv
results/alphagenome_readlevel_locus_example/interpretation.md
```

Selected locus:

```text
chr19:35694000-35694200
19 reads
625 read-position observations
```

Keep status:

- Final thesis-relevant result.
- Keep figure, table, and interpretation.
- `all_read_bin_values.tsv` is large and can be excluded from GitHub if needed.

## Phase 9: External Experimental BigWig Benchmark

Purpose:

- Add a conventional experimental A549 H3K4me3 ChIP-seq BigWig track as a fair reference for AlphaGenome.
- Compare four aligned population-level tracks:
  - external experimental H3K4me3 BigWig
  - AlphaGenome A549 H3K4me3 prediction
  - aggregated HyenaDNA 6mA prediction
  - aggregated observed DiMeLo 6mA signal

Scientific role:

- This is the final and cleanest AlphaGenome benchmark.
- It separates matched comparisons from cross-assay comparisons.

Final external track:

```text
Experiment: ENCSR203XPU
File: ENCFF074PND
Assay: A549 H3K4me3 Histone ChIP-seq
Output type: fold change over control
Assembly: GRCh38
```

Final result directory:

```text
results/population_track_benchmark_ENCSR203XPU_200bp_final/
```

Important final files:

```text
results/population_track_benchmark_ENCSR203XPU_200bp_final/text_summary.md
results/population_track_benchmark_ENCSR203XPU_200bp_final/pairwise_metrics.tsv.gz
results/population_track_benchmark_ENCSR203XPU_200bp_final/chromosome_specific_metrics.tsv.gz
results/population_track_benchmark_ENCSR203XPU_200bp_final/coverage_sensitivity.tsv.gz
results/population_track_benchmark_ENCSR203XPU_200bp_final/weighted_correlations.tsv.gz
results/population_track_benchmark_ENCSR203XPU_200bp_final/normalization_parameters.tsv
results/population_track_benchmark_ENCSR203XPU_200bp_final/validated_track_metadata.tsv
results/population_track_benchmark_ENCSR203XPU_200bp_final/representative_loci.tsv.gz
results/population_track_benchmark_ENCSR203XPU_200bp_final/config.json
```

Main results:

```text
AlphaGenome vs ENCSR203XPU:
Pearson       = 0.9499
Spearman      = 0.6489
Normalized MAE = 0.0196
Normalized RMSE = 0.0486
Shared bins   = 30,000

HyenaDNA vs DiMeLo:
Pearson       = 0.6379
Spearman      = 0.5255
Normalized MAE = 0.1058
Normalized RMSE = 0.1522
Shared bins   = 20,376
```

Keep status:

- This is the main final AlphaGenome output.
- Keep summary tables, metadata, config, and final figures.
- Large canonical bin tables can be excluded from GitHub unless needed.

## Phase 10: Final Thesis Figures

Purpose:

- Create cleaner, publication/thesis-ready figures without noisy titles.
- Combine plots into panels.

Clean scripts:

```text
scripts/make_thesis_correlation_heatmap_panel.py
scripts/make_thesis_density_three_panel.py
scripts/add_peak_agreement_example.py
```

Important outputs:

```text
results/population_track_benchmark_ENCSR203XPU_200bp_final/figures/thesis_correlation_heatmap_panel.png
results/population_track_benchmark_ENCSR203XPU_200bp_final/figures/thesis_correlation_heatmap_panel.svg
results/population_track_benchmark_ENCSR203XPU_200bp_final/figures/thesis_correlation_heatmap_panel.pdf
```

Also for the GSM2421502 1 kb benchmark:

```text
outputs/population_track_benchmark_GSM2421502_1kb/figures/thesis_density_three_panel.*
```

Keep status:

- Keep final thesis figures.
- Keep scripts that regenerate them.

## Phase 11: 5mC Visual Checks

Purpose:

- Make visual checks for CpG 5mC using DiMeLo/HyenaDNA only.
- AlphaGenome was not included in the 5mC visual check.

Scientific role:

- Useful side analysis for checking HyenaDNA visually.
- Not part of the main AlphaGenome H3K4me3 story.

Keep status:

- This can be archived separately or moved to a HyenaDNA/methylation-specific section.
- It does not belong in the main AlphaGenome benchmark narrative.

## What Is Actually “Old”?

The following are genuinely old/superseded:

```text
OLD/scripts/plot_alphagenome_comparison.py
OLD/scripts/promoter_followup_analysis.py
OLD/scripts/summarize_promoter_followup_metrics.py
OLD/scripts/run_200bp_followup.py
OLD/scripts/alphagenome_read_sequence_sensitivity.py
OLD/scripts/evaluate_alphagenome_vs_dimelo.py
OLD/scripts/fit_validation_thresholds.py
```

These represent development history. They may be useful for reference, but the clean project should rely on `src/` and `scripts/`.

The following are not conceptually old, but are misplaced in `OLD/`:

```text
OLD/outputs/4chrom_test_regions.tsv
OLD/outputs/4chrom_val_regions.tsv
```

These should be moved or copied into a clean metadata/regions location.

The following are old generated/intermediate outputs:

```text
OLD/outputs/alphagenome_test_128bp.tsv
OLD/outputs/dimelo_test_128bp*.tsv
OLD/outputs/hyenadna_test_128bp*.tsv
OLD/outputs/benchmark*.tsv
OLD/outputs/alphagenome_readseq_*.tsv
```

These should generally stay archived or be regenerated. Do not put large versions in GitHub.

## Recommended Final Organization

Suggested clean structure:

```text
alphagenome/
  README.md
  docs/
    ALPHAGENOME_RESTRUCTURING_CONTEXT.md
    ALPHAGENOME_PHASE_MAP.md
    methods_summary.md
    results_summary.md
  metadata/
    selected_a549_h3k4me3_tracks.tsv
    regions/
      4chrom_test_regions.tsv
      4chrom_val_regions.tsv
    external_tracks/
      ENCSR203XPU.metadata.json
  src/
    alphagenome_query.py
    benchmark_utils.py
    population_track_benchmark.py
    reference_tracks.py
    rebin_alphagenome_to_200bp.py
  scripts/
    inspect_alphagenome_h3k4me3_tracks.py
    make_4chrom_test_regions.py
    run_alphagenome_4chrom.py
    export_alphagenome_128bp_tracks.py
    build_dimelo_128bp_tracks.py
    build_hyenadna_128bp_tracks.py
    build_alphagenome_readseq_aggregate.py
    evaluate_readseq_alphagenome_200bp.py
    evaluate_alphagenome_readlevel.py
    make_readlevel_locus_example.py
    add_peak_agreement_example.py
    make_thesis_correlation_heatmap_panel.py
  results/
    population_track_benchmark_ENCSR203XPU_200bp_final/
    alphagenome_readlevel_locus_example/
    readseq_summaries/
  tests/
  OLD/
  to_delete/
```

## Minimal GitHub Keep List

If the goal is a clean GitHub repository, keep:

```text
docs/
metadata/selected_a549_h3k4me3_tracks.tsv
metadata/regions/4chrom_test_regions.tsv
metadata/regions/4chrom_val_regions.tsv
metadata/external_tracks/ENCSR203XPU.metadata.json
src/
scripts/
tests/
results/population_track_benchmark_ENCSR203XPU_200bp_final/text_summary.md
results/population_track_benchmark_ENCSR203XPU_200bp_final/pairwise_metrics.tsv.gz
results/population_track_benchmark_ENCSR203XPU_200bp_final/chromosome_specific_metrics.tsv.gz
results/population_track_benchmark_ENCSR203XPU_200bp_final/normalization_parameters.tsv
results/population_track_benchmark_ENCSR203XPU_200bp_final/validated_track_metadata.tsv
results/population_track_benchmark_ENCSR203XPU_200bp_final/config.json
results/alphagenome_readlevel_locus_example/interpretation.md
results/alphagenome_readlevel_locus_example/same_locus_read_table.tsv
results/alphagenome_readlevel_locus_example/same_locus_read_example.png
results/readseq_summaries/*.json
```

Exclude from GitHub:

```text
cache/
cache_readseq*/
*.npz
*.npy
*.bigWig
*.bw
*.bam
*.bai
slurm-*.out
slurm-*.err
large per-bin TSVs
large read-level TSVs
__pycache__/
```

## One-Sentence Final Story

The AlphaGenome analysis evolved from a first reference-sequence sanity check against DiMeLo, to a fair population-level benchmark against a matched ENCODE A549 H3K4me3 ChIP-seq BigWig, plus a read-level ONT-sequence analysis showing that DNA sequence alone explains population-level regulatory patterns much better than individual molecule-level DiMeLo variation.

