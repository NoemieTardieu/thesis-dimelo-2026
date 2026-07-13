# AlphaGenome Analysis Restructuring Context

This document summarizes the AlphaGenome-related work so the project can be reorganized cleanly for GitHub and thesis reproducibility. It is intended as context for a fresh restructuring prompt, not as the final thesis text.

## High-Level Scientific Goal

The original goal was to use AlphaGenome as an independent reference baseline to test whether the A549 DiMeLo-seq data from clones C1 and E5B contained biologically meaningful H3K4me3-associated signal, and to compare that signal with HyenaDNA predictions.

The final interpretation became more precise:

- AlphaGenome is a pretrained sequence-to-functional-genomics model.
- It does not use nanopore reads, DiMeLo 6mA probabilities, 5mC calls, HyenaDNA predictions, or sample identity as input in the standard reference-based analysis.
- The main AlphaGenome input was the GRCh38 reference sequence for held-out genomic intervals.
- AlphaGenome output was the selected A549 H3K4me3 Histone ChIP-seq-like prediction track.
- DiMeLo and HyenaDNA are read-level signals, so they were mapped back to GRCh38 coordinates and aggregated into population-level genomic bins before comparison.

The AlphaGenome work therefore provides:

1. A sequence-based population-level baseline.
2. A matched experimental H3K4me3 ChIP-seq benchmark using ENCSR203XPU.
3. Evidence that aggregated DiMeLo and HyenaDNA signals show meaningful regulatory structure.
4. Evidence that read-level DiMeLo variation is difficult to predict from DNA sequence alone.

## Key Biological Assumptions

- C1 and E5B are both clones of the A549 cell line.
- DiMeLo 6mA signal is used as a proxy for antibody-directed H3K4me3-associated chromatin occupancy, not as the same molecular quantity as conventional H3K4me3 ChIP-seq.
- AlphaGenome predicts conventional A549 H3K4me3 ChIP-seq-like signal from DNA sequence.
- Direct AlphaGenome-vs-DiMeLo comparisons are cross-assay comparisons and should be interpreted as shared H3K4me3-associated genomic structure, not identical measurement agreement.

## Main Genomic Scope

The main benchmark used the previously defined held-out test regions:

- File: `outputs/4chrom_test_regions.tsv`
- Split: `test`
- Chromosomes: `chr11`, `chr16`, `chr17`, `chr19`
- Regions: 60 total, 15 per chromosome
- Coordinate system: GRCh38 / hg38, chr-prefixed chromosome names, 0-based half-open intervals

Validation regions also existed and were used in earlier auxiliary threshold/classification analyses, but the final population-level benchmark used the held-out test regions.

## AlphaGenome Track Selection

AlphaGenome metadata was inspected after installing the package and setting `ALPHAGENOME_API_KEY`.

Selected AlphaGenome track:

- Name: `EFO:0001086 Histone ChIP-seq H3K4me3`
- Ontology term / curie: `EFO:0001086`
- Biosample: `A549`
- Biosample type: `cell_line`
- Life stage: `adult`
- Assay title: `Histone ChIP-seq`
- Histone mark: `H3K4me3`
- Data source: `encode`
- Strand: `.`
- Endedness: `single`
- Genetically modified: `False`

Important metadata output:

- `outputs/metadata/selected_a549_h3k4me3_tracks.tsv`

There was one qualifying A549 H3K4me3 track.

## AlphaGenome Reference-Based Prediction Workflow

Primary workflow:

```text
GRCh38 reference DNA sequence
        ↓
AlphaGenome predict_interval()
        ↓
Predicted A549 H3K4me3 signal
```

AlphaGenome received only the reference DNA sequence for each held-out interval.

It did not receive:

- ONT read sequences
- DiMeLo 6mA values
- CpG 5mC measurements
- HyenaDNA predictions
- C1/E5B sample identity
- BAM alignment information
- raw nanopore electrical signal

The selected AlphaGenome output had native 128 bp resolution. This was verified from cache provenance:

- Example cache: `cache/chr16_4300000_4400000_chip_histone_29f73c7408be.npz`
- Output shape example: `(1024, 4)`
- Returned/input width: `131072 bp`
- Resolution: `131072 / 1024 = 128 bp`

The first smoke query succeeded after adjusting to supported AlphaGenome input lengths. AlphaGenome supports input sequence lengths such as `16384`, `131072`, `524288`, and `1048576`; the main run used `131072 bp`.

## Why 128 bp and Why 200 bp

Initial comparisons used 128 bp bins because this matched the native AlphaGenome output stride/resolution for the selected H3K4me3 output.

Later, following promoter feedback, the final population-level benchmark used 200 bp bins:

- AlphaGenome still natively outputs 128 bp predictions.
- AlphaGenome values were rebinned/averaged into common non-overlapping 200 bp genomic bins.
- DiMeLo and HyenaDNA signals were aggregated to the same 200 bp bins.
- 200 bp was chosen as a more biologically interpretable scale, approximately nucleosome-scale plus linker DNA.
- The main purpose of 200 bp was harmonization and smoother population-level comparison, not changing the AlphaGenome model itself.

## DiMeLo and HyenaDNA Aggregation

DiMeLo and HyenaDNA started as read-level signals.

For each ONT read:

1. Read-level positions were mapped back to GRCh38 using BAM alignment and CIGAR operations.
2. Overlapping tensor windows were deduplicated/collapsed by read position.
3. Valid observed/predicted positions were assigned to genomic bins.
4. Values were averaged across all valid read-level observations in each bin.

Observed DiMeLo binned signal:

```text
D_b = sum(valid observed 6mA probabilities in bin b) / number of valid observed positions
```

HyenaDNA binned signal:

```text
H_b = sum(valid predicted 6mA probabilities in bin b) / number of valid predicted positions
```

Important details:

- Missing observations were not treated as zero.
- Coverage was tracked separately.
- In the final benchmark, `dimelo_coverage` refers to number of valid observed positions/observations, not number of reads.
- Read count was stored separately as `dimelo_read_count`.
- Final ENCSR benchmark used minimum DiMeLo coverage of 5 valid observations.

Relevant scripts:

- `build_dimelo_128bp_tracks.py`
- `build_hyenadna_128bp_tracks.py`
- `reference_tracks.py`
- `benchmark_utils.py`
- `population_track_benchmark.py`

## HyenaDNA Input Clarification

HyenaDNA used original ONT read sequences from BAM `query_sequence` as model input at read level.

Important distinction:

- Input to HyenaDNA: original basecalled ONT read sequence.
- Evaluation/visualization: predictions were mapped back through the BAM alignment/CIGAR to GRCh38 coordinates and averaged into genomic bins.

So HyenaDNA was not trained/evaluated as a pure reference-track model, even though its outputs were later projected onto reference coordinates for fair comparison with AlphaGenome and BigWig tracks.

## Initial AlphaGenome-DiMeLo-HyenaDNA Comparison

The early analysis compared:

- AlphaGenome A549 H3K4me3 prediction
- observed DiMeLo 6mA signal for C1 and E5B
- HyenaDNA predicted 6mA signal for C1 and E5B

These were plotted over the same genomic regions, initially at 128 bp and later at 200 bp.

The early pooled AlphaGenome-vs-DiMeLo result was approximately:

- Pearson: `0.477`
- Spearman: `0.326`
- AUROC: `0.809`
- AUPRC: `0.517`
- Random AUPRC baseline: `0.123`

Interpretation:

- AlphaGenome could identify high-DiMeLo-signal regions better than chance.
- This supported that the DiMeLo data contained biologically meaningful regulatory signal.
- This was still a cross-assay comparison, not a direct same-assay model evaluation.

## Final Population-Level ENCODE Benchmark

To make the AlphaGenome comparison fairer, an external matched experimental A549 H3K4me3 ChIP-seq BigWig was added.

External track:

- Experiment accession: `ENCSR203XPU`
- File accession: `ENCFF074PND`
- File: `external_tracks/ENCSR203XPU/ENCFF074PND_ENCSR203XPU_A549_H3K4me3_fold_change_GRCh38.bigWig`
- Metadata: `external_tracks/ENCSR203XPU/ENCSR203XPU.metadata.json`
- Assay: Histone ChIP-seq
- Biosample: A549 cell line
- Target: H3K4me3
- Assembly: GRCh38
- Output type: fold change over control
- Replicates: biological replicates 1, 2, and 3, pooled signal

Final four-track benchmark:

```text
T = external ENCSR203XPU experimental A549 H3K4me3 ChIP-seq signal
A = AlphaGenome-predicted A549 H3K4me3 signal
H = aggregated HyenaDNA-predicted 6mA signal
D = aggregated observed DiMeLo 6mA signal
```

Primary matched comparisons:

- `T-A`: AlphaGenome vs external ENCSR203XPU H3K4me3 ChIP-seq
- `H-D`: HyenaDNA vs observed DiMeLo 6mA

Secondary cross-assay comparisons:

- `T-H`
- `T-D`
- `A-H`
- `A-D`

Main output directory:

- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/`

Important final tables:

- `canonical_raw_bins.tsv.gz`
- `canonical_normalized_bins.tsv.gz`
- `pairwise_metrics.tsv.gz`
- `chromosome_specific_metrics.tsv.gz`
- `coverage_sensitivity.tsv.gz`
- `weighted_correlations.tsv.gz`
- `normalization_parameters.tsv`
- `validated_track_metadata.tsv`
- `representative_loci.tsv.gz`
- `text_summary.md`
- `config.json`

## Final ENCSR203XPU 200 bp Numerical Results

Primary matched comparisons:

```text
AlphaGenome vs ENCSR203XPU A549 H3K4me3:
Pearson r      = 0.9499
Spearman rho   = 0.6489
Normalized MAE = 0.0196
Normalized RMSE = 0.0486
Shared bins    = 30,000

HyenaDNA vs observed DiMeLo 6mA:
Pearson r      = 0.6379
Spearman rho   = 0.5255
Normalized MAE = 0.1058
Normalized RMSE = 0.1522
Shared bins    = 20,376
```

Secondary cross-assay comparisons:

```text
ENCSR203XPU vs HyenaDNA:
Pearson r    = 0.5630
Spearman rho = 0.2530
Shared bins  = 20,377

ENCSR203XPU vs DiMeLo:
Pearson r    = 0.5336
Spearman rho = 0.3214
Shared bins  = 20,376

AlphaGenome vs HyenaDNA:
Pearson r    = 0.5732
Spearman rho = 0.2569
Shared bins  = 20,377

AlphaGenome vs DiMeLo:
Pearson r    = 0.5415
Spearman rho = 0.3606
Shared bins  = 20,376
```

AlphaGenome vs ENCSR203XPU Pearson by chromosome:

```text
chr11 = 0.9450
chr16 = 0.9576
chr17 = 0.9556
chr19 = 0.9417
```

This showed that the high AlphaGenome-vs-ENCODE agreement was not driven by a single chromosome.

## Correlation and Normalization Details

Pairwise Pearson and Spearman correlations were calculated on raw binned values, not normalized values.

The reported final pairwise metrics used pair-specific valid-bin intersections:

- `intersection = pair_specific`

A common four-track intersection was also calculated as a sensitivity/consistency output:

- `intersection = common_four_track`

Robust normalization was used for visualization and normalized error metrics:

- Each complete track was normalized once across all evaluation bins.
- 1st percentile mapped toward 0.
- 99th percentile mapped toward 1.
- Values were clipped to `[0, 1]`.
- Normalization was not performed separately per region or per chromosome.

Normalization parameters were saved in:

- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/normalization_parameters.tsv`

## Blacklist and Contig Filtering

Final benchmark config:

- `blacklist_bed = null`
- `include_alt_contigs = false`
- `include_mito = false`

Therefore:

- No ENCODE blacklist filtering was applied.
- Alternative contigs were not included.
- Mitochondrial DNA was not included.
- Only requested canonical chromosomes `chr11`, `chr16`, `chr17`, and `chr19` were analyzed.

## Important Final Figures

Final ENCSR203XPU benchmark figures:

- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/pearson_heatmap.png`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/spearman_heatmap.png`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/density/density_T_A.png`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/density/density_H_D.png`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/density/density_A_D.png`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/correlation_by_chromosome.png`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/coverage_sensitivity.png`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/peak_agreement/high_signal_peak_agreement.png`

Thesis-ready combined heatmap panel:

- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/thesis_correlation_heatmap_panel.png`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/thesis_correlation_heatmap_panel.svg`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/thesis_correlation_heatmap_panel.pdf`

Script:

- `make_thesis_correlation_heatmap_panel.py`

This combines Pearson and Spearman heatmaps side by side with panel labels A and B and no subplot titles.

## GSM2421502 1 kb Benchmark

There is also an older/external population benchmark directory:

- `outputs/population_track_benchmark_GSM2421502_1kb/`

This appears to be a 1 kb population-level benchmark using another external BigWig/track source. It contains:

- `text_summary.md`
- `canonical_raw_bins.tsv.gz`
- `canonical_normalized_bins.tsv.gz`
- `pairwise_metrics.tsv.gz`
- `figures/density/density_T_A.png`
- `figures/density/density_H_D.png`
- `figures/density/density_A_D.png`

Thesis-ready three-panel density figure was generated here:

- `outputs/population_track_benchmark_GSM2421502_1kb/figures/thesis_density_three_panel.png`
- `outputs/population_track_benchmark_GSM2421502_1kb/figures/thesis_density_three_panel.svg`
- `outputs/population_track_benchmark_GSM2421502_1kb/figures/thesis_density_three_panel.pdf`

Script:

- `make_thesis_density_three_panel.py`

Panels:

- A: external track vs AlphaGenome (`T-A`)
- B: HyenaDNA vs DiMeLo (`H-D`)
- C: AlphaGenome vs DiMeLo (`A-D`)

This script regenerates the density plots from the normalized binned table, removes old plot titles, and adds A/B/C labels.

## Representative Locus Selection

Representative loci in the final benchmark were selected using quantitative criteria, not only visual inspection.

Implementation:

- Function: `select_representative_loci()` in `population_track_benchmark.py`
- Window size: 10 bins
- Bin size: 200 bp
- Local window size: 2 kb
- Step size: 5 bins = 1 kb
- Values: globally robust-normalized track values
- Metrics: local normalized MAE and local Pearson correlation

Selection categories:

- `all_four_agree`
- `alphagenome_matches_external`
- `hyena_matches_dimelo`
- `dimelo_external_disagree`

Output:

- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/representative_loci.tsv.gz`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/representative_loci/`

Note:

- The automatically selected `all_four_agree` locus was mostly low-signal/background.
- A separate high-signal peak-agreement example was added.

High-signal peak agreement:

- Script: `add_peak_agreement_example.py`
- Output: `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/peak_agreement/high_signal_peak_agreement.png`
- Selected locus: `chr17:81870000-81872000`

## AlphaGenome on Original ONT Read Sequences

To address promoter feedback, AlphaGenome was also prompted with original basecalled ONT read sequences.

Workflow:

```text
ONT BAM query_sequence
        ↓
AlphaGenome predict_sequence()
        ↓
Read-coordinate A549 H3K4me3-like prediction
        ↓
BAM CIGAR projection back to GRCh38
        ↓
200 bp aggregation
```

Important clarification:

- `read.query_sequence` is the original basecalled ONT read sequence stored in the BAM.
- It is not the aligned reference sequence.
- It can include read-specific substitutions, insertions, soft-clipped sequence, genuine variants, and basecalling errors.
- It does not include raw nanopore electrical signal.
- It does not include modified-base probabilities.

Large ONT-read AlphaGenome run:

- Attempted reads: 1,200
- C1 selected reads: 600
- E5B selected reads: 600
- Selection: 10 reads per region per sample across 60 held-out regions
- Retained predictions: 1,198
- Excluded reads: 2 reads longer than 131,072 bp
- Output: `outputs/alphagenome_readseq_200bp_10reads.tsv`
- Summary: `outputs/alphagenome_readseq_200bp_10reads.summary.json`
- Benchmark summary: `outputs/alphagenome_readseq_200bp_10reads_benchmark.summary.tsv`

This ONT-read AlphaGenome analysis was a complementary baseline, not the primary reference-based AlphaGenome analysis.

## AlphaGenome Read-Level Evaluation

AlphaGenome predictions from ONT read sequences were also evaluated directly at read level against observed DiMeLo 6mA.

Command used:

```bash
cd /data/leuven/383/vsc38330/alphagenome

/scratch/leuven/383/vsc38330/.venv/bin/python evaluate_alphagenome_readlevel.py \
  --cache-dir cache_readseq_10reads \
  --out-prefix outputs/alphagenome_readlevel_10reads
```

Read-level results:

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

Interpretation:

- AlphaGenome performs well at population-level H3K4me3 prediction.
- AlphaGenome performs weakly at read-level DiMeLo prediction.
- This suggests that individual molecule-level DiMeLo variation cannot be explained well by DNA sequence alone.
- This supports the need for models incorporating molecule-level methylation/chromatin information, not only DNA sequence.

## Same-Locus Read-Level Example

A qualitative same-locus read-level example was generated to show why read-level prediction is difficult.

Output directory:

- `outputs/alphagenome_readlevel_locus_example/`

Important files:

- `same_locus_read_example.png`
- `same_locus_read_table.tsv`
- `all_read_bin_values.tsv`
- `interpretation.md`

Selected locus:

- `chr19:35694000-35694200`

Summary:

- 19 reads
- 625 read-position observations

Interpretation:

- Reads map to the same genomic locus and therefore have very similar sequence context.
- AlphaGenome gives broadly similar regulatory scores across those reads.
- Observed DiMeLo 6mA signal varies strongly between reads.
- This illustrates why sequence-only read-level prediction is hard.
- Aggregating across reads recovers locus-level signal.

## 5mC Visual Checks

There was also a separate visual analysis for CpG 5mC with DiMeLo/HyenaDNA only, without AlphaGenome.

Relevant outputs:

- `outputs/5mc_visual/`
- Example file: `outputs/5mc_visual/hyenadna_5mc_test_1000bp.tsv`

Purpose:

- Visual sanity check of HyenaDNA vs observed 5mC-related signal.
- Not part of the core AlphaGenome H3K4me3 benchmark.

## File/Directory Cleanup Needs

The current `alphagenome/` directory contains many generated files, exploratory scripts, SLURM logs, caches, and multiple output versions. It should be reorganized before GitHub upload.

Suggested final structure:

```text
alphagenome/
  README.md
  docs/
    ALPHAGENOME_RESTRUCTURING_CONTEXT.md
    thesis_methods_alpha_genome.md
    thesis_results_alpha_genome.md
  configs/
    population_benchmark_ENCSR203XPU_200bp.yaml
  src/
    alphagenome_query.py
    track_building.py
    population_benchmark.py
    plotting.py
    readseq_analysis.py
    utils.py
  scripts/
    inspect_alphagenome_h3k4me3_tracks.py
    run_alphagenome_queries.py
    export_alphagenome_tracks.py
    build_dimelo_tracks.py
    build_hyenadna_tracks.py
    run_population_benchmark.py
    make_thesis_correlation_heatmap_panel.py
    make_thesis_density_three_panel.py
    add_peak_agreement_example.py
  slurm/
    run_alphagenome_4chrom_wice.sbatch
    run_200bp_followup_wice.sbatch
    run_population_benchmark_wice.sbatch
  tests/
    test_benchmark.py
    test_synthetic_population_benchmark.py
  data/
    metadata/
    regions/
    external_tracks/README.md
  outputs/
    final/
      population_track_benchmark_ENCSR203XPU_200bp_final/
      thesis_figures/
    exploratory/
    logs/
  cache/
    README.md
```

Important GitHub policy:

- Do not commit API keys.
- Do not commit large raw BAMs, BigWigs, `.npz` caches, or huge generated tables unless explicitly desired.
- Keep small metadata TSV/JSON files if useful for reproducibility.
- Keep scripts, configs, tests, and final small summary tables.
- Put large final outputs in `.gitignore` or document them with download/regeneration instructions.

## Files That Should Probably Be Kept

Core scripts:

- `population_track_benchmark.py`
- `alphagenome_query.py`
- `inspect_alphagenome_h3k4me3_tracks.py`
- `export_alphagenome_128bp_tracks.py`
- `build_dimelo_128bp_tracks.py`
- `build_hyenadna_128bp_tracks.py`
- `benchmark_utils.py`
- `reference_tracks.py`
- `evaluate_alphagenome_readlevel.py`
- `make_readlevel_locus_example.py`
- `diagnose_alphagenome_readlevel_failure.py`
- `add_peak_agreement_example.py`
- `make_thesis_correlation_heatmap_panel.py`
- `make_thesis_density_three_panel.py`

Core final outputs:

- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/text_summary.md`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/pairwise_metrics.tsv.gz`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/chromosome_specific_metrics.tsv.gz`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/normalization_parameters.tsv`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/validated_track_metadata.tsv`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/thesis_correlation_heatmap_panel.*`
- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/peak_agreement/high_signal_peak_agreement.*`
- `outputs/alphagenome_readlevel_locus_example/same_locus_read_example.png`
- `outputs/alphagenome_readlevel_locus_example/interpretation.md`

Important configs and metadata:

- `outputs/population_track_benchmark_ENCSR203XPU_200bp_final/config.json`
- `outputs/metadata/selected_a549_h3k4me3_tracks.tsv`
- `external_tracks/ENCSR203XPU/ENCSR203XPU.metadata.json`

## Files That Should Probably Not Be Committed

Likely large/generated files:

- AlphaGenome cache directories
- `cache/`
- `cache_readseq_10reads/`
- large `.npz` files
- large `.tsv` files with per-bin/read-level values
- downloaded BigWig files
- SLURM `.out` and `.err` logs
- temporary exploratory outputs
- duplicate benchmark directories

Keep regeneration instructions instead.

## Suggested `.gitignore` Entries

```gitignore
# secrets
.env
env
*.key
*API_KEY*

# AlphaGenome caches and large generated data
cache*/
**/cache*/
*.npz
*.npy
*.bigWig
*.bw
*.bam
*.bai
*.cram
*.crai

# SLURM logs
slurm-*.out
slurm-*.err
*.out
*.err

# large tabular outputs
outputs/**/canonical_raw_bins.tsv.gz
outputs/**/canonical_normalized_bins.tsv.gz
outputs/**/all_read_bin_values.tsv
outputs/**/alphagenome_readseq_*.tsv

# Python/cache
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/

# local notebooks/checkpoints
.ipynb_checkpoints/
```

## Reproducibility Commands to Preserve

Generate thesis heatmap panel:

```bash
cd /data/leuven/383/vsc38330/alphagenome

env MPLCONFIGDIR=/tmp/mplconfig \
/scratch/leuven/383/vsc38330/.venv/bin/python \
make_thesis_correlation_heatmap_panel.py
```

Generate GSM2421502 three-panel density figure:

```bash
cd /data/leuven/383/vsc38330/alphagenome

env MPLCONFIGDIR=/tmp/mplconfig \
/scratch/leuven/383/vsc38330/.venv/bin/python \
make_thesis_density_three_panel.py
```

Read-level AlphaGenome evaluation:

```bash
cd /data/leuven/383/vsc38330/alphagenome

/scratch/leuven/383/vsc38330/.venv/bin/python evaluate_alphagenome_readlevel.py \
  --cache-dir cache_readseq_10reads \
  --out-prefix outputs/alphagenome_readlevel_10reads
```

## Main Thesis Takeaway

AlphaGenome strongly reproduced a matched experimental A549 H3K4me3 ChIP-seq track from ENCODE, confirming that the selected AlphaGenome output behaved as expected on the held-out genomic regions. Aggregated DiMeLo and HyenaDNA signals showed moderate agreement with both AlphaGenome and the experimental ChIP-seq track, supporting that the DiMeLo data contain meaningful H3K4me3-associated regulatory information. However, AlphaGenome performed poorly when evaluated directly at read level against individual DiMeLo observations, even when prompted with original ONT read sequences. This suggests that read-level DiMeLo variation is not sufficiently explained by DNA sequence alone and motivates the next modeling step: separating sequence-based methylation prediction and regulatory prediction, for example by modeling `P(M | D)` and then `P(Reg | M, D)` rather than trying to predict regulatory state from DNA sequence alone.

## Suggested Prompt for the Next Restructuring Chat

```text
I have an AlphaGenome analysis directory that became messy during development. Please help me restructure it into a clean GitHub-ready project while preserving reproducibility.

Use `ALPHAGENOME_RESTRUCTURING_CONTEXT.md` as the project context. The main goals are:

1. Keep only scripts, configs, metadata, tests, and small final summary outputs needed to reproduce the AlphaGenome/DiMeLo/HyenaDNA benchmark.
2. Separate source code, SLURM scripts, configs, docs, final figures, final summary tables, exploratory outputs, caches, and large generated files.
3. Do not delete anything automatically. First create a file inventory and proposed cleanup manifest.
4. Never remove API keys, raw data, BigWigs, BAMs, checkpoints, or final results unless explicitly confirmed.
5. Add or update `.gitignore` so large caches, BAMs, BigWigs, `.npz`, SLURM logs, and secrets are not committed.
6. Create a clean README explaining how to rerun the final ENCSR203XPU 200 bp population benchmark, the read-level AlphaGenome ONT analysis, and thesis figure generation.
7. Preserve the final scientific outputs:
   - ENCSR203XPU 200 bp population benchmark metrics and figures
   - thesis correlation heatmap panel
   - high-signal peak agreement figure
   - same-locus read-level AlphaGenome example
   - selected AlphaGenome metadata
8. Where possible, refactor duplicate plotting/benchmark scripts into reusable modules without changing scientific results.

Please inspect the directory first, then propose a restructuring plan, then implement it carefully with no destructive cleanup unless I explicitly approve.
```

