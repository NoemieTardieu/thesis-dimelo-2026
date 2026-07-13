# DiMeLo-seq Preprocessing Context for Repository Restructuring

This document summarizes the preprocessing work that should be preserved when reorganizing the thesis repository for a clean GitHub upload. It is intended as context for a later restructuring prompt, not as a final Methods section.

## Project Scope

The thesis preprocessing pipeline starts from ONT DiMeLo-seq modBAM files and produces long-context methylation/regulatory signal representations for downstream modeling, especially HyenaDNA-style long-context models.

The final direction is **not** the older CNN/5-mer route. The important final preprocessing route is:

```text
raw mark-specific modBAM files
  -> Phase 1 QC and exploratory summaries
  -> Phase 2 long-context interval backend
  -> Phase 3 HyenaDNA-compatible training index
```

The older `modkit extract full -> 5-mer CNN tensor` workflow was exploratory and should be treated as historical/optional unless it is needed for comparison.

## Source Data

Primary mark-specific modBAM inputs:

```text
/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam
/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam
/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam
```

Each BAM contains aligned ONT reads plus modified-base tags:

- `MM` / `Mm`: positions of candidate modified bases along reads.
- `ML` / `Ml`: modification probabilities on the BAM 0-255 scale.
- Standard alignment information: read name, chromosome, reference start, CIGAR, strand, mapping quality, read sequence, base qualities, etc.

The three marks correspond to separate DiMeLo-seq experiments. The preprocessing treats each mark separately, then combines mark-specific backend manifests into a shared Phase 3 index.

## Phase 1: QC and Read-Level Parsing

Main script:

```text
thesis_dimelo/src/preprocessing/phase1_dimelo_qc.py
```

Purpose:

- Parse modBAM files.
- Read `MM/Mm` and `ML/Ml` modified-base tags.
- Extract A-channel and C-channel modification information.
- Apply basic read/alignment filters.
- Produce read-level and binned QC outputs per histone mark.

Main outputs:

```text
thesis_dimelo/src/preprocessing/phase1_output/*.per_read.tsv
thesis_dimelo/src/preprocessing/phase1_output/*.binned_tracks.npz
thesis_dimelo/src/preprocessing/phase1_output/*.png
```

Summary/QC script:

```text
thesis_dimelo/src/preprocessing/phase1_qc_summary.py
```

Important summary outputs:

```text
thesis_dimelo/src/preprocessing/phase1_output/summary_report/phase1_qc_summary.tsv
thesis_dimelo/src/preprocessing/phase1_output/summary_report/phase1_npz_inventory.tsv
thesis_dimelo/src/preprocessing/phase1_output/summary_report/phase1_qc_report.md
thesis_dimelo/src/preprocessing/phase1_output/summary_report/*.svg
```

Exploratory visualization script:

```text
thesis_dimelo/src/preprocessing/phase_1_visualization/phase1_exploratory_viz.py
```

Purpose:

- Produce early visual summaries of signal distributions.
- Explore co-occurrence between A-derived regulatory proxy signal and C methylation signal.
- Generate preliminary plots used to decide whether the data were usable.

## Modkit Outputs Used Earlier

There was also a separate `modkit` exploration path:

- `modkit extract calls` for per-read summaries.
- `modkit pileup` for bedMethyl-like genomic tracks.
- `modkit bedmethyl tobigwig` for IGV-compatible visualization.
- `modkit extract full` for a CNN/5-mer modeling attempt.

The `modkit extract full` TSV format was used in the old CNN attempt. It has columns like:

```text
read_id
forward_read_position
ref_position
chrom
mod_strand
ref_strand
ref_mod_strand
fw_soft_clipped_start
fw_soft_clipped_end
alignment_start
alignment_end
read_length
mod_qual
mod_code
base_qual
ref_kmer
query_kmer
canonical_base
modified_primary_base
inferred
flag
```

That path created event-level rows and 5-mer tensors. It is **not** the final HyenaDNA preprocessing path. For restructuring, keep these scripts/notes only if you want to preserve exploratory history; they should not be mixed into the main final pipeline.

## Phase 2: Long-Context Interval Generation

Main script:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/generate_long_context_intervals.py
```

Purpose:

- Partition the genome into fixed long-context windows.
- Use the BAM header to obtain chromosome sizes.
- Assign each interval to a chromosome-based split.

Final interval settings:

```text
window_size = 1,000,000 bp
stride      = 1,000,000 bp
```

Chromosome split:

```text
train = chr1-16
valid = chr17-18
test  = chr19, chr20, chr21, chr22, chrX
```

Main output:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/intervals_long_context.tsv
```

Schema:

```text
window_id
chrom
start
end
split
```

Example row:

```text
chr1:0-1000000    chr1    0    1000000    train
```

This interval table is a key artifact and should be preserved.

## Phase 2: Methylation Interval Backend

Main builder:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/build_methylation_interval_backend.py
```

All-marks runner:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/run_backend_all_marks.py
```

Purpose:

- Read the original mark-specific BAMs directly.
- Build per-base methylation labels over each 1 Mb interval.
- Store one compressed NPZ per interval per mark.
- Store one manifest TSV per mark.

This is the important final preprocessing step for HyenaDNA-style modeling.

### Backend Inputs

Inputs:

```text
mark-specific BAM
intervals_long_context.tsv
target_base = C
```

Marks:

```text
h3k27ac
h3k27me3
h3k4me3
```

### Backend Filtering and Thresholds

The backend does **not** use the old CNN threshold `mod_qual >= 0.8`.

Instead, it applies thresholds directly while parsing BAM modification tags:

```text
min_mapq = 20
ml_threshold = 128
min_coverage = 3
methylated_frac_thr = 0.7
unmethylated_frac_thr = 0.3
```

Meaning:

- Reads with mapping quality below 20 are skipped.
- A modified C call is counted only if its `ML/Ml` value is at least 128.
- A genomic base must have coverage of at least 3 reads to receive a known label.
- If `meth_counts / coverage >= 0.7`, the position is labeled methylated.
- If `meth_counts / coverage <= 0.3`, the position is labeled unmethylated.
- If coverage is too low or the fraction is ambiguous, the position is labeled unknown/masked.

Discrete label convention:

```text
0 = unmethylated
1 = methylated
2 = unknown / masked
```

### Backend Logic

For each interval:

1. Fetch reads overlapping the interval.
2. Skip unmapped reads and low-MAPQ reads.
3. Map read/query positions to reference positions.
4. Count C coverage per genomic position.
5. Parse `MM/Mm` and `ML/Ml` tags.
6. Count confident modified C calls where `ML >= 128`.
7. Convert coverage and methylated counts into discrete labels `{0, 1, 2}`.
8. Save arrays and metadata into an NPZ file.
9. Append interval-level summary statistics to the mark manifest.

### Backend NPZ Contents

Each interval NPZ contains:

```text
methyl_ids
coverage
meth_counts
n_reads_used
chrom
start
end
mark
target_base
min_mapq
ml_threshold
min_coverage
methylated_frac_thr
unmethylated_frac_thr
```

Important arrays:

- `methyl_ids`: per-base discrete labels, length equal to interval length.
- `coverage`: per-base C coverage.
- `meth_counts`: per-base confident methylated C counts.
- `n_reads_used`: number of reads used for the interval.

### Backend Output Layout

Final backend root:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/backends_all_marks/
```

Expected mark-specific directories:

```text
backend_h3k27ac_C/
backend_h3k27me3_C/
backend_h3k4me3_C/
```

Each has:

```text
manifest_<mark>_C.tsv
<mark>_C_labels/*.npz
```

Example:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/backends_all_marks/backend_h3k27ac_C/manifest_h3k27ac_C.tsv
thesis_dimelo/src/preprocessing/phase2_long_context/backends_all_marks/backend_h3k27ac_C/h3k27ac_C_labels/chr1_0_1000000.npz
```

Manifest schema:

```text
window_id
chrom
start
end
split
mark
target_base
npz_path
known_frac
methylated_frac_known
n_reads_used
```

## Phase 2 Backend QC

Main script:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/summarize_backend_manifests.py
```

Purpose:

- Summarize backend completeness and label availability.
- Report known-label fraction.
- Report methylated fraction among known labels.
- Report read usage by mark and split.

Important outputs:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/backends_all_marks/summary/backend_manifest_qc_summary.tsv
thesis_dimelo/src/preprocessing/phase2_long_context/backends_all_marks/summary/backend_manifest_qc_summary.md
```

These outputs are useful for thesis reporting and for validating a rebuilt pipeline.

## Phase 2: Windowed Reg-vs-M Analysis

Main script:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/windowed_oe_enrichment_heatmap.py
```

Purpose:

- Analyze local relationships between A-derived regulatory proxy signal and C methylation inside the 1 Mb windows.
- Sub-bin each 1 Mb interval into smaller bins.
- Compute observed/expected enrichment heatmaps.

Feature definitions:

- `Reg`: A-channel regulatory proxy.
  - Main default: `a_mod_per_kb`.
  - Alternative explored: `a_mod_frac`.
- `M`: C methylation feature.
  - Main default: `c_meth_frac`.

Common bin sizes explored:

```text
10 kb
1 kb
200 bp
50 bp
```

Important output directories include:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/windowed_enrichment_1kb_combined/
thesis_dimelo/src/preprocessing/phase2_long_context/windowed_enrichment_200bp_combined/
thesis_dimelo/src/preprocessing/phase2_long_context/windowed_enrichment_50bp_combined/
```

Typical outputs:

```text
windowed_oe_summary.tsv
windowed_oe_notes.md
oe_heatmap_pooled.svg
oe_heatmap_by_mark.svg
oe_heatmap_<mark>.svg
feature_cache_*.npz
```

These analyses are useful for interpretation and thesis figures, but they are not the core model input. The core model input is the interval backend plus sequence extraction.

## Phase 3: HyenaDNA-Compatible Training Index

Main script:

```text
thesis_dimelo/src/preprocessing/phase3_hyena/build_phase3_training_index.py
```

Purpose:

- Merge mark-specific Phase 2 manifests into one model-facing index.
- Keep one row per genomic window.
- Attach each mark's backend NPZ path and summary statistics.

Main output:

```text
thesis_dimelo/src/preprocessing/phase3_hyena/phase3_training_index.tsv
```

Schema:

```text
window_id
chrom
start
end
split
h3k27ac_npz_path
h3k27ac_known_frac
h3k27ac_methylated_frac_known
h3k27ac_n_reads_used
h3k27me3_npz_path
h3k27me3_known_frac
h3k27me3_methylated_frac_known
h3k27me3_n_reads_used
h3k4me3_npz_path
h3k4me3_known_frac
h3k4me3_methylated_frac_known
h3k4me3_n_reads_used
```

This is the primary handoff artifact for HyenaDNA-style model dataset construction.

## Distinction From the Abandoned CNN Path

The abandoned CNN/5-mer path used:

```text
modkit extract full TSV
canonical_base = C
mod_code = m
label = 1 if mod_qual >= 0.8
label = 0 if mod_qual < 0.8
valid 5-mer ref_kmer/query_kmer
integer-encoded chunked NPZ tensors
```

Representative outputs:

```text
/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/merged_c1_C_m_q0p8_training.valid5mer.uppercase.cnn_features.tsv.gz
/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/merged_e5b_C_m_q0p8_validation.valid5mer.uppercase.cnn_features.tsv.gz
/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/tensors_integer_valid5mer_uppercase
/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/tensors_integer_valid5mer_uppercase
```

This was useful exploration, but it should not be confused with the final long-context preprocessing. In the final HyenaDNA-oriented workflow:

- The data are interval-based, not event-row-based.
- Labels are per-base arrays, not one label per extracted TSV row.
- Unknown labels are represented explicitly as `2`.
- The key thresholds are `ML >= 128`, `coverage >= 3`, `fraction >= 0.7`, and `fraction <= 0.3`.
- The old `mod_qual >= 0.8` rule is not the final modeling label definition.

## Files and Directories To Preserve in a Clean Repository

High-priority source code:

```text
thesis_dimelo/src/preprocessing/phase1_dimelo_qc.py
thesis_dimelo/src/preprocessing/phase1_qc_summary.py
thesis_dimelo/src/preprocessing/phase_1_visualization/
thesis_dimelo/src/preprocessing/phase2_long_context/generate_long_context_intervals.py
thesis_dimelo/src/preprocessing/phase2_long_context/build_methylation_interval_backend.py
thesis_dimelo/src/preprocessing/phase2_long_context/run_backend_all_marks.py
thesis_dimelo/src/preprocessing/phase2_long_context/methylation_interval_store.py
thesis_dimelo/src/preprocessing/phase2_long_context/summarize_backend_manifests.py
thesis_dimelo/src/preprocessing/phase2_long_context/windowed_oe_enrichment_heatmap.py
thesis_dimelo/src/preprocessing/phase3_hyena/build_phase3_training_index.py
```

High-priority metadata/config outputs:

```text
thesis_dimelo/src/preprocessing/phase2_long_context/intervals_long_context.tsv
thesis_dimelo/src/preprocessing/phase3_hyena/phase3_training_index.tsv
thesis_dimelo/src/preprocessing/phase2_long_context/backends_all_marks/*/manifest_*_C.tsv
```

High-priority documentation:

```text
thesis_dimelo/THESIS_WORKFLOW_UP_TO_PHASE2.txt
thesis_dimelo/PHASE2_LONG_CONTEXT_SUBSECTION_DRAFT.md
thesis_dimelo/PHASE3_PREPARATION_CHECKLIST.txt
thesis_dimelo/THESIS_WORKFLOW_REMAINING_PHASES_PLANNING.txt
```

Useful figures/tables for thesis, but not necessarily for GitHub if too large:

```text
thesis_dimelo/src/preprocessing/phase1_output/summary_report/
thesis_dimelo/src/preprocessing/phase_1_visualization/output/
thesis_dimelo/src/preprocessing/phase2_long_context/backends_all_marks/summary/
thesis_dimelo/src/preprocessing/phase2_long_context/windowed_enrichment_*/*summary.tsv
thesis_dimelo/src/preprocessing/phase2_long_context/windowed_enrichment_*/*.svg
```

Large generated files that probably should not go to GitHub:

```text
*.bam
*.bai
large backend *.npz files
large cached feature_cache_*.npz files
training checkpoints *.pt
large generated tensor chunks
Slurm stdout/stderr logs unless selected examples are needed
```

Suggested GitHub strategy:

- Commit scripts, small metadata TSVs, small summaries, and documentation.
- Put large BAM/NPZ/checkpoint artifacts in `.gitignore`.
- Provide a `README.md` explaining where raw data and generated outputs are expected.
- Add a small smoke-test dataset or tiny example manifest if possible.

## Suggested Clean Repository Structure

A cleaner final structure could be:

```text
thesis_dimelo/
  README.md
  docs/
    preprocessing_overview.md
    phase1_qc.md
    phase2_long_context.md
    phase3_hyena_dataset.md
  configs/
    marks.yaml
    paths.example.yaml
    preprocessing_thresholds.yaml
  src/
    preprocessing/
      phase1_qc/
      phase2_long_context/
      phase3_hyena/
    visualization/
    utils/
  scripts/
    run_phase1_qc.sh
    run_phase2_backend.sh
    build_phase3_index.sh
  metadata/
    intervals_long_context.tsv
    phase3_training_index.tsv
  notebooks/
    exploratory_archive/
  outputs/
    README.md
```

Possible archive folder for old experiments:

```text
archive/
  cnn_5mer_exploration/
  modkit_extract_full_notes/
  slurm_logs_selected/
```

## Key Implementation Details To Preserve

1. Chromosome split must stay fixed and documented:

```text
train = chr1-16
valid = chr17-18
test  = chr19-22 + chrX
```

2. Label convention must stay explicit:

```text
0 = unmethylated
1 = methylated
2 = unknown / masked
```

3. HyenaDNA preprocessing thresholds must be documented:

```text
min_mapq = 20
ml_threshold = 128
min_coverage = 3
methylated_frac_thr = 0.7
unmethylated_frac_thr = 0.3
```

4. The final pipeline should clearly say that the old CNN `mod_qual >= 0.8` threshold is not the final HyenaDNA label rule.

5. Mark-specific outputs should be indexed through manifests rather than hard-coded scattered paths.

6. Generated outputs should be reproducible from scripts, not manually copied into the source tree.

## One-Sentence Final Pipeline Summary

The final preprocessing pipeline reads mark-specific ONT DiMeLo-seq modBAM files, parses modified-base tags directly, partitions the genome into 1 Mb chromosome-split intervals, builds per-base C-methylation label arrays with explicit unknown masking for each histone mark, summarizes and validates those interval backends, and merges the mark-specific manifests into a Phase 3 HyenaDNA-compatible training index.
