# AlphaGenome Benchmark Handoff: chr16, chr11, chr17, chr19

## Ready-to-use request for the next Codex chat

Work in:

```text
/data/leuven/383/vsc38330/hyena-dna-main
```

Implement an AlphaGenome benchmark for the existing DiMeLo/HyenaDNA project.
Use the exact held-out test regions from chromosomes 16, 11, 17, and 19.
Query AlphaGenome's human hg38 model for histone ChIP predictions, select
H3K4me3 tracks using output metadata and an explicitly documented biosample
selection rule, aggregate the experimental DiMeLo 6mA signal to the same
reference coordinates and 128 bp resolution, and compare the tracks using
Pearson, Spearman, AUROC, and AUPRC. Do not put the API key in source code or
Git. Start with one test region as a smoke test, cache every API response, and
then process all 60 held-out regions. Read this entire document before editing
or running anything.

## Scientific objective

The current project models:

```text
P(Reg, M | D)
```

where:

- `D`: DNA sequence
- `M`: endogenous CpG methylation represented by modkit 5mC probabilities
- `Reg`: DiMeLo-derived adenine modification probability used as a read-level
  proxy for H3K4me3-associated regulatory signal

The promoter requested a state-of-the-art grounding experiment using Google
DeepMind AlphaGenome on the same held-out data.

The direct AlphaGenome comparison is:

```text
AlphaGenome H3K4me3 CHIP_HISTONE prediction
versus
aggregated experimental DiMeLo 6mA/H3K4me3-associated track
```

AlphaGenome does not provide a directly equivalent 5mC output. Therefore:

- the primary AlphaGenome benchmark is for the regulatory/6mA task;
- do not claim AlphaGenome predicts nanopore 6mA probabilities;
- describe this as cross-assay agreement between predicted H3K4me3 ChIP signal
  and the DiMeLo H3K4me3-associated adenine signal;
- optional DNase or ATAC outputs may be exploratory secondary comparisons, but
  they are not replacements for the primary H3K4me3 comparison.

## AlphaGenome facts that matter

Official resources:

- Repository: <https://github.com/google-deepmind/alphagenome>
- Documentation: <https://www.alphagenomedocs.com/>
- API key: <https://deepmind.google.com/science/alphagenome/>

As of June 2026:

- install with `pip install alphagenome` or from the official repository;
- create a client using `dna_client.create(API_KEY)`;
- use `dna_client.DnaClient.predict_interval` for reference intervals;
- human intervals default to hg38 and are zero-based, half-open;
- AlphaGenome accepts sequences up to 1,048,576 bp;
- request `dna_client.OutputType.CHIP_HISTONE`;
- `CHIP_HISTONE` output is fold-change over control, summed in 128 bp bins;
- output values are in `TrackData.values`;
- track descriptions are in `TrackData.metadata`;
- use `model.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)`
  to inspect tracks before choosing H3K4me3 and biosample filters;
- the API is intended for non-commercial, limited/medium-scale analyses.

Never commit the API key. Read it from an environment variable such as:

```bash
export ALPHAGENOME_API_KEY='...'
```

## Reference assembly

The BAM chromosome names and lengths match hg38/GRCh38. A local indexed hg38
reference is available at:

```text
/data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa
/data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa.fai
```

AlphaGenome intervals are zero-based and half-open, matching the BED region
coordinates used here.

## Chromosomes and evaluation design

Use these four chromosomes:

```text
chr16
chr11
chr17
chr19
```

Each chromosome has 100 selected 100 kb regions. The region split was performed
at genomic-region level using seed 7:

```text
70 train regions
15 validation regions
15 test regions
```

Only the 15 held-out `test` regions per chromosome should be used for the
primary AlphaGenome benchmark. This gives:

```text
4 chromosomes x 15 test regions = 60 AlphaGenome intervals
```

Do not evaluate on the complete chromosomes first. The 60 test intervals are
the fair comparison to the leakage-free HyenaDNA test sets and require only 60
API predictions.

Authoritative region-split files:

```text
preprocessing_chr16_merged_e5b/outputs/merged_e5b_chr16_selected_top100_overlap16k_full5000_region_split.region_splits.tsv
preprocessing_chr16_merged_e5b/outputs/merged_e5b_chr11_selected_top100_overlap16k_full5000_region_split.region_splits.tsv
preprocessing_chr16_merged_e5b/outputs/merged_e5b_chr17_selected_top100_overlap16k_full5000_region_split.region_splits.tsv
preprocessing_chr16_merged_e5b/outputs/merged_e5b_chr19_selected_top100_overlap16k_full5000_region_split.region_splits.tsv
```

Note: these TSV files contain carriage returns. Strip them when parsing or use
Python's CSV parser.

Generate one benchmark BED with:

```bash
cd /data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b

{
  for chrom in 16 11 17 19; do
    file="outputs/merged_e5b_chr${chrom}_selected_top100_overlap16k_full5000_region_split.region_splits.tsv"
    tr -d '\r' < "$file" |
      awk -F '\t' -v OFS='\t' 'NR > 1 && $6 == "test" {print $2,$3,$4,$5,$6}'
  done
} > outputs/alphagenome_4chrom_test_regions.bed

wc -l outputs/alphagenome_4chrom_test_regions.bed
# Expected: 60
```

Some region names on chr11/17/19 incorrectly begin with `chr16_rank...`.
Treat this as a naming artifact. The `chrom`, `start`, and `end` columns are
authoritative.

## Existing experimental test data

Combined E5B/C1 test tensors:

```text
preprocessing_chr16_merged_e5b/outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz
preprocessing_chr16_merged_e5b/outputs/merged_e5b_c1_chr11_selected_top100_overlap16k_full5000_region_split.test.npz
preprocessing_chr16_merged_e5b/outputs/merged_e5b_c1_chr17_selected_top100_overlap16k_full5000_region_split.test.npz
preprocessing_chr16_merged_e5b/outputs/merged_e5b_c1_chr19_selected_top100_overlap16k_full5000_region_split.test.npz
```

Corresponding metadata:

```text
preprocessing_chr16_merged_e5b/outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv
preprocessing_chr16_merged_e5b/outputs/merged_e5b_c1_chr11_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv
preprocessing_chr16_merged_e5b/outputs/merged_e5b_c1_chr17_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv
preprocessing_chr16_merged_e5b/outputs/merged_e5b_c1_chr19_selected_top100_overlap16k_full5000_region_split.test.metadata.tsv
```

Test-set sizes:

| Chromosome | Windows | Valid 5mC positions | Valid 6mA positions |
|---|---:|---:|---:|
| chr16 | 1,877 | 1,094,769 | 10,193,258 |
| chr11 | 1,393 | 737,013 | 7,387,341 |
| chr17 | 1,632 | 694,019 | 7,227,300 |
| chr19 | 1,550 | 987,781 | 8,214,234 |

Tensor arrays include:

```text
input_ids
target_5mC
mask_5mC
target_6mA
mask_6mA
sample_id
```

The tensors are in read coordinates. Reference-locus conversion must use BAM
alignment/CIGAR information; do not approximate a locus using
`alignment_start + tensor_position`.

BAM files:

```text
/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam
/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam
```

Existing overlap-aware evaluator demonstrating the correct read-position
deduplication logic:

```text
preprocessing_chr16_merged_e5b/evaluate_region_split_hyenadna_two_head_overlap_aggregated.py
```

It collapses duplicate observations from overlapping 32 kb windows by:

```text
sample / read_id / read_position
```

For AlphaGenome, go one step further:

1. collapse duplicated overlap-window observations within each original read;
2. map each read position through the BAM CIGAR to a reference position;
3. aggregate experimental 6mA probabilities across reads at each reference
   locus;
4. aggregate those locus values to the same 128 bp bins as AlphaGenome.

The recent locus-variance scripts contain reference-mapping and aggregation
ideas:

```text
preprocessing_chr16_merged_e5b/analyze_locus_variance.py
preprocessing_chr16_merged_e5b/analyze_full_chromosome_locus_variance.py
```

## Existing HyenaDNA benchmark

Current four-chromosome no-sample checkpoint:

```text
preprocessing_chr16_merged_e5b/outputs/hyenadna_small32k_4chrom_overlap16k_full5000_region_split_nosample_short_2epochs_1000batches.pt
```

Current overlap-aggregated per-chromosome summary:

```text
preprocessing_chr16_merged_e5b/outputs/4chrom_nosample_short_overlapagg_per_chrom_test_summary.tsv
```

6mA results:

| Chromosome | Pearson | Spearman | AUROC | AUPRC | Positive fraction |
|---|---:|---:|---:|---:|---:|
| chr16 | 0.2063 | 0.1649 | 0.6863 | 0.1174 | 0.0519 |
| chr19 | 0.1975 | 0.1532 | 0.6771 | 0.1188 | 0.0552 |
| chr11 | 0.2124 | 0.1839 | 0.6892 | 0.1198 | 0.0520 |
| chr17 | 0.1862 | 0.1449 | 0.6683 | 0.1023 | 0.0472 |

These metrics are based on per-read target probabilities after removing
duplicate overlap-window observations. An AlphaGenome 128 bp track comparison
is not numerically identical. For a fair direct comparison, also aggregate
HyenaDNA predictions and experimental targets to the same 128 bp reference
bins and recompute both models' metrics there.

## Required AlphaGenome workflow

### 1. Environment and authentication

- Create a dedicated environment if needed.
- Install the official `alphagenome` package.
- Read the API key from `ALPHAGENOME_API_KEY`.
- Add AlphaGenome result/cache directories and secrets to `.gitignore`.
- Never print the full API key in logs.

### 2. Inspect output metadata before querying all regions

Create the client and fetch human output metadata:

```python
from alphagenome.models import dna_client

model = dna_client.create(api_key)
metadata = model.output_metadata(
    organism=dna_client.Organism.HOMO_SAPIENS
)
```

Inspect all `CHIP_HISTONE` tracks whose mark/name corresponds to H3K4me3.
Record:

- exact metadata columns available in the installed API version;
- H3K4me3 track names;
- ontology terms;
- biosample/cell-type descriptions;
- assay source and strand, if relevant.

The biological identity of `merged_e5b` and `merged_c1` is not documented
clearly enough in this repository to choose an AlphaGenome ontology term
automatically. Ask the user or inspect project metadata before selecting the
closest AlphaGenome biosample.

Do not silently choose the best-correlating AlphaGenome track. That would be
test-set cherry-picking. Use one of these defensible rules:

1. preselect the biologically closest biosample based on known sample identity;
2. if no exact match exists, report a prespecified small panel of plausible
   tracks separately;
3. optionally compute a clearly labeled across-track mean, fixed before looking
   at test performance.

### 3. Smoke test one interval

Use one 100 kb held-out region. Example:

```python
from alphagenome.data import genome
from alphagenome.models import dna_client

interval = genome.Interval(
    chromosome="chr16",
    start=4_300_000,
    end=4_400_000,
)

outputs = model.predict_interval(
    interval=interval,
    ontology_terms=[SELECTED_ONTOLOGY_TERM],
    requested_outputs=[dna_client.OutputType.CHIP_HISTONE],
)
```

The exact return attribute and filtering syntax must be verified against the
installed AlphaGenome version. Save:

- raw returned values;
- complete track metadata;
- interval and resolution;
- package version;
- selected ontology terms;
- query timestamp;
- a diagnostic plot.

### 4. Process all 60 test regions

- Query each 100 kb test interval once.
- Cache each response immediately using a deterministic filename containing
  chromosome, start, end, output type, and ontology selection.
- Make the pipeline resumable and skip completed valid cache files.
- Add retry/backoff for transient API failures.
- Preserve raw metadata rather than saving only a selected vector.
- Validate that output interval and bin coordinates align with the requested
  interval.

### 5. Construct the experimental comparison track

Preferred target:

```text
mean experimental 6mA probability per 128 bp reference bin
```

Also save:

```text
number of unique reads per bin
number of observed adenine positions per bin
positive-read/position fraction at threshold 0.5
within-bin variance or standard error
sample-specific C1 and E5B tracks
pooled track, only as a secondary summary
```

Recommended primary comparisons:

- AlphaGenome versus C1 experimental track;
- AlphaGenome versus E5B experimental track;
- report pooled C1/E5B only as an additional result.

This avoids hiding sample-specific experimental behavior while still keeping
AlphaGenome independent of a sample-ID input.

### 6. Harmonize scales

AlphaGenome H3K4me3 values and DiMeLo probabilities have different units.
Therefore:

- Pearson and Spearman can be computed directly after alignment;
- for AUROC/AUPRC, define experimental positive bins in advance;
- do not apply the old per-base `target >= 0.5` rule blindly to a 128 bp mean;
- consider and document one primary bin-positive definition, such as:
  - mean 6mA probability above a biologically justified threshold;
  - positive fraction above a threshold;
  - top fixed percentile of experimental H3K4me3-associated signal;
- include sensitivity analyses for alternative thresholds;
- do not tune thresholds on the test regions.

A validation set may be used to select thresholds or transformations. The
held-out test regions must remain untouched until the rule is fixed.

### 7. Metrics

Report per region, per chromosome, and pooled across all test regions:

```text
Pearson
Spearman
AUROC
AUPRC
positive fraction / random AUPRC baseline
AUPRC enrichment = AUPRC / positive fraction
number of evaluated bins
coverage distribution
```

Use bootstrap confidence intervals by resampling genomic regions, not
individual bins, because adjacent bins are correlated.

For direct model comparison, produce one table containing:

```text
chromosome
model (AlphaGenome or HyenaDNA)
track/biosample
number of regions
number of bins
Pearson
Spearman
AUROC
AUPRC
AUPRC enrichment
```

### 8. Deliverables expected from the next chat

Create, at minimum:

```text
preprocessing_chr16_merged_e5b/alphagenome/
  README.md
  make_4chrom_test_regions.py
  inspect_alphagenome_h3k4me3_tracks.py
  run_alphagenome_smoke.py
  run_alphagenome_4chrom.py
  build_dimelo_128bp_tracks.py
  evaluate_alphagenome_vs_dimelo.py
  plot_alphagenome_comparison.py
  cache/
  outputs/
```

Also create:

- a requirements or environment file;
- a smoke-test command;
- a resumable full-run command;
- SLURM scripts only if API access works from compute nodes;
- a final TSV/JSON summary;
- regional example plots showing AlphaGenome and DiMeLo tracks together.

## Methodological cautions

1. **AlphaGenome is not a nanopore modification caller.** It predicts reference
   regulatory tracks learned from public functional-genomics assays.
2. **The two signal scales differ.** Do not compare their absolute MAE without
   defining a normalization/calibration procedure.
3. **Use hg38 coordinates exactly.** Check chromosome names and half-open
   intervals.
4. **Avoid overlap leakage.** Use only regions whose split is `test`.
5. **Avoid track selection on the test result.** Biosample and H3K4me3 track
   rules must be fixed beforehand.
6. **Account for 128 bp resolution.** Aggregate experimental and HyenaDNA
   results to precisely matching reference bins.
7. **Do not compare AlphaGenome to 5mC as if it predicts methylation.**
8. **Cache API responses.** Re-querying 60 intervals unnecessarily wastes quota.
9. **Record software/API version and metadata.** AlphaGenome outputs and client
   interfaces may evolve.
10. **Treat neighboring bins as correlated.** Bootstrap by region.

## Success criteria

The benchmark is complete when:

- exactly 60 held-out regions have valid cached AlphaGenome predictions;
- H3K4me3 track selection is biologically justified and documented;
- experimental DiMeLo signal is mapped to reference coordinates and binned at
  128 bp;
- AlphaGenome, HyenaDNA, and experimental targets share identical evaluated
  bins;
- results are reported per chromosome and pooled with region-level confidence
  intervals;
- all limitations of cross-assay comparison are stated explicitly.

