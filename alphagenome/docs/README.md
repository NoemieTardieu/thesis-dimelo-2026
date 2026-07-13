# AlphaGenome A549 H3K4me3 benchmark

This directory compares AlphaGenome's predicted A549 H3K4me3 ChIP-seq signal
with DiMeLo H3K4me3-associated adenine modification and HyenaDNA predictions.
It is a cross-assay agreement benchmark: AlphaGenome does **not** predict
nanopore 6mA probabilities.

## Environment and authentication

Install the pinned API client in a dedicated environment:

```bash
python -m pip install -r requirements.txt
export ALPHAGENOME_API_KEY='...'
```

The key is read only from the environment and is never written to output.
`cache/`, `outputs/`, and `.env*` are ignored by Git.

## 1. Freeze regions and A549 tracks

Run from this directory:

```bash
python make_4chrom_test_regions.py --split test --out outputs/4chrom_test_regions.tsv
python make_4chrom_test_regions.py --split val --out outputs/4chrom_val_regions.tsv
python inspect_alphagenome_h3k4me3_tracks.py
```

The metadata command writes all histone tracks, all H3K4me3 tracks, and the
tracks whose metadata explicitly identifies A549. Review
`outputs/metadata/selected_a549_h3k4me3_tracks.tsv` before querying. All
selected tracks are reported separately; their arithmetic mean is fixed in
advance as the primary AlphaGenome summary.

## 2. Query AlphaGenome

```bash
python run_alphagenome_smoke.py
python run_alphagenome_4chrom.py
python export_alphagenome_128bp_tracks.py
```

The full query can also run as a resumable WICE CPU job:

```bash
sbatch run_alphagenome_4chrom_wice.sbatch
```

The job sources the private key file, uses the AlphaGenome virtual
environment, skips the existing smoke cache, and exports the aligned track
after all 60 caches are valid.

Each interval is cached immediately in one atomic NPZ containing raw values,
complete returned track metadata, interval/resolution metadata, client version,
ontology terms, and query timestamp. Valid caches are skipped on restart.
Because AlphaGenome does not accept arbitrary 100 kb inputs, each benchmark
region is centered inside the smallest supported model interval (131,072 bp).
Expansion is clamped at chromosome boundaries using the hg38 FASTA index. The
original 100 kb region and expanded model interval are both recorded.
The export retains only returned 128 bp bins fully contained in each held-out
100 kb interval and creates the authoritative comparison grid.

## 3. Build experimental and model tracks

Validation thresholds are created without loading test tracks:

```bash
python build_dimelo_128bp_tracks.py \
  --split val --regions outputs/4chrom_val_regions.tsv
python fit_validation_thresholds.py \
  --validation-track outputs/dimelo_val_128bp.tsv
```

Run both validation steps on WICE with:

```bash
sbatch run_dimelo_validation_thresholds_wice.sbatch
```

Build test tracks on the exact AlphaGenome grid:

```bash
python build_dimelo_128bp_tracks.py \
  --split test --regions outputs/4chrom_test_regions.tsv \
  --grid outputs/alphagenome_test_128bp.tsv

python build_hyenadna_128bp_tracks.py \
  --split test --regions outputs/4chrom_test_regions.tsv \
  --grid outputs/alphagenome_test_128bp.tsv --device cuda
```

Build the test DiMeLo track on WICE with:

```bash
sbatch run_dimelo_test_128bp_wice.sbatch
```

Build the HyenaDNA test track on an H100 with:

```bash
sbatch run_hyenadna_test_128bp_wice.sbatch
```

Both builders collapse duplicate overlap-window observations by
sample/read/read-position, map forward read positions through primary BAM CIGAR
alignments, and aggregate identical reference bins. C1 and E5B remain separate
primary A549-clone tracks; `pooled` is secondary.

## 4. Evaluate and plot

```bash
python evaluate_alphagenome_vs_dimelo.py \
  --alphagenome outputs/alphagenome_test_128bp.tsv \
  --dimelo outputs/dimelo_test_128bp.tsv \
  --hyenadna outputs/hyenadna_test_128bp.tsv \
  --thresholds outputs/validation_thresholds.json

python plot_alphagenome_comparison.py \
  --alphagenome outputs/alphagenome_test_128bp.tsv \
  --dimelo outputs/dimelo_test_128bp.tsv \
  --hyenadna outputs/hyenadna_test_128bp.tsv
```

Run the final evaluation, 2,000 region-level bootstrap replicates, and plots on
WICE with:

```bash
sbatch run_final_benchmark_wice.sbatch
```

The primary positive label uses the validation-derived 90th percentile of the
pooled DiMeLo signal. Validation-derived 95th and 80th percentiles are
sensitivity analyses. Primary confidence intervals use 2,000 region-level
bootstrap replicates. Results are emitted per region, per chromosome, and
pooled across chromosomes.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover CRLF parsing, overlap collapse, reverse-strand mapping,
indels/soft-clips, atomic cache validation, and a synthetic BAM-to-128 bp
integration case.
