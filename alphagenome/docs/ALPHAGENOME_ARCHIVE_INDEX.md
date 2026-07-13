# AlphaGenome Archive Index

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
