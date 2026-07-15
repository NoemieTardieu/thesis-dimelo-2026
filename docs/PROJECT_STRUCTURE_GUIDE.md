# Thesis Project Structure Guide

This repository is organized around three thesis pillars:

- `preprocessing/`
- `hyena-dna/`
- `alphagenome/`

Everything thesis-related has been consolidated under:

```text
/data/leuven/383/vsc38330/thesis_project_clean/
```

The parent folder `/data/leuven/383/vsc38330/` is intentionally kept almost empty. Hidden runtime folders such as `.codex`, `.agents`, `.config`, `.cache`, `.ondemand`, and `.git`, plus `code-server-ipc.sock`, are not thesis content and were left in place to avoid breaking Codex, code-server, Git, or the active session.

## Top-Level Folders

```text
thesis_project_clean/
  preprocessing/
  hyena-dna/
  alphagenome/
  docs/
  configs/
  metadata/
  inventories/
  server_artifacts/
```

### `preprocessing/`

This pillar contains the DiMeLo preprocessing work used to produce the thesis-ready inputs for downstream modeling.

It keeps the final preprocessing route:

- Phase 1 quality control and exploratory summaries.
- Phase 2 long-context backend construction.
- Phase 3 HyenaDNA index/input preparation.
- Lightweight plots, summaries, manifests, and documentation.

It also contains preprocessing-related archive and quarantine folders:

- `preprocessing/OLD/`: older or uncertain preprocessing material that may still be historically useful.
- `preprocessing/to_delete/`: clearly unused, abandoned, generated, or failed preprocessing material kept only as a deletion quarantine.
- `preprocessing/server_artifacts/`: large server-local data and residual original project material that should not go to GitHub.

### `hyena-dna/`

This pillar contains thesis-specific HyenaDNA modeling, evaluation, interpretation, and results.

It keeps the final no-sample modeling and analysis route:

- `P(M | D)` models.
- `P(Reg | D, M)` models.
- overlap-aggregated evaluation.
- cross-chromosome generalization.
- threshold metrics.
- variance and paired-read interpretability analyses.

The original upstream HyenaDNA source tree is preserved under `hyena-dna/upstream_hyena_dna/` because thesis scripts may still depend on that source tree, configuration layout, or dependency shims.

Important archive/quarantine folders:

- `hyena-dna/OLD/`: older HyenaDNA experiments, notebooks, sample-conditioned models, debug runs, smoke tests, tiny runs, and superseded scripts that may still help explain the development path.
- `hyena-dna/to_delete/`: CNN work, discarded generated tensors, old residual work islands, SLURM files, cache folders, and other material not used in the final thesis.
- `hyena-dna/server_artifacts/`: checkpoints, upstream checkpoints, and other large local artifacts that are needed on the server but should not be versioned in GitHub.

### `alphagenome/`

This pillar contains AlphaGenome benchmarking, analysis, metadata, tests, and thesis-ready outputs.

The active thesis AlphaGenome result is the ENCSR203XPU 200 bp population-track benchmark. Secondary/exploratory outputs are kept separately when they are useful for the thesis narrative.

Important folders:

- `alphagenome/src/`: reusable AlphaGenome code.
- `alphagenome/scripts/`: runnable scripts and analysis entry points.
- `alphagenome/tests/`: lightweight unit tests.
- `alphagenome/metadata/`: selected track metadata and benchmark region definitions.
- `alphagenome/results/`: final and near-final thesis outputs.
- `alphagenome/archive/`: phase-organized historical outputs and scripts.
- `alphagenome/OLD/`: compatibility pointer only; the former OLD contents were reorganized.
- `alphagenome/to_delete/`: caches, SLURM files, old residual scaffolding, and discarded generated material.
- `alphagenome/server_artifacts/`: large generated tables and external data that should stay server-local.

### `docs/`

Global documentation lives here.

Useful files include:

- `root_cleanup_notes.md`: explains how the original messy root was consolidated.
- `superseded_and_failed_files_log.md`: records older files, failed routes, and why they were superseded.
- `PROJECT_STRUCTURE_GUIDE.md`: this file.

Pillar-specific documentation stays inside each pillar, for example:

- `alphagenome/docs/`
- `hyena-dna/docs/`
- `preprocessing/docs/`

### `configs/`

This is reserved for shared or top-level configuration files.

Pillar-specific configs should usually stay inside their pillar unless they are genuinely shared across the full thesis project.

### `metadata/`

This contains global metadata that applies across pillars.

The most important file is:

- `metadata/external_data_registry.tsv`

That registry records large files and server-resident data roots that are needed for reproducibility but are not intended for GitHub.

### `inventories/`

This folder records what was moved and why.

Important files:

- `moved_files.tsv`: all main file moves into the clean structure.
- `old_manifest.tsv`: files moved into `OLD/`.
- `to_delete_manifest.tsv`: files moved into `to_delete/`.
- `root_cleanup_manifest.tsv`: root-level cleanup moves.
- `alphagenome_old_reorg_manifest.tsv`: AlphaGenome OLD-to-archive reorganization.
- `restructure_scripts/`: scripts used during the cleanup.

The manifests are the audit trail. If something seems missing, check these before assuming it was deleted.

### `server_artifacts/`

This folder contains server-local material that belongs to the thesis project but should not be pushed to GitHub.

Examples include:

- local environments moved from the old root.
- local tool installs.
- large residual data.
- runtime or generated artifacts.

This folder is ignored by Git.

## Folder Meaning Rules

### Active folders

Folders such as `src/`, `scripts/`, `models/`, `evaluation/`, `analysis/`, `plotting/`, `metadata/`, and `results/` contain the active or thesis-relevant project state.

These are the folders to inspect first when writing the thesis, reproducing the final analyses, or preparing the GitHub version.

### `results/`

`results/` contains final or near-final thesis outputs that are small enough and meaningful enough to keep visible.

Large generated files are not kept here unless they are intentionally small summaries or selected figures. Heavy outputs should be in `server_artifacts/` or listed in `metadata/external_data_registry.tsv`.

### `archive/`

`archive/` is used when a pillar needs a structured historical record.

At the moment, this is most important for AlphaGenome. The old broad AlphaGenome `OLD/` folder was reorganized into phase-specific archive folders so that the development path is understandable.

Archived files are not the final thesis workflow, but they may explain:

- earlier benchmark phases.
- validation experiments.
- exploratory analyses.
- scripts or outputs that informed the final design.

### `OLD/`

`OLD/` means historical, superseded, uncertain, or potentially useful.

It does not mean trash.

Files in `OLD/` are kept because they may still help with:

- understanding how the final workflow evolved.
- recovering an earlier approach.
- checking why a route was abandoned.
- documenting thesis decisions.

Current pillar-specific meaning:

- `preprocessing/OLD/`: older preprocessing, modeling-prep, visualization, notebooks, and uncertain material that was not part of the final route but may be useful context.
- `hyena-dna/OLD/`: earlier HyenaDNA scripts, sample-conditioned work, debug/smoke runs, tiny experiments, notebooks, and superseded model variants.
- `alphagenome/OLD/`: contains only `README.md`. The previous contents were reorganized into `alphagenome/archive/`, `alphagenome/results/`, `alphagenome/metadata/`, and `alphagenome/server_artifacts/`.

### `to_delete/`

`to_delete/` is a quarantine folder.

Nothing in it has been permanently deleted. It is separated because it is not part of the final thesis structure and is likely safe to remove later after manual review.

Typical contents:

- SLURM job scripts and logs.
- Python cache folders.
- failed or abandoned route outputs.
- CNN-related work, since the CNN was not used in the final thesis.
- old generated tensors or checkpoints that are not final.
- command debris and accidental files.

### `server_artifacts/`

`server_artifacts/` means server-local, not GitHub-facing.

These files may still be important for reproducibility, but they are large, generated, environment-specific, or otherwise unsuitable for version control.

Examples:

- BAM/BigWig/NPZ/checkpoint-style data.
- original residual project directories with large backend files.
- upstream model checkpoints.
- local virtual environments or tool installs.

## Preprocessing Pillar

```text
preprocessing/
  src/
  docs/
  metadata/
  results/
  OLD/
  to_delete/
  server_artifacts/
```

The active preprocessing files live mainly in `src/`, with outputs and lightweight summaries in `results/` and reproducibility inputs in `metadata/`.

The final route is centered on:

- Phase 1 QC.
- Phase 2 long-context preprocessing.
- Phase 3 HyenaDNA input/index preparation.

`preprocessing/OLD/` contains older preprocessing attempts and uncertain but possibly useful scripts. These were kept because they may explain earlier design decisions or contain reusable snippets.

`preprocessing/to_delete/` contains abandoned preprocessing branches, command debris, SLURM files, and unused generated material. CNN-related preprocessing artifacts also belong here.

`preprocessing/server_artifacts/` contains server-resident material such as the residual original `thesis_dimelo` tree, large backend arrays, genome/index files, and local tool installs. These are intentionally kept out of GitHub.

## HyenaDNA Pillar

```text
hyena-dna/
  preprocessing/
  models/
  evaluation/
  analysis/
  plotting/
  results/
  metadata/
  docs/
  upstream_hyena_dna/
  OLD/
  to_delete/
  server_artifacts/
```

The active thesis HyenaDNA code is organized by workflow stage:

- `preprocessing/`: HyenaDNA-specific preprocessing and input preparation.
- `models/`: final model definitions and training/inference scripts.
- `evaluation/`: benchmark and metric scripts.
- `analysis/`: interpretability, variance, paired-read, and downstream analyses.
- `plotting/`: thesis figure generation.
- `results/`: selected thesis-ready outputs and lightweight summaries.
- `metadata/`: metadata needed to interpret or regenerate the analyses.
- `docs/`: HyenaDNA-specific work logs and notes.

`hyena-dna/upstream_hyena_dna/` preserves the upstream HyenaDNA repository. It is not treated as thesis-authored code, but it remains in the clean structure because the thesis scripts may rely on it.

`hyena-dna/OLD/` contains superseded model variants, debug scripts, sample-conditioned work, smoke tests, notebooks, and early exploratory runs.

`hyena-dna/to_delete/` contains CNN work, generated tensors/checkpoints that are not final, old residual work folders, SLURM scripts/logs, and cache directories.

`hyena-dna/server_artifacts/` contains large checkpoints and other local artifacts that are needed on the server but should not be committed.

## AlphaGenome Pillar

```text
alphagenome/
  src/
  scripts/
  tests/
  docs/
  metadata/
  results/
  archive/
  OLD/
  to_delete/
  server_artifacts/
```

The active AlphaGenome workflow is focused on the final ENCSR203XPU 200 bp benchmark.

Active folders:

- `src/`: reusable AlphaGenome benchmark and analysis code.
- `scripts/`: runnable scripts for generating outputs.
- `tests/`: lightweight unit tests.
- `metadata/`: selected metadata and benchmark regions.
- `results/`: final and near-final thesis results.
- `docs/`: phase maps, archive notes, and AlphaGenome-specific documentation.

Final or near-final result folders include:

- `results/population_track_benchmark_ENCSR203XPU_200bp_final/`: primary thesis AlphaGenome benchmark.
- `results/secondary_external_track_GSM2421502_1kb/`: selected secondary/exploratory 1 kb external-track summaries and thesis density figures.
- `results/readseq_summaries/`: small read-sequence sensitivity summaries.
- `results/alphagenome_readlevel_locus_example/`: same-locus read-level example.

### AlphaGenome `archive/`

The old broad AlphaGenome `OLD/` bucket was reorganized into `archive/` according to the phase map.

Current archive phases:

- `phase00_setup_smoke/`: setup checks, smoke tests, API probing, and early metadata discovery.
- `phase02_initial_128bp_reference_benchmark/`: initial 128 bp reference benchmark summaries.
- `phase03_validation_thresholds/`: threshold and validation experiments.
- `phase04_visual_promoter_followup/`: promoter-focused visual follow-up analyses.
- `phase05_200bp_pre_encode_followup/`: pre-final 200 bp benchmark work before the ENCODE-centered final run.
- `phase06_read_sequence_sensitivity/`: read-sequence sensitivity experiments.
- `phase07_readlevel_diagnostics/`: read-level diagnostic summaries and plots.
- `phase09_external_bigwig_exploratory/`: exploratory external BigWig benchmarks.
- `phase10_encode_benchmark_dev_runs/`: ENCSR quick, development, and non-final runs.
- `phase12_5mc_visual_check/`: auxiliary 5mC visual checks, not central to the final AlphaGenome thesis result.

This makes the historical AlphaGenome work easier to interpret without mixing it with the final result folders.

### AlphaGenome `OLD/`

`alphagenome/OLD/` now contains only `README.md`.

That README is a pointer explaining that the previous OLD contents were reorganized into:

- `alphagenome/archive/`
- `alphagenome/results/`
- `alphagenome/metadata/`
- `alphagenome/server_artifacts/`

### AlphaGenome `server_artifacts/`

Large generated AlphaGenome tables and external data live here instead of in Git-facing folders.

Examples include:

- per-bin benchmark TSVs.
- read-sequence TSVs.
- canonical raw/normalized bin tables.
- large per-read generated tables.
- external BigWig files.

## What Should Go To GitHub

Good GitHub candidates:

- active source code.
- scripts needed to reproduce final outputs.
- documentation.
- manifests.
- small metadata files.
- selected thesis-ready figures.
- small summary tables.

Do not commit:

- `server_artifacts/`.
- large BAM/CRAM/BigWig/NPZ/NPY/PT/checkpoint files.
- local virtual environments.
- SLURM logs.
- API keys or secrets.
- cache folders.
- command debris.

Review before committing:

- `OLD/`: useful historical material, but not always needed in a public GitHub release.
- `archive/`: meaningful historical AlphaGenome material, but not the active final workflow.
- `to_delete/`: quarantine only; generally not GitHub material.

## Audit Trail

The cleanup was move-first: files were moved into the clean structure rather than copied, and nothing was permanently deleted.

Use these files to trace decisions:

```text
inventories/moved_files.tsv
inventories/old_manifest.tsv
inventories/to_delete_manifest.tsv
inventories/root_cleanup_manifest.tsv
inventories/alphagenome_old_reorg_manifest.tsv
metadata/external_data_registry.tsv
docs/superseded_and_failed_files_log.md
```

Current manifest sizes after cleanup:

- `moved_files.tsv`: 346 rows.
- `old_manifest.tsv`: 668 rows.
- `to_delete_manifest.tsv`: 895 rows.
- `root_cleanup_manifest.tsv`: 36 rows.
- `alphagenome_old_reorg_manifest.tsv`: 127 rows.

## Validation Status

After the restructuring:

- Active Python syntax checks passed for 69 files.
- AlphaGenome unit tests passed: 9 tests run, 0 failures.
- `alphagenome/OLD/` contains only `README.md`.
- AlphaGenome benchmark region TSVs are in `alphagenome/metadata/regions/`.
- Large AlphaGenome generated tables were moved to `alphagenome/server_artifacts/alphagenome_archive_large_tables/`.

