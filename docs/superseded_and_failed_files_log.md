# Superseded And Failed Files Log

This restructure uses manifests as the detailed log of historical files:

- `inventories/to_delete_manifest.tsv`: unused, failed, generated, SLURM, CNN-era, cache, and other quarantined files.
- `inventories/old_manifest.tsv`: historically useful or uncertain files kept outside the active workflow.
- `inventories/moved_files.tsv`: active files moved into the clean project tree.

Main decisions:

- CNN/5-mer workflows were not used in the final thesis and were quarantined in pillar-local `to_delete/`.
- Sample-conditioned, tiny/debug/smoke, and early HyenaDNA iterations were moved to `hyena-dna/OLD/`.
- AlphaGenome quick/smoke/dry-run/cache outputs were moved out of the active tree.
- Large external data remain on server paths and are documented in `metadata/external_data_registry.tsv`.
