# Preprocessing Pillar

Active workflow:

1. Phase 1 QC parses DiMeLo modBAM modified-base tags and summarizes read-level/binned signal.
2. Phase 2 builds 1 Mb long-context interval backends and mark-specific manifests.
3. Phase 3 merges backend manifests into the HyenaDNA-facing training index.

`OLD/` contains exploratory code kept for reference. `to_delete/` is a quarantine for unused old preprocessing and CNN-era work; nothing has been permanently deleted.
