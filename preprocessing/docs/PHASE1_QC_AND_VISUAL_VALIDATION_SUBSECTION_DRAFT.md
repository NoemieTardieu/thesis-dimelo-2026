## Phase 1 QC and Visual Validation

### Methods (brief)
DiMeLo-seq modBAM files were first processed at read level to extract A- and C-channel modification summaries and standard QC features (mapping quality, read length, per-read modification densities). The initial QC outputs were then complemented with revised visual analyses motivated by promoter feedback. Specifically, joint distributions were re-plotted with contour-based topographic maps and marginal distributions, using three configurations: (i) binary methylation (`M` in {0,1}), (ii) methylation fraction, and (iii) Reg-density versus methylation fraction. Reg was represented either as normalized rank in [0,1] (comparability view) or as raw/log-transformed density (distributional view). Correlation summaries (Spearman, Pearson, mutual information) were exported per mark.

### Results (brief)
QC confirmed that data were usable for downstream modeling and that signal behavior differed across marks. The revised plots showed that binary methylation views are interpretable but compress structure, while methylation-fraction views recover more continuous patterns. Reg rank-normalization enabled cross-mark comparability but induces near-uniform marginals; therefore, raw/log Reg distribution plots were added to preserve biological density structure. Overlay contour plots enabled direct comparison between `h3k27ac`, `h3k27me3`, and `h3k4me3`, with clearer separation in continuous-M configurations. Together, these analyses provide sufficient Phase 1 validation to proceed with Phase 2/3 data-model integration.

### Interpretation Notes for Figures
- In per-read joint plots, one dot corresponds to one read.
- `log1p(x) = log(1 + x)` was only used for heavy-tailed raw-density visualization.
- Binary `M` emphasizes class balance; methylation fraction emphasizes graded structure.

### Status of Option 1 (peak-restricted view)
Peak-restricted plotting code is implemented and tested, but final biological interpretation requires mark-matched peak files for GM12878:
- H3K27ac peaks
- H3K27me3 peaks
- H3K4me3 peaks

Current local test used CTCF peaks (`ENCFF797SDL`), which validates the workflow technically but is not the final mark-matched analysis.

### Finalization Step Once Histone Peaks Are Added
Run:

```bash
/data/leuven/383/vsc38330/.venv/bin/python /data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase_1_visualization/phase1_feedback_viz.py \
  --peaks-bed-map h3k27ac=/path/to/h3k27ac_peaks.bed \
  --peaks-bed-map h3k27me3=/path/to/h3k27me3_peaks.bed \
  --peaks-bed-map h3k4me3=/path/to/h3k4me3_peaks.bed
```

This will produce peak-only per-mark and overlay figures in:
- `/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase_1_visualization/output_feedback`
