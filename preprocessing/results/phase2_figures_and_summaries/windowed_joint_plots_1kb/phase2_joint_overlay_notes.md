# Phase 2 Joint Overlay Plots

- Input source: cached 1 kb Phase 2 feature arrays from `windowed_enrichment_1kb_combined`
- Each point corresponds to one 1 kb genomic bin passing the Phase 2 C-coverage filter
- Reg feature: `a_mod_per_kb`
- M feature: `c_meth_frac`
- Option 2 overlay: Reg normalized to [0,1] vs methylation fraction
- Option 3 overlay: `log1p(a_mod_per_kb)` vs methylation fraction
- Output: /data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/windowed_joint_plots_1kb/phase2_option2_overlay_regNorm_vs_mfrac_by_mark.svg
- Output: /data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/windowed_joint_plots_1kb/phase2_option3_overlay_log1pReg_vs_mfrac_by_mark.svg
