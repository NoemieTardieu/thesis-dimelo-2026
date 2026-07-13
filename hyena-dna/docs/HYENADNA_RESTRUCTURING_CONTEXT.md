# HyenaDNA / DiMeLo Thesis Project Context Summary

## Project Location

Main repository:

`/data/leuven/383/vsc38330/hyena-dna-main`

Main working folder for the HyenaDNA/DiMeLo experiments:

`/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b`

Most outputs are currently inside:

`/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/outputs`

The repository was originally HyenaDNA, but I added a substantial downstream pipeline for DiMeLo-seq preprocessing, tensor generation, model training, evaluation, variance analysis, interpretability and plotting.

I want to reorganize everything cleanly for GitHub. Heavy output files should not be committed, but key scripts, lightweight summary TSV/JSON/Markdown files, and figure-generation code should be preserved.

## Scientific Objective

The overall biological question is whether HyenaDNA can predict methylation and DiMeLo-derived regulatory signal from long-read nanopore data.

Definitions:

- `D`: DNA sequence
- `M`: endogenous CpG methylation, 5mC
- `Reg`: DiMeLo-derived 6mA signal, used as proxy for H3K4me3-associated regulatory activity
- `C`: sample/condition, initially `merged_e5b` and `merged_c1`, but later removed from the model because sample conditioning is not realistic at inference time

The final modeling direction became:

1. `P(M | D)`  
   Predict CpG methylation from DNA sequence.

2. `P(Reg | D, M)`  
   Predict DiMeLo regulatory 6mA signal from DNA sequence plus observed/estimated 5mC context.

This was motivated by the observation that `P(Reg | D)` and `P(Reg, M | D)` were possible but only weakly predictive for read-level 6mA, because 6mA has substantial read-to-read heterogeneity.

## Data Sources

Main BAM files:

- `/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam`
- `/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam`

Reference genome:

- `/data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa`

Modkit extract-full files were used to create labels. Important C1 extract-full files include:

- `/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/by_chrom/extract_full_chr16.tsv.gz`
- `/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/by_chrom/extract_full_chr11.tsv.gz`
- `/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/by_chrom/extract_full_chr17.tsv.gz`
- `/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_c1/by_chrom/extract_full_chr19.tsv.gz`

The working chromosomes were:

- chr16: main training chromosome
- chr11, chr17, chr19: held-out cross-chromosome generalization chromosomes

## Preprocessing Pipeline

### Tensor Generation

Main script:

`make_chr16_dimelo_tensors.py`

Purpose:

- Convert aligned BAM/modkit-derived calls into HyenaDNA-style tensors.
- Each tensor contains:
  - `input_ids`: DNA sequence tokens
  - `target_5mC`
  - `mask_5mC`
  - `target_6mA`
  - `mask_6mA`
  - optionally `sample_id`

Labels are sparse:

- 5mC loss/metrics are computed only where CpG/5mC labels exist.
- 6mA loss/metrics are computed only where adenine/6mA labels exist.
- Missing labels are not treated as zero.

### Long-Read Overlap Windowing

HyenaDNA-small-32k can only process sequences of length 32768 bp.

To avoid losing information from ONT reads longer than 32 kb, long reads were split into overlapping windows:

- window length: 32768 bp
- overlap: 16384 bp

This preserves more of each long read than truncating to the first 32 kb.

Important output examples:

- `outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.train.npz`
- `outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.val.npz`
- `outputs/merged_e5b_c1_chr16_selected_top100_overlap16k_full5000_region_split.test.npz`

Analogous files exist for chr11, chr17 and chr19.

### Region-Level Train/Val/Test Splitting

Main scripts:

- `create_region_split_tensors.py`
- `combine_sample_split_tensors.py`

Purpose:

- Avoid leakage from overlapping or neighboring reads.
- Split by genomic region, not by individual read/window.

Split proportions:

- 70% train
- 15% validation
- 15% test

The two samples, `merged_e5b` and `merged_c1`, were combined while preserving sample identity initially. Later no-sample models became preferred.

### Overlap Aggregation During Evaluation

When long reads were split into overlapping windows, the same read/base could appear in more than one window.

To avoid double-counting overlapping windows, evaluation was modified to aggregate predictions by:

`sample / read_id / read_position`

This removed duplicated overlap observations before computing metrics.

Example overlap aggregation on chr16 test for no-sample model:

- raw 6mA window valid positions: about 10.19M
- aggregated read positions: about 5.87M
- duplicate overlap observations removed: about 4.32M

This correction was important because the promoter asked whether overlap windows from the same read were aggregated before final scoring.

## Main Model Families

### Earlier Model: `P(Reg, M | D, C)`

Script:

`train_region_split_hyenadna_two_head_sample_conditioned.py`

Evaluation:

`evaluate_region_split_hyenadna_two_head_sample_conditioned_extended.py`

Architecture:

- HyenaDNA-small-32k pretrained backbone
- original head removed
- sample embedding for `merged_e5b` vs `merged_c1`
- two task-specific heads:
  - 5mC head
  - 6mA head
- last HyenaDNA block and final normalization optionally unfrozen

This model performed reasonably, but promoter noted sample conditioning was not realistic at inference time unless sample encodes truly different assays/modifications. Therefore no-sample models became preferred.

### No-Sample `P(Reg, M | D)`

Sample conditioning was removed.

This addressed the promoter’s comment:

> sample conditioning is not realistic, because at inference time you cannot capture sample-specific biases.

The no-sample model had similar performance for 5mC and slightly lower/mixed performance for 6mA, but was more scientifically appropriate.

### `P(M | D)` Model

Main script:

`train_region_split_hyenadna_5mc_only.py`

Evaluation:

`evaluate_region_split_hyenadna_5mc_only_overlap_aggregated.py`

Purpose:

- Predict CpG 5mC from DNA sequence only.
- This became the first component of a chain-rule style model.

Training improvements:

- no sample conditioning
- longer training than smoke tests
- frequent training logs every N batches
- best checkpoint saving
- early stopping
- lower learning rate experiments
- overlap-aggregated evaluation

Main conclusion:

- `P(M | D)` learns a reasonable signal.
- 5mC prediction is easier than 6mA prediction.
- Sequence contains substantial information for CpG methylation.

### `P(Reg | D, M)` Model

Main script:

`train_region_split_hyenadna_6ma_methyl_conditioned_nosample.py`

Evaluation:

`evaluate_region_split_hyenadna_6ma_methyl_conditioned_nosample_overlap_aggregated.py`

Architecture:

- HyenaDNA-small-32k backbone
- no sample conditioning
- input features:
  - DNA sequence
  - observed 5mC values filled with zero where unobserved
  - 5mC observation mask
- output:
  - 6mA / regulatory signal
- decoder:
  - MLP head
  - best balanced model used hidden dim 256 and dropout 0.15
- last HyenaDNA block and final layer norm unfrozen

Best balanced checkpoint:

`outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_lastblock_25epochs_1000batches.best.pt`

Best balanced chr16 test metrics, overlap-aggregated:

- MAE: 0.1103
- Pearson: 0.2276
- Spearman: 0.2082
- AUROC: 0.6927
- AUPRC: 0.1216
- positive fraction: 0.0519
- AUPRC enrichment: 2.35x

Interpretation:

- The model learns real signal above random.
- But read-level 6mA prediction remains moderate/hard.
- DNA + 5mC are not sufficient to strongly explain all read-level regulatory variation.

### Positive/Focal-Loss Experiment

The 6mA signal is sparse, with only about 5% positive positions.

A final improvement was tested:

- `--pos-weight-6ma 2.0`
- `--focal-gamma-6ma 1.0`

Checkpoint:

`outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_posw2_focal1_lastblock_22epochs_1000batches.best.pt`

Test metrics:

- MAE: 0.1716
- Pearson: 0.2042
- Spearman: 0.1380
- AUROC: 0.7009
- AUPRC: 0.1245
- mean_pred: 0.1748
- mean_target: 0.0832

Interpretation:

- AUROC and AUPRC slightly improved.
- Calibration and correlation worsened.
- The model overpredicted 6mA.
- Therefore, imbalance alone is not the main limitation.
- The balanced low-LR MLP256 model remains the preferred final model.

## Cross-Chromosome Generalization

The final balanced `P(Reg | D, M)` model was trained on chr16 and evaluated on held-out chromosomes chr11, chr17 and chr19.

Script used:

`evaluate_region_split_hyenadna_6ma_methyl_conditioned_nosample_overlap_aggregated.py`

Final cross-chromosome metrics:

| chromosome | Pearson | Spearman | AUROC | AUPRC | pos_frac | AUPRC enrichment |
|---|---:|---:|---:|---:|---:|---:|
| chr16 | 0.2276 | 0.2082 | 0.6927 | 0.1216 | 0.0519 | 2.35x |
| chr11 | 0.2303 | 0.2235 | 0.6939 | 0.1221 | 0.0520 | 2.35x |
| chr17 | 0.2050 | 0.1829 | 0.6749 | 0.1068 | 0.0472 | 2.26x |
| chr19 | 0.2178 | 0.1989 | 0.6844 | 0.1228 | 0.0552 | 2.22x |

Conclusion:

- The model generalizes to other chromosomes.
- It is not just memorizing chr16-specific regions.
- However, performance remains moderate, so read-level 6mA is only partially explained by DNA + 5mC.

Plotting scripts:

- `plot_reg_given_m_crosschrom_auroc_auprc.py`
- `plot_reg_given_m_roc_pr_curves.py`

Generated plots:

- `outputs/reg_given_m_crosschrom_auroc_auprc.png`
- `outputs/reg_given_m_crosschrom_auroc_auprc.svg`
- `outputs/reg_given_m_crosschrom_auroc_auprc.pdf`

ROC/PR curve outputs:

- `outputs/reg_given_m_crosschrom_roc_pr_curves.chr16.roc_pr.png`
- `outputs/reg_given_m_crosschrom_roc_pr_curves.chr11.roc_pr.png`
- `outputs/reg_given_m_crosschrom_roc_pr_curves.chr17.roc_pr.png`
- `outputs/reg_given_m_crosschrom_roc_pr_curves.chr19.roc_pr.png`
- `outputs/reg_given_m_crosschrom_roc_pr_curves.combined_roc_pr.png`
- `outputs/reg_given_m_crosschrom_roc_pr_curves.curve_summary.tsv`

## Threshold Metrics / F1

Because 6mA positives are rare, AUPRC is the main metric. F1 was added as a secondary threshold-dependent metric.

Script:

`evaluate_reg_given_m_threshold_metrics.py`

Logic:

1. Generate overlap-aggregated predictions.
2. Binarize target 6mA with `target >= 0.5`.
3. Scan prediction thresholds on validation.
4. Choose best F1 threshold on validation.
5. Apply that threshold to test.
6. Also report fixed threshold `0.5`.

Purpose:

- F1 is valid for imbalanced data, but threshold-dependent.
- It complements AUPRC and AUROC.
- It should not replace AUPRC as the main sparse-signal metric.

Completed F1/threshold analysis for the final balanced `P(Reg | D, M)` model:

Script:

`evaluate_reg_given_m_threshold_metrics.py`

Checkpoint:

`outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_lastblock_25epochs_1000batches.best.pt`

Output files:

- `outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_threshold_metrics.threshold_metrics.tsv`
- `outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_threshold_metrics.threshold_metrics.json`
- `outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_threshold_metrics.val_threshold_scan.tsv`

The prediction threshold that maximized F1 on validation was:

- threshold: 0.117

Validation metrics at threshold 0.117:

- precision: 0.1167
- recall: 0.2750
- F1: 0.1638
- predicted positive fraction: 0.1178
- true positive fraction: 0.0500

Test metrics using the validation-selected threshold 0.117:

- precision: 0.1238
- recall: 0.3009
- F1: 0.1755
- specificity: 0.8835
- predicted positive fraction: 0.1260
- true positive fraction: 0.0519

Test metrics using fixed threshold 0.5:

- precision: 0.3861
- recall: 0.0007
- F1: 0.0014
- predicted positive fraction: 0.000096

Interpretation:

- A fixed prediction threshold of 0.5 is not useful for this sparse 6mA task because almost no positions are predicted positive.
- A validation-selected threshold gives non-trivial F1 and recall, but precision remains low and the model predicts more positives than truly present.
- This reinforces the main conclusion that the model learns a weak-to-moderate ranking signal, but thresholded read-level 6mA classification remains difficult.
- AUPRC remains the more informative primary metric for the imbalanced 6mA task.

## Variance Analysis

### Read-to-Read Variance Across Loci

Main script:

`analyze_full_chromosome_locus_variance.py`

Purpose:

- Quantify how much 5mC and 6mA vary between reads covering the same reference-genome locus.
- This was used to test whether read-level prediction from sequence is realistic.

Important output for C1 chr16:

`outputs/full_chr16_c1_locus_variance.per_locus_variance.tsv.gz`

Associated audit:

- `verify_locus_variance_thresholds.py`
- `outputs/full_chr16_c1_locus_variance_threshold_audit.audit.tsv`
- `outputs/full_chr16_c1_locus_variance_threshold_audit.threshold_density_ecdf.png`
- `outputs/full_chr16_c1_locus_variance_threshold_audit.threshold_count_histograms.png`

Key interpretation:

- There is substantial read-to-read variability.
- 6mA is sparse and heterogeneous.
- This partly explains why `P(Reg | D)` and `P(Reg | D, M)` are hard at read level.

### CIGAR / DNA Heterogeneity Analysis

Main script:

`analyze_variance_loci_cigar_dna.py`

Purpose:

- Respond to promoter’s question:
  - Are high-variance loci explained by underlying sequenced DNA or alignment differences?
- Selected high- and low-variance loci.
- Compared reads for:
  - mismatch rate
  - indels
  - soft clipping
  - MAPQ
  - NM tag
  - strand balance
  - read base vs reference

Important output:

- `outputs/full_chr16_c1_variance_cigar_dna_top500_bottom500.selected_loci.tsv`
- `outputs/full_chr16_c1_variance_cigar_dna_top500_bottom500.per_read.tsv.gz`
- `outputs/full_chr16_c1_variance_cigar_dna_top500_bottom500.per_locus_summary.tsv`
- `outputs/full_chr16_c1_variance_cigar_dna_top500_bottom500.group_summary.tsv`
- `outputs/full_chr16_c1_variance_cigar_dna_top500_bottom500.cigar_dna_plots.png`
- `outputs/full_chr16_c1_variance_cigar_dna_top500_bottom500.summary.json`

Conclusion:

- For 6mA, high-variance loci did not show a strong alignment-artifact signature.
- Mapping quality and mismatch rates did not explain the 6mA heterogeneity.
- Therefore, 6mA read-level variation is not simply due to obvious CIGAR/alignment artifacts.

## Relationship Between 5mC and 6mA

Main script:

`analyze_final_m_reg_relationship.py`

Purpose:

- Analyze correlation/co-occurrence between observed 5mC and observed 6mA.
- Done on the actual model tensors, including overlap-collapsed read positions.

Important output for chr16 val/test:

- `outputs/final_m_reg_relationship_chr16_val_test_overlapagg.window.tsv`
- `outputs/final_m_reg_relationship_chr16_val_test_overlapagg.read_overlap_collapsed.tsv`
- `outputs/final_m_reg_relationship_chr16_val_test_overlapagg.region.tsv`
- `outputs/final_m_reg_relationship_chr16_val_test_overlapagg.correlation_summary.tsv`
- `outputs/final_m_reg_relationship_chr16_val_test_overlapagg.read_overlap_collapsed.cooccurrence.tsv`
- `outputs/final_m_reg_relationship_chr16_val_test_overlapagg.hexbin.png`

Findings:

- 5mC and 6mA were negatively correlated in the chr16 val/test set.
- Read-overlap-collapsed correlation:
  - Pearson around -0.706
  - Spearman around -0.698
- This supports including methylation information in `P(Reg | D, M)`.

Also ran 4-chrom test relationship analysis:

- `outputs/final_m_reg_relationship_4chrom_test_overlapagg.*`

## Interpretability Analyses

Two interpretability analyses were done.

### 1. Simple Read-Pair Example

Script:

`select_locus_read_pair_interpretability.py`

Purpose:

- Pick two reads at the same locus with very different observed 6mA.
- Compare:
  - DNA sequence differences
  - observed 6mA
  - predicted 6mA from `P(Reg | D, M)`
  - local observed 5mC

Important outputs:

- `outputs/chr16_c1_read_pair_interpretability_with_reg_predictions_min6ma001.interpretability_pairs.png`
- `outputs/chr16_c1_read_pair_interpretability_with_reg_predictions_min6ma001.selected_pairs.tsv`
- `outputs/chr16_c1_read_pair_interpretability_with_reg_predictions_min6ma001.dna_context_alignment.tsv`
- `outputs/chr16_c1_read_pair_interpretability_with_reg_predictions_min6ma001.summary.json`

Example result:

- same locus, one read with high observed 6mA and another with much lower observed 6mA
- model predictions were much closer to each other than the observed values
- this suggests the model predicts smoother regulatory propensity, not exact read-level heterogeneity

### 2. Systematic Paired-Read Interpretability Analysis

Script:

`paired_read_interpretability_analysis.py`

Purpose:

Compare pairs of reads at the same genomic locus and classify them into:

- Case A: similar DNA, different observed 6mA
- Case B: different DNA, similar observed 6mA
- Case C: different DNA, different observed 6mA
- Case D: similar DNA, similar observed 6mA

The analysis computes:

- DNA distance
- observed 6mA pairwise MAE
- predicted 6mA pairwise MAE
- observed 5mC pairwise MAE
- predicted 5mC if available
- pairwise correlation distances
- mismatch-to-reference rate
- high substitution-distance flags
- per-position DNA audit for selected pairs

Important corrections added:

- reverse-strand reads are complemented for reference-oriented DNA comparison
- bases are compared only at the same GRCh38 positions
- selected pairs receive per-position audit output
- Case B is not selected if DNA audit fails / substitution distance too high
- prediction panels are omitted when predictions are unavailable
- pair dependence is reported by unique reads/loci
- grouped/leave-one-locus-out analyses are descriptive if too few loci

Important outputs from medium run:

- `outputs/chr16_c1_paired_read_interpretability_observed_medium.all_pairs.tsv`
- `outputs/chr16_c1_paired_read_interpretability_observed_medium.selected_pairs.tsv`
- `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot1_dna_vs_observed6ma.png`
- `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot2_observed_vs_predicted6ma.png`
- `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot3_dna_vs_6ma_colored_5mc.png`
- `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot4_observed5mc_vs_observed6ma.png`
- `outputs/chr16_c1_paired_read_interpretability_observed_medium.pair_A_lowDNA_high6mA_1.png`
- `outputs/chr16_c1_paired_read_interpretability_observed_medium.pair_B_highDNA_low6mA_2.png`
- `outputs/chr16_c1_paired_read_interpretability_observed_medium.report.md`

Final corrected analysis script:

`paired_read_interpretability_analysis.py`

Cross-run combiner:

`summarize_paired_read_interpretability_runs.py`

Purpose:

- Eventually combine paired-read analysis across chromosomes.

## Centromere Negative Control

A chr16 centromere region was tested as a negative-control region.

Initial observation:

- 6mA signal was very low:
  - mean around 0.03
  - median 0

Initial model performance was close to random:

- AUROC around 0.51
- Pearson around 0
- Spearman around 0

Promoter comment:

- Low AUROC alone is not proof that the model avoids hallucinating signal.
- If signal is zero everywhere, predicting zero can be perfect.
- Better control:
  - include centromere regions in training
  - test on centromeres from other chromosomes

This was noted as an important future/secondary validation, but not the main final model result.

## Promoter Comments and Responses

### Comment 1: sample conditioning is unrealistic

Response:

- Removed sample conditioning.
- Final models are no-sample.
- If samples represent different histone modifications in future, better design would be separate output heads per modification, not sample embeddings.

### Comment 2: centromere AUROC 0.51 is not sufficient evidence

Response:

- Acknowledged.
- Centromere analysis should be treated as preliminary.
- Better future approach: include centromere regions in training and hold out centromeres from other chromosomes.

### Comment 3: overlap windows from same read should be aggregated

Response:

- Implemented overlap-aggregated evaluation by `sample/read_id/read_position`.
- All final metrics use overlap aggregation.

### Comment 4: methylation conditioning should help if 5mC and 6mA are associated

Response:

- Implemented no-sample `P(Reg | D, M)`.
- Found modest improvement over `P(Reg | D)`.
- Relationship analysis confirmed 5mC/6mA association, mostly negative in current data.

### Comment 5: compare with AlphaGenome / SoTA

Response:

- Separate AlphaGenome analysis was done in another folder/conversation.
- HyenaDNA results should be compared to AlphaGenome population-track benchmark.

## Final Scientific Conclusions

### Main Conclusion for `P(M | D)`

- CpG methylation is reasonably predictable from DNA sequence.
- 5mC has stronger sequence-associated structure than 6mA.
- `P(M | D)` is a sensible model and can be used as part of a chain-rule formulation.

### Main Conclusion for `P(Reg | D, M)`

- 6mA/regulatory signal is learnable above random, but only moderately.
- Best AUPRC is about 0.12, with positive fraction around 0.05.
- This corresponds to AUPRC enrichment around 2.2-2.35x random baseline.
- Cross-chromosome generalization is stable, meaning the model learns transferable signal.
- But performance remains modest, indicating read-level 6mA is not fully explained by DNA + 5mC.

### Interpretation

The model likely learns a regulatory propensity signal rather than exact read-level regulatory state.

The limitation is likely due to:

- sparse 6mA positives
- read-level heterogeneity
- biological variability not captured by local DNA/5mC alone
- missing chromatin context
- experimental noise
- possible need for locus-level aggregation or external regulatory annotations

### Final Model Recommendation

Use as final balanced model:

`P(Reg | D, M)` no-sample MLP256 low-LR model:

`outputs/hyenadna_small32k_chr16_overlap16k_full5000_reg_given_m_nosample_mlp256_lowlr_lastblock_25epochs_1000batches.best.pt`

Use positive/focal model only as an ablation showing imbalance weighting slightly increases AUROC/AUPRC but harms calibration/correlation.

## Important Scripts to Preserve

### Preprocessing / Tensor Generation

- `make_chr16_dimelo_tensors.py`
- `create_region_split_tensors.py`
- `combine_sample_split_tensors.py`
- `prepare_chr19_chr11_chr17_overlap16k_full5000_region_splits.sh`

### Main Training Scripts

- `train_region_split_hyenadna_5mc_only.py`
- `train_region_split_hyenadna_6ma_methyl_conditioned_nosample.py`
- `train_region_split_hyenadna_two_head_sample_conditioned.py`
- `train_region_split_hyenadna_6ma_methyl_conditioned.py`

### Main Evaluation Scripts

- `evaluate_region_split_hyenadna_5mc_only_overlap_aggregated.py`
- `evaluate_region_split_hyenadna_6ma_methyl_conditioned_nosample_overlap_aggregated.py`
- `evaluate_region_split_hyenadna_two_head_sample_conditioned_extended.py`
- `evaluate_reg_given_m_threshold_metrics.py`

### Analysis Scripts

- `analyze_full_chromosome_locus_variance.py`
- `verify_locus_variance_thresholds.py`
- `analyze_variance_loci_cigar_dna.py`
- `analyze_final_m_reg_relationship.py`
- `select_locus_read_pair_interpretability.py`
- `paired_read_interpretability_analysis.py`
- `summarize_paired_read_interpretability_runs.py`

### Plotting Scripts

- `plot_hyenadna_epoch_losses.py`
- `plot_reg_given_m_crosschrom_auroc_auprc.py`
- `plot_reg_given_m_roc_pr_curves.py`

## Important Lightweight Result Files to Preserve

These are useful for thesis/GitHub and small enough to keep if not too large:

- `outputs/reg_given_m_crosschrom_generalization_metrics.tsv`
- `outputs/reg_given_m_crosschrom_auroc_auprc.png/svg/pdf`
- `outputs/reg_given_m_crosschrom_roc_pr_curves.curve_summary.tsv`
- `outputs/final_m_reg_relationship_chr16_val_test_overlapagg.correlation_summary.tsv`
- `outputs/final_m_reg_relationship_chr16_val_test_overlapagg.read_overlap_collapsed.cooccurrence.tsv`
- `outputs/chr16_overlap16k_full5000_sample_vs_nosample_test_summary.tsv`
- `outputs/chr16_overlap16k_full5000_nosample_window_vs_overlapagg_test_summary.tsv`
- `outputs/chr16_overlap16k_full5000_decoder_smoke_comparison.tsv`
- `outputs/chr16_overlap_overnight_fair_overlap_cross_chrom_test_summary.tsv`
- paired-read selected pair summaries and reports
- variance audit summary TSV/JSON files

Heavy files that should likely be excluded from GitHub:

- `.npz` tensor files
- `.pt` checkpoint files
- large `.tsv.gz` per-locus/per-read tables
- large slurm logs unless summarized
- large BAM/extract-full files, which are external inputs

## Suggested New Repository Structure

A clean future structure could be:

```text
hyena-dna-main/
  README.md
  DIMELO_HYENADNA_WORK_LOG.txt
  dimelo_hyenadna/
    preprocessing/
      make_dimelo_tensors.py
      create_region_split_tensors.py
      combine_sample_split_tensors.py
    models/
      train_5mc_only.py
      train_reg_given_m.py
      train_two_head_baseline.py
    evaluation/
      eval_5mc_overlap_aggregated.py
      eval_reg_given_m_overlap_aggregated.py
      eval_threshold_metrics.py
    analysis/
      analyze_locus_variance.py
      analyze_cigar_dna_variance.py
      analyze_m_reg_relationship.py
      paired_read_interpretability.py
    plotting/
      plot_losses.py
      plot_crosschrom_auroc_auprc.py
      plot_roc_pr_curves.py
    scripts/
      slurm/
        train/
        eval/
        analysis/
  results/
    summaries/
    figures/
  docs/
    methods_summary.md
    model_summary.md
    promoter_feedback_responses.md
  .gitignore
```

The restructuring should preserve the original HyenaDNA source files but separate my thesis-specific additions into a clearer package/folder.
