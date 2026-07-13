# Paired-read interpretability analysis

- Eligible read pairs: 15
- Unique reads represented: 6
- Loci represented: 1
- Selected examples: 1

- P(Reg|D,M) predicted 6mA distances available for 15/15 pairs
- Pairwise substitution-distance audit flags > 0.1: 8/15 pairs

## Selected Cases

- D_lowDNA_low6mA: 1

## Correlations

- DNA distance vs observed 6mA MAE: Pearson=0.14396274574732668, Spearman=0.38928571428571423, n=15
- Observed 5mC MAE vs observed 6mA MAE: Pearson=-0.15563865444807626, Spearman=-0.021428571428571425, n=15

## Methylation added-value analysis

This is an association analysis, not causal evidence.

```json
{
  "status": "not_enough_data",
  "n_pairs": 15,
  "n_loci": 1
}
```

## Plots

- plot1_png: `outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256.plot1_dna_vs_observed6ma.png`
- plot1_svg: `outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256.plot1_dna_vs_observed6ma.svg`
- plot2_png: `outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256.plot2_observed_vs_predicted6ma.png`
- plot2_svg: `outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256.plot2_observed_vs_predicted6ma.svg`
- plot3_png: `outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256.plot3_dna_vs_6ma_colored_5mc.png`
- plot3_svg: `outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256.plot3_dna_vs_6ma_colored_5mc.svg`
- plot4_png: `outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256.plot4_observed5mc_vs_observed6ma.png`
- plot4_svg: `outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256.plot4_observed5mc_vs_observed6ma.svg`
- pair_1: `outputs/chr16_c1_paired_read_interpretability_final_pred_mlp256.pair_D_lowDNA_low6mA_1.png`
