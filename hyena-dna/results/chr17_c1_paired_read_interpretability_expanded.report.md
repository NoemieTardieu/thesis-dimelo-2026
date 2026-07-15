# Paired-read interpretability analysis

- Eligible read pairs: 15
- Unique reads represented: 6
- Loci represented: 1
- Selected examples: 1

- P(Reg|D,M) predicted 6mA distances available for 0/15 pairs
- Pairwise substitution-distance audit flags > 0.1: 11/15 pairs

## Selected Cases

- D_lowDNA_low6mA: 1

## Correlations

- DNA distance vs observed 6mA MAE: Pearson=0.060970952203540874, Spearman=0.01608579730793425, n=15
- Observed 5mC MAE vs observed 6mA MAE: Pearson=0.6559751814326971, Spearman=0.5964285714285712, n=15

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

- plot1_png: `hyena-dna/results/chr17_c1_paired_read_interpretability_expanded.plot1_dna_vs_observed6ma.png`
- plot1_svg: `hyena-dna/results/chr17_c1_paired_read_interpretability_expanded.plot1_dna_vs_observed6ma.svg`
- plot3_png: `hyena-dna/results/chr17_c1_paired_read_interpretability_expanded.plot3_dna_vs_6ma_colored_5mc.png`
- plot3_svg: `hyena-dna/results/chr17_c1_paired_read_interpretability_expanded.plot3_dna_vs_6ma_colored_5mc.svg`
- plot4_png: `hyena-dna/results/chr17_c1_paired_read_interpretability_expanded.plot4_observed5mc_vs_observed6ma.png`
- plot4_svg: `hyena-dna/results/chr17_c1_paired_read_interpretability_expanded.plot4_observed5mc_vs_observed6ma.svg`
- pair_1: `hyena-dna/results/chr17_c1_paired_read_interpretability_expanded.pair_D_lowDNA_low6mA_1.png`
