# Paired-read interpretability analysis

- Eligible read pairs: 40
- Loci represented: 3
- Selected examples: 2

## Correlations

- DNA distance vs observed 6mA MAE: Pearson=0.1531368539125246, Spearman=0.2033771106941839, n=40
- Observed 5mC MAE vs observed 6mA MAE: Pearson=-0.6736007812631895, Spearman=-0.6204502814258913, n=40

## Methylation added-value analysis

This is an association analysis, not causal evidence.

```json
{
  "status": "ok",
  "n_pairs": 40,
  "n_loci": 3,
  "models": {
    "dna_only": {
      "cv_r2": -0.888539715337616,
      "cv_mae": 0.11687366102755722,
      "mean_coefficients": [
        0.1894814111604931,
        0.022377281093653407
      ],
      "predictors": [
        "intercept",
        "dna_distance"
      ]
    },
    "dna_plus_5mc": {
      "cv_r2": -0.4481948661411532,
      "cv_mae": 0.09730120981983147,
      "mean_coefficients": [
        0.2577475134101634,
        0.005467618077883664,
        -0.3641864360845273
      ],
      "predictors": [
        "intercept",
        "dna_distance",
        "observed_5mc_mae"
      ]
    },
    "delta_cv_r2_adding_5mc": 0.4403448491964628,
    "delta_cv_mae_adding_5mc": -0.019572451207725758
  }
}
```

## Plots

- plot1_png: `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot1_dna_vs_observed6ma.png`
- plot1_svg: `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot1_dna_vs_observed6ma.svg`
- plot2_png: `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot2_observed_vs_predicted6ma.png`
- plot2_svg: `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot2_observed_vs_predicted6ma.svg`
- plot3_png: `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot3_dna_vs_6ma_colored_5mc.png`
- plot3_svg: `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot3_dna_vs_6ma_colored_5mc.svg`
- plot4_png: `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot4_observed5mc_vs_observed6ma.png`
- plot4_svg: `outputs/chr16_c1_paired_read_interpretability_observed_medium.plot4_observed5mc_vs_observed6ma.svg`
- pair_1: `outputs/chr16_c1_paired_read_interpretability_observed_medium.pair_A_lowDNA_high6mA_1.png`
- pair_2: `outputs/chr16_c1_paired_read_interpretability_observed_medium.pair_B_highDNA_low6mA_2.png`
