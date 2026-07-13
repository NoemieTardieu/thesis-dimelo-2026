# Population-Level Track Benchmark Summary

This is an evaluation-only comparison. External bigWig and AlphaGenome tracks were not used as HyenaDNA training or pretraining targets.

Pearson and Spearman correlations use raw binned values. MAE/RMSE use globally fitted robust [0,1] normalized values.

| Comparison | role | n bins | Pearson | Spearman | normalized MAE | normalized RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T-A | primary_model_to_target | 30000 | 0.9499 | 0.6489 | 0.0196 | 0.0486 |
| T-H | secondary_cross_assay | 20377 | 0.5630 | 0.2530 | 0.1352 | 0.1862 |
| T-D | secondary_cross_assay | 20376 | 0.5336 | 0.3214 | 0.1769 | 0.2204 |
| A-H | secondary_cross_assay | 20377 | 0.5732 | 0.2569 | 0.1381 | 0.1862 |
| A-D | secondary_cross_assay | 20376 | 0.5415 | 0.3606 | 0.1807 | 0.2223 |
| H-D | primary_model_to_target | 20376 | 0.6379 | 0.5255 | 0.1058 | 0.1522 |

Missing bigWig values and bins below DiMeLo coverage thresholds were excluded; missing observations were not set to zero.
