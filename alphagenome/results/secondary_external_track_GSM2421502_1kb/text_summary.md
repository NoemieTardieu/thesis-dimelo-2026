# Population-Level Track Benchmark Summary

This is an evaluation-only comparison. External bigWig and AlphaGenome tracks were not used as HyenaDNA training or pretraining targets.

Primary metrics use pair-specific valid bins and robust [0,1] normalized values.

| Comparison | n bins | Pearson | Spearman | normalized MAE | normalized RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| T-A | 6000 | 0.9528 | 0.6410 | 0.0220 | 0.0567 |
| T-H | 4105 | 0.5759 | 0.1373 | 0.1406 | 0.1921 |
| T-D | 4105 | 0.6724 | 0.3224 | 0.2014 | 0.2361 |
| A-H | 4105 | 0.6200 | 0.2891 | 0.1378 | 0.1865 |
| A-D | 4105 | 0.6800 | 0.4268 | 0.2001 | 0.2334 |
| H-D | 4105 | 0.6658 | 0.5514 | 0.1222 | 0.1639 |

Missing bigWig values and bins below DiMeLo coverage thresholds were excluded; missing observations were not set to zero.
