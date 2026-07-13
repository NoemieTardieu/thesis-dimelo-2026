# HyenaDNA Pillar

Active thesis direction:

- `P(M | D)` predicts CpG 5mC from sequence.
- `P(Reg | D, M)` predicts DiMeLo 6mA regulatory signal from DNA plus methylation context.

Final evaluation uses overlap aggregation by sample/read/read-position and includes cross-chromosome generalization, threshold metrics, variance analysis, and paired-read interpretability.

Large tensors and checkpoints are retained on the server and ignored by Git.
