# Reviewed Colab Results - 19 October 2026

These are real, completed experiments: five conditions, seeds 42/43/44, ten epochs each. External evaluation uses the same 138 Montgomery cases for every run. Values are verified against the private returned predictions; see [results review](../../docs/RESULTS_REVIEW.md) and the [final report](../../docs/FINAL_REPORT.md).

**Primary conclusion:** corrected unsharp minus identity mean ROC-AUC = -0.0104167, paired 95% interval [-0.0420286, +0.0196157]. No enhancement benefit is established. Exploratory methods' higher point estimates are not confirmed superiority.

| Artifact | Meaning |
|---|---|
| `model_comparison.csv` | Full-precision external metrics for each of the 15 runs |
| `method_summary.csv` | Metric means and sample SD over three seeds; no pooling of patients |
| `validation_comparison.csv` | Selected-epoch validation metrics, separately labeled |
| `generalization_comparison.csv` | Matched validation/external scores and selected epochs |
| `paired_comparisons.csv`, `metrics.json` | Joint paired bootstrap intervals and original aggregate metrics |
| `training_history.csv` | All 150 epoch records; seconds exclude checkpoint saving |
| `experiment_config.json`, `environment.json` | Actual frozen settings and runtime dependencies |
| `checkpoint_metadata.json` | The 15 selected checkpoint identities; no weights |
| `data_validation.json` | Aggregate split counts and recorded preparation checks |
| `colab_software_validation.json` | Recorded Colab test outcome; local paths removed for publication |
| `preflight.json` | Original synthetic hardware preflight; not full-training throughput |
| `input_provenance.json` | Private input fingerprints and timing interpretation |
| `verification.json`, `notebook_review.json` | Independent numerical and execution review records |
| `review_tools/` | Independent audit implementation and three self-tests; does not change the training package |
| `figures/` | Reviewed summary figures |
| `original_figures/` | Original aggregate and per-run plots from the returned ZIP |
| `SHA256SUMS.txt` | Hash inventory of this public evidence directory |

Original `results_for_review.zip`, executed notebook outputs, case-level predictions, labels/IDs, split membership, image hashes, original images, and checkpoint tensors stay outside the public repository. The aggregate evidence package is intentionally different from the private original archive; its own hash inventory identifies its contents.

The raw ZIP has 168 members and 167 checked SHA-256 entries. Its hash is `655b3c6be8f14eb507004d5aed7acd19e6e690692185b0c1d5ae4f2bddea44e6`. It supports independent metric recomputation by authorized reviewers. Aggregate public tables alone do not permit that per-case verification or full model replay.
