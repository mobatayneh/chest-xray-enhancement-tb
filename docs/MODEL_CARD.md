# Research Model Card

## Model family and purpose

Fifteen EfficientNet-V2-S binary classifiers compare five preprocessing conditions across seeds 42, 43, and 44. They use ImageNet initialization, a single-logit output, and the frozen protocol in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). Output class 1 means the dataset label "TB-consistent", not a verified microbiological diagnosis or calibrated clinical probability.

Intended use is reproducible retrospective research and technical education. No deployment model is selected by the external test. These artifacts do not establish suitability for patient screening, diagnosis, triage, or treatment decisions.

## Data and evaluation

Shenzhen: 529 training and 133 validation images. Montgomery: 138 external cases, 80 normal and 58 TB-consistent. Patient-level independence, demographic fairness, prospective performance, and pretrained-data overlap are unverified. The [data card](DATA_CARD.md) records access and attribution.

External mean AUC spans 0.8194 to 0.8517. The primary corrected-unsharp comparison does not establish improvement over identity. Sensitivity is unstable at the fixed 0.5 threshold; identity seeds yield 31.03%, 13.79%, and 87.93%. See the complete [final report](FINAL_REPORT.md), including all seeds and uncertainty rather than a selected best score.

## Availability and reproducibility limits

No model tensor files are included in this repository. Selected checkpoint paths and SHA-256 identities appear in the aggregate evidence. The owner retains the large checkpoints separately. Independent reviewers checked metadata and predictions but did not rehash the weight files or rerun inference.

No inference API, approved clinical threshold, fairness claim, or clinical calibration claim is supplied. Any new research model or threshold selection requires development-only validation and newly reserved external evidence after the current test results have been observed.
