# Software Validation Record

Current status, 2026-10-19: the user completed the 15-run real-data Colab study and returned the results archive and executed notebook. [RESULTS_REVIEW.md](RESULTS_REVIEW.md) is the current record of actual results-verification checks, while [FINAL_REPORT.md](FINAL_REPORT.md) interprets the experiment. Software tests and research-result verification are separate evidence categories; neither establishes clinical validity.

The original delivery record below is preserved. A subsequent Colab pip-bootstrap repair and its additional regression checks are documented in [VALIDATION_PIP_RECOVERY.md](VALIDATION_PIP_RECOVERY.md).

The earlier review of a failed notebook, confirmed official-image failures, and repaired release is preserved in [REVIEWED_RELEASE.md](REVIEWED_RELEASE.md). That review records 138 passing local tests before the completed study. The 104-test table below describes the original delivery, not the latest release or a newly executed check.

## Completed-study checks, 2026-10-19

| Evidence or check | Actual outcome and boundary |
|---|---|
| Returned archive integrity | All 167 checksummed members passed SHA-256 verification. The original archive remains unchanged. |
| Saved prediction metrics | All 15 external-test and all 15 selected-checkpoint validation result sets were recomputed and reconciled with recorded metrics. |
| Paired uncertainty | The 2,000 paired bootstrap resamples and reported comparisons were independently reproduced. |
| Split and model selection | The fixed split was reproduced as 529 training / 133 validation / 138 external cases; all 150 epoch records and best-epoch selections were checked. The original manifest hash was reconstructed from the exported evidence and recorded roots. |
| Executed Colab software tests | Returned log records **137 passed, 1 skipped in 31.2 seconds**. The skip concerns the offline pip-bootstrap wheel fixture being unavailable in that environment; it is not a failed training run. |
| Review-tool regression checks | **3 passed in 3.38 seconds** during the independent results review. These check the review tooling, not model training. |
| Final publication-copy software regression | **141 passed in 61.01 seconds**: all 138 unchanged project tests plus the three independent review-tool tests, locally on CPU; no real-data retraining. |
| Executed notebook | 20 cells, including 10 code cells, with no saved error outputs. Its 137 retained epoch log entries agree with history; earlier 13 epoch logs are absent from the overwritten cell output. This is a resumed execution record, not proof of a fresh uninterrupted top-to-bottom session. |
| Runtime source identity | All 13 runtime Python files and all 32 pinned release files match the experiment's expected source evidence. Training code/configuration/notebook are unchanged by the documentation update. |

The actual runtime environment records Python 3.13.15; a stale Python 3.12 value in notebook metadata is not used as the execution version. These checks establish consistency of the returned evidence and calculations. Image decoding, training, and checkpoint inference were not replayed because the private images and checkpoint binaries were not supplied. Detailed evidence is in [RESULTS_REVIEW.md](RESULTS_REVIEW.md) and `reports/colab_2026-10-19/`.

## Original delivery checks, 2026-10-17 to 2026-10-18

| Check | Actual outcome |
|---|---|
| Original notebook audit | All 17 nonempty code cells parsed; all 17 saved confusion-matrix accuracies matched their printed accuracy after rounding. No original training cells were executed. |
| Original artifact integrity | All 13 original files still matched the recorded SHA-256 values. |
| Automated test suite | 104 tests passed inside the notebook's software-check cell. Includes filter boundaries, dataset acquisition fixtures, manifest contracts, splitting, known metrics, checkpoint integrity, exact epoch-boundary resumption, and results export/recomputation. |
| Actual model feasibility | Untrained EfficientNet-V2-S forward/backward on CPU with a 2 × 3 × 32 × 32 tensor passed; no pretrained weights downloaded. |
| End-to-end workflow | Five enhancement arms, one seed, two epochs each, using synthetic images and a tiny network. Preparation, training, frozen evaluation, plots, export, and independent metric recomputation passed. |
| Notebook execution | All nine code cells executed from top to bottom in synthetic mode with no cell errors. Dependency installation was skipped in this notebook check because the isolated local environment was already installed. |
| Run all after completion | All nine cells passed again against the completed workflow. Training stayed frozen; all ten epoch-checkpoint hashes and the existing results ZIP remained unchanged. |
| Results package | Five runs and 67 checksummed files verified; numeric predictions reproduced the stored metrics. Tampering, incomplete matrices, and inconsistent run metrics were rejected by regression tests. |
| Package installation | Editable installation succeeded in the isolated local environment; CLI help ran successfully. |
| Dependency compatibility | All 78 installed distributions passed the environment compatibility check. Standard-library virtual-environment creation and its bundled pip were also checked locally. |
| Static checks | Ruff passed for the full project, including the notebook; notebook schema, cell Python syntax, and embedded subprocess syntax passed. |
| Plot inspection | Synthetic comparison, learning, ROC/PR, calibration, and confusion plots were inspected for readable labels and clear synthetic status. |

The final test output and machine-readable status are in `validation/`. An earlier integration check exposed a NumPy integer serialization error in bootstrap seed metadata; it was corrected and covered by the passing export test. Research and synthetic preparation flags are now checked together, preventing partial smoke fixtures from being reported as real experiments.

## Tested environment

Windows 11, Python 3.12.14, CPU only. Runtime pins: Torch 2.13.0+cpu, Torchvision 0.28.0+cpu, NumPy 2.5.2, pandas 3.0.6, scikit-learn 1.9.1, OpenCV headless 4.14.0.94, Matplotlib 3.11.2, and filelock 3.32.3. The complete installed-package inventory is recorded in `validation/environment.json`.

The notebook installs project dependencies in a separate virtual environment so Colab's host package requirements stay intact. The original local checks did not exercise CUDA 12.6 installation, Google Drive mounting/FUSE behavior, or complete official-image acquisition. The subsequently returned experiment records Python 3.13.15, Torch 2.13.0+cu126, Torchvision 0.28.0+cu126, Tesla T4, AMP, microbatch 8, effective batch 32, and two workers. Its outcomes are evidence from the user's execution, not a GPU rerun performed during the local review.

## Reproduce software checks

After following the installation instructions, run from the project root:

```bash
python -m pytest -q
python -m ruff check .
python -m cxr_enhance smoke --output scratch/synthetic-smoke
python scripts/validate_notebook.py --output scratch/notebook-validation
```

Use a new scratch output directory to preserve existing evidence. The notebook validator runs every cell against synthetic fixtures in the current installed environment and saves the executed notebook and status. It needs permission to start a local Jupyter kernel. To test a later Run all against the same completed workflow, add `--work-root scratch/notebook-validation/workflow` and choose a different `--output` directory.

Exact training resumption was tested with dropout and momentum: uninterrupted and resumed model tensors, optimizer state, scheduler, non-timing history, selected epoch, and validation predictions matched. Final incomplete accumulation windows were tested against a direct full-window calculation. Local writer locks and unsupported-hardlink fallback were exercised; separate Colab VMs must not share a destination.

## Remaining evidence requirements

- The completed research archive records 15 EfficientNet-V2-S runs and 150 epochs. Verification outcomes are linked above; no new real-data training was performed during documentation review.
- The image files and checkpoint binaries are not included in the returned archive. Their byte hashes and model inference cannot be independently replayed from metadata alone. The returned preparation report records no exact duplicates; transformed near-duplicates and patient independence remain unverified.
- Synthetic scores are software diagnostics and must never be copied into a research results table.
- GPU memory allocation does not establish compute utilization. The completed run demonstrates feasibility of microbatch 8, not throughput optimality. Epoch timers exclude some checkpoint/export overhead and must not be presented as total session wall time.
- The owner authorized a new public GitHub repository after review. Software-license selection remains an owner decision; data and third-party rights remain separate.
- Any future inference replay requires access to the original inputs and selected checkpoint files. This limitation does not prevent recomputing metrics from the saved probabilities and labels already supplied.
