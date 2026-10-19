# Colab Execution and Recovery Guide

The owner completed the real 15-run study and returned its archive and executed notebook on 2026-10-19. **No rerun is required to review those results.** This guide remains the reproduction/recovery reference for the unchanged delivered workflow. Current findings and review limits are in [FINAL_REPORT.md](FINAL_REPORT.md) and [RESULTS_REVIEW.md](RESULTS_REVIEW.md). Keep the completed experiment frozen; reproduce into a separate destination when creating a genuinely new run.

## 1. Open the delivered notebook and source package

Upload `04_colab_improved_training.ipynb` to Google Colab and select a GPU runtime with Python 3.12 (recommended) or 3.13. The pinned environment requires Python 3.12 or newer and does not support 3.11. Use the accompanying `chest-xray-enhancement-tb.zip` in the notebook's package-upload cell. This workflow does not require a public repository or a GitHub token.

Download the [clean notebook](../04_colab_improved_training.ipynb), [Colab source ZIP](../releases/colab-v2/chest-xray-enhancement-tb.zip), and [checksums](../releases/colab-v2/SHA256SUMS.txt) from this repository. Use the dedicated source ZIP, not GitHub's automatic repository ZIP: the unchanged notebook expects a `chest-xray-enhancement-tb/` archive root and source-checksum inventory. The source bundle is for execution; the full reviewed result figures remain in the repository's `reports/` directory.

For the reviewed release, open the new notebook and upload its matching new source ZIP. Do not continue editing the previous notebook or copy its temporary recovery cells. The new notebook checks SHA-256 hashes of the runtime code, configurations, dependency files, tests, and scripts. Each release extracts into its own `/content/cxr-source-<package-id>/chest-xray-enhancement-tb` directory, so an old extraction cannot silently replace the reviewed code. Browser-added suffixes on the uploaded ZIP filename are accepted; its contents must match.

Run the nine code cells in order. The pip bootstrap is automatic. Tests use local temporary storage and save their logs to Drive. Read the resource estimate before the training cell. The last cell downloads `results_for_review.zip`. Keep the same Drive work directory to resume downloaded images; do not delete its previous logs or checkpoints. An experiment that already trained under different source code must remain separate and will intentionally fail the resume contract.

Colab maintains its own dependency pins, including host pandas and ipykernel versions; see its [official environment configuration](https://raw.githubusercontent.com/googlecolab/colabtools/main/setup.py). To avoid conflicts, the notebook creates an isolated virtual environment at `/content/cxr-enhance-venv` and installs research dependencies there. All package commands, tests, and runtime checks use that environment's interpreter, shown as `PIPELINE_PYTHON`. The notebook kernel retains its Colab packages and imports only standard-library, Colab, and IPython display utilities. Installation includes `pip check` inside the isolated environment.

A new Colab runtime recreates the environment under `/content`; Drive data and checkpoints are retained. Run the setup cells before resuming. Keep the pinned Torch/Torchvision pair together; changing just one can break binary operators. If the pinned CUDA build cannot use the runtime GPU, preserve the diagnostic output and resolve the environment before a real run. A local install uses `WORK_ROOT/.venv`; local verification with `CXR_SKIP_INSTALL=1` uses the already prepared current interpreter without invoking pip.

Setup checks pip inside both new and existing environments. New environments are explicitly created with `--without-pip`, then bootstrapped using the target interpreter's offline `ensurepip` when available. If that fails, the notebook host pip manages only the target environment with `--python TARGET install pip`, as supported by [pip's interpreter-management documentation](https://pip.pypa.io/en/stable/topics/python-option/). This fallback requires host pip 22.3 or newer and package-index access. It does not install the research dependencies into Colab's host. Every setup attempt creates a fresh log directory, preserving earlier failure logs.

The historical manual pip repair is recorded in `VALIDATION_PIP_RECOVERY.md`; it is unnecessary in the reviewed notebook. Every command has a unique log filename and records interrupted status, allowing the same cell to be retried without colliding with its previous log.

The notebook targets PyTorch 2.13.0 and Torchvision 0.28.0 from the CUDA 12.6 (`cu126`) wheel channel. An earlier failed notebook recorded successful installation and `cuda_available=True` on a Tesla T4 with Python 3.13.15 but stopped during acquisition. That historical failure and its repairs are preserved in [REVIEWED_RELEASE.md](REVIEWED_RELEASE.md). The subsequently completed study records the same Python/Torch/Torchvision versions, T4 execution, physical microbatch 8, effective batch 32, and AMP. Local software verification remains CPU-based; the returned results are evidence from the user's GPU execution.

The notebook separates fast **synthetic software checks** from the 15 real-data experiments. Synthetic images and a small diagnostic model cannot establish research performance. The true EfficientNet architecture can also be checked with random weights without downloading pretrained weights; this is a forward/backward feasibility check only. Actual research runs use the explicitly recorded ImageNet pretrained weights.

## 2. Configure persistent storage and data

Mount Google Drive and select a stable project directory for prepared metadata, runs, and checkpoints. The default is `/content/drive/MyDrive/cxr-enhancement-tb`. Keep a private dataset copy with the source terms. The notebook exposes the paths to configure; the expected private directory shape is:

```text
MyDrive/cxr-enhancement-tb/
  data/
    shenzhen/CXR_png/CHNCXR_0001_0.png
    montgomery/CXR_png/MCUCXR_0001_0.png
  prepared/
  runs/balanced/
```

The filenames above illustrate the schema, not a complete inventory. Read [DATA_CARD.md](DATA_CARD.md) before acquisition. The real-data preparation requires all 662 Shenzhen and 138 Montgomery images and their expected class totals. Do not bypass these checks with a partial dataset. A missing or duplicate image is a stop condition requiring investigation, not an invitation to silently discard records.

For official acquisition outside the notebook:

```bash
python -m cxr_enhance download --dataset shenzhen --output /content/data/shenzhen
python -m cxr_enhance download --dataset montgomery --output /content/data/montgomery
```

The downloader writes acquisition receipts and compares content rather than overwriting a changed existing image. Receipts are locally computed hashes, not publisher-supplied signatures. Local storage such as `/content` avoids Drive I/O latency during acquisition; copy a verified local acquisition to private Drive storage for persistence. Existing Drive datasets can be used directly through the documented roots.

The loader accepts native uint8/uint16 grayscale PNGs and exactly replicated three-channel grayscale PNGs, as found in the official Shenzhen sample. Collapsing equal channels preserves the integer intensity plane; genuinely different color channels and alpha channels remain rejected. Floating-point resize roundoff is clipped to the normalized range before enhancement. A failed managed download resumes from verified receipts when the data cell is rerun; empty dataset directories are also recognized as incomplete.

Preparation writes `manifest.csv` and `data_validation.json`, preserving split membership and input hashes. It refuses to replace different prepared evidence. A complete verified patient map is optional, using `dataset,sample_id,patient_id`, with sample IDs equal to filename stems and a consistent patient namespace across both cohorts. Otherwise the report explicitly records case-level splitting and unverified patient independence.

Run preflight before launching a new suite. It checks physical microbatches at or below 8, GPU memory, and checkpoint storage using the actual selected model, then writes `resolved_config.json` in the suite directory. This released search does not benchmark larger batches or establish maximum throughput. Train with that resolved configuration and keep its microbatch identical across all arms and seeds. The timing is a synthetic forward/backward estimate: real data loading, validation, preprocessing, checkpoint I/O, Drive latency, and GPU variability add cost. Check both runtime disk capacity and Drive quota; mounted filesystem free-space reporting alone does not prove sufficient Drive quota. Do not change physical batch size midway through the completed comparison or reinterpret memory allocation as compute utilization.

Every epoch checkpoint is retained. The 15-run, 10-epoch suite can therefore save 150 model/optimizer checkpoints, which can require tens of gigabytes. Large weights are excluded from the review ZIP. Do not delete earlier checkpoints merely to make a storage check pass; choose sufficient storage before the experiment.

## 3. Train, evaluate once, and export

Run the notebook from preparation through training. It uses the frozen development split, pretrained EfficientNet-V2-S, five methods, three seeds, and ten epochs per run. It saves the best validation model by ROC-AUC, with validation loss as the tie-breaker. Do not change methods, thresholds, architecture, or preprocessing using Montgomery results.

After all 15 runs finish, the evaluation stage records selected checkpoint hashes and freezes the analysis before computing external-test scores. It saves individual predictions and metrics, seed summaries, paired bootstrap intervals, and plots. The final export produces `results_for_review.zip` and verifies its checksums and reproducible metrics.

External inference applies the same deterministic backend policy as training. Export first writes and verifies a temporary archive, then publishes the final filename. The verifier compares per-run predictions with the aggregate and with held-out case IDs, labels, and groups in the split manifest. The exported software-test evidence is selected by the experiment's source hash; an older retained test log is not reported as validation of newer code.

Evaluation uses probability threshold 0.5. Average precision is reported by its correct name; it is not labeled trapezoidal PR-AUC. The main comparison is corrected unsharp minus identity in ROC-AUC. Exploratory comparisons are retained even if they are worse, and their intervals are not multiplicity-adjusted.

## Resuming after an interruption

Use one Colab runtime per dataset and experiment destination. Local process locks prevent concurrent writers on one runtime; they do not coordinate separate Colab VMs sharing a Drive folder.

- Reopen the same notebook/code version, restore the pinned environment, and remount the same Drive directory.
- Reuse the same prepared manifest and suite path. Do not create a new split or rerun preflight into a different configuration for an existing experiment.
- Run training with `--resume`. Completed runs are reused; unfinished runs resume at their latest completed epoch. A partially completed epoch may run again.
- Keep all checkpoint files and metadata together. A config/data/code/environment mismatch is deliberately rejected. Preserve the error and investigate rather than editing metadata to force acceptance.
- If evaluation was interrupted, rerun evaluation against the same frozen selected models. This is technical recovery, not a new opportunity for test-guided tuning.
- If a results ZIP already exists, verify it. To retain an additional export, choose a new filename; existing packages are not overwritten.

Checkpoint loading uses Torch serialization with trusted local state. Load only checkpoints produced by this workflow or by a source you trust; never load an unknown `.pt` file merely to inspect it.

## Review evidence and preservation

The user has returned the real results archive and executed notebook. For a new run or any additional provenance check, preserve the following privately:

1. **`results_for_review.zip`** exactly as exported, including `SHA256SUMS.txt`.
2. **The executed notebook**, saved with its outputs, installation checks, hardware/preflight report, and training/evaluation logs.
3. **Any error or interruption logs**, including changes of Colab GPU or environment and recovery attempts. If execution stopped before export, return these logs and the saved run metadata.
4. **Checkpoint locations and hashes**, with access to the selected best checkpoints if requested. Keep large checkpoints separate from the ZIP and do not publish them automatically.

The review package contains run-level metrics, `experiment_config.json`, `environment.json`, `training_history.csv`, `model_comparison.csv`, predictions with labels and probabilities, split evidence, checkpoint metadata, `evaluation_report.md`, and relevant plots. Learning curves and confusion matrices are run-specific; aggregate files preserve the method and seed identity rather than silently selecting a favorable run.

Do not replace prediction files with screenshots of scores. Independent review needs numeric probabilities, labels, and case identifiers to recompute results. Keep case-level data and original logs private unless separately approved for release. The repository documentation now reports the completed real experiment, with the exact verification scope in [RESULTS_REVIEW.md](RESULTS_REVIEW.md). Original images and checkpoint binaries remain outside the returned ZIP and are needed only for deeper content or inference-replay checks; their absence must remain disclosed.
