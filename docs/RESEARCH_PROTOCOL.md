# Frozen Research Protocol

Protocol version: 1. Approved implementation plan date: 2026-10-17. Status update, 2026-10-19: **the 15-run Colab experiment is complete and the returned evidence passed the documented Phase 4 consistency and metric checks**. This document retains the approved protocol; reporting the observed results does not amend it. Any later scientific amendment must be recorded as a new study before evaluation and approved when it changes the experiment's meaning.

The returned configuration records 10 epochs for each of five methods and three seeds, physical microbatch 8, effective batch 32, two workers, and CUDA AMP. The prepared evidence records 529 training cases (261 normal / 268 TB-consistent), 133 validation cases (65 / 68), and 138 external cases (80 / 58), grouped by case ID without a verified patient map. Review scope and actual checks are in [RESULTS_REVIEW.md](RESULTS_REVIEW.md); outcomes are in [FINAL_REPORT.md](FINAL_REPORT.md).

## Research question and comparison

Does a specified image-enhancement transform change classification of normal versus TB-consistent chest radiographs, holding training conditions fixed? Use Torchvision EfficientNet-V2-S with explicit `IMAGENET1K_V1` weights, a single-logit classifier, and BCEWithLogitsLoss. The stronger M variant is retained as a historical recipe, not added to the primary experiment, to avoid confounding preprocessing with architecture and to control the cost of 15 runs.

Five conditions are crossed with seeds 42, 43, 44: `identity`, `legacy_unsharp`, `corrected_unsharp`, `gaussian_hef`, `ideal_hef`. The primary contrast is **corrected_unsharp minus identity** in external-test ROC-AUC. All other contrasts and metrics are exploratory. The anticipated benefit of enhancement is a hypothesis; negative or inconclusive results must be retained.

## Data and transforms

Use all valid Shenzhen cases for a fixed 80/20 train/validation allocation, seed 2026, and all Montgomery cases exclusively for external evaluation. Preserve patient groups when a trustworthy complete map exists; otherwise explicitly report case-level splitting and unverified patient independence. Reject duplicates and input anomalies before starting. No fitted normalization statistics or preprocessing parameter selection use the external cohort.

Decode native integer depth, linearly normalize each image independently to [0,1], and resize the longest edge to 384 preserving aspect ratio. Apply enhancement to the resized image before centered zero padding to 384x384. Repeat grayscale into three channels and apply ImageNet normalization. Training augmentation is one rotation in [-10,10] degrees with probability 0.5; evaluation has no random augmentation. Per-image min-max normalization is fixed, not a learned or test-tuned transform.

Exactly replicated three-channel grayscale PNGs are collapsed to their identical intensity plane without a luminance conversion or bit-depth change. Genuinely unequal color channels and alpha-channel images are rejected. Floating-point interpolation overshoot is clipped to [0,1] immediately after resizing; this is a numerical range correction, not a test-tuned enhancement.

UM uses Gaussian kernel 5x5, sigma 1, amount 1.5. `legacy_unsharp` quantizes to uint8 and reproduces the historical GaussianBlur/addWeighted output, intentionally ignoring the unused threshold. `corrected_unsharp` uses signed floating-point residuals and suppresses absolute residuals <= 10/255 before adding the residual. The legacy/corrected comparison therefore also includes quantization differences; it is not a pure one-variable threshold ablation.

HEF uses gain `0.5 + 1.5 * highpass` and per-image min-max normalization. Cutoff is `(30/480) * min(height,width)` in DFT bins on the unpadded resized image; it is fixed before testing. Gaussian highpass is `1-exp(-distance_squared/(2*cutoff**2))` and takes the real inverse transform. Ideal highpass is a binary radial mask and retains the original magnitude reconstruction behavior. Frequency-grid units and boundary conventions are implemented explicitly, including odd dimensions. These are specified research transforms, not a claim to reproduce all published HEF implementations.

## Optimization, selection, and recovery

Use 10 epochs, SGD learning rate 0.01, momentum 0.9, weight decay 0.0005, and cosine learning-rate decay over the full 10 epochs to 0.000001. BCE is unweighted because the source development classes are nearly balanced. No class resampling or threshold search is included. Select by maximum validation ROC-AUC, then minimum validation loss on ties. Classification uses probability >= 0.5.

Target effective batch size is 32 with initial microbatch 8; preflight checks feasible microbatches at or below 8 and fixes the selected value across all methods/seeds before training. The completed experiment used 8; larger physical batches were not benchmarked, so this is not a claim of optimal GPU throughput. Accumulated gradients are normalized by the actual samples in the accumulation window, including a final partial window. BatchNorm still observes the microbatch, so accumulation is not mathematically identical to a true 32-image physical batch. Enable AMP on CUDA and use full precision on CPU. Enable deterministic controls, seed Python/NumPy/Torch, and keep worker/data-order and augmentation decisions reproducible. Re-seed model construction identically for all arms of each seed.

Checkpoints are distinct per epoch and include model, optimizer, scheduler, AMP scaler, RNG states, complete history, configuration, data fingerprints, and initialization provenance. Resume only a matching experiment at the latest completed epoch. Interrupted partial epochs may be replayed; bitwise equivalence is only expected within compatible deterministic environments. Do not overwrite a checkpoint to bypass a provenance mismatch. Retain the best validation checkpoint and all preserved historical evidence.

## External evaluation and statistical reporting

Finish all configured training runs and freeze the selected checkpoint identities and configuration before Montgomery evaluation. Evaluate the full configured grid as one planned analysis. Technical restart may continue that same immutable grid; do not choose an arm, epoch, threshold, or new hyperparameter based on a viewed Montgomery score.

Save per-case labels/probabilities and full-precision metrics. Report ROC-AUC, average precision (not trapezoidal PR-AUC), accuracy, specificity, sensitivity, precision, F1, classwise summaries, Brier score, confusion matrices, ROC/PR plots, calibration curves, and training/validation loss curves. Give each seed's result and arithmetic mean/sample standard deviation across seeds. Do not pool three predictions per case as three independent patients.

Use 2,000 paired bootstrap resamples with the same case/group indices across methods and seeds. Keep all seed predictions for each resampled case together; compute a metric per seed then average across seeds. Compare method-minus-identity deltas within each resample, and use percentile 95% intervals. Resample verified patient groups when available; otherwise report case-level uncertainty. A one-class ROC-AUC is undefined and must not be coerced to a number. Missing classes or invalid score arrays fail the evaluation. Three seeds assess limited optimization variation, not a population-level sample size of three times the number of cases. Exploratory intervals are not adjusted for multiplicity.

## Acceptance and handoff

Code acceptance requires filter boundary tests, data/label/split checks, known-result metric tests, checkpoint/recovery tests, and an end-to-end synthetic run that exports checksummed artifacts. A real EfficientNet forward/backward smoke check can run without pretrained downloads; it is not training evidence. Validation reports must identify actual commands and outcomes rather than infer success from code inspection.

Colab must install the documented environment, check GPU/memory/storage, acquire or locate the data, validate/freeze the manifest, train the 15 runs, evaluate the selected checkpoints, and export `results_for_review.zip`. The ZIP must distinguish synthetic diagnostics from real results. Keep large checkpoints separately. The owner returned the real results ZIP and executed notebook on 2026-10-19; the review is documented separately and does not require rerunning the experiment to update this report. Checkpoint metadata identifies the evaluated files, but independent inference replay requires the retained checkpoint binaries and images. Do not use the observed external scores to tune this completed comparison.
