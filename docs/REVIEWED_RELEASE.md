# Reviewed Colab Release — 2026-10-19

**Historical software-repair record.** This document describes the earlier failed notebook and the repaired release before real training. Later on 2026-10-19, the owner returned the completed 15-run Colab experiment. Its current results and verification are in [FINAL_REPORT.md](FINAL_REPORT.md) and [RESULTS_REVIEW.md](RESULTS_REVIEW.md). The dated tests and limitations below describe that earlier release checkpoint and are retained for provenance; they are not a claim that research training is still pending.

## Scope and evidence

The review covers the returned `04_colab_improved_training.ipynb`, all nine workflow cells and their source builder, and the Python data, filters, configuration, training, checkpoints, evaluation, plotting, export, CLI, and test modules. The original uploaded notebook remains unchanged. Its SHA-256 is `fa2f54e05eccc2769a40d48f0551791b9c8528000bf2a766884b853f9205a255`.

The returned outputs record Python 3.13.15, Tesla T4, successful installation of Torch 2.13.0+cu126 / Torchvision 0.28.0+cu126, `cuda_available=True`, and 104 passing software tests. Execution stopped at the first official Shenzhen image with `Expected uint8/uint16 grayscale image`. No preflight, research training, external model evaluation, or research export was reached in that notebook.

The source format was checked directly: `CHNCXR_0001_0.png` decodes as a 2919 × 3000 × 3 uint8 array with exactly equal channels. Rejecting every three-channel image was a software error. Subsequent actual-image checks also exposed floating-point interpolation overshoot that small synthetic fixtures had missed.

## Confirmed repairs

| Finding | Repair and regression coverage |
|---|---|
| Pip can be absent in an existing isolated environment; Colab can lack `ensurepip`. | Explicit pipless venv creation, target-pip probe, offline ensurepip attempt, then host-pip `--python` fallback that manages only the target environment. Real offline pipless-environment recovery is tested. |
| Official replicated-grayscale RGB images were rejected. | Collapse only exactly equal RGB channels while preserving integer depth/intensities. Genuine color and alpha remain rejected. uint8/uint16, pixel-duplicate, and acquisition tests cover this. |
| Float32 area resize produced values such as 1.000000119, rejected by Unsharp. | Clip resized normalized pixels to [0,1] before filtering. All five methods have a regression test using a reproducing plateau fixture. |
| An already extracted source folder silently overrode a revised ZIP. | Versioned extraction directories and expected SHA-256 hashes for code, configurations, requirements, scripts, and tests. Incorrect packages fail before data operations. |
| Interrupted commands left a log name that blocked retry. | Unique log filenames and final status recording even on interruption. Regression interrupts then retries the same command while preserving the first log. |
| Empty dataset directories could bypass acquisition. | Recognize empty cohorts and interrupted managed acquisitions; reuse verified receipts and retain manual complete datasets for strict preparation. |
| Old software-validation evidence was retained as if it validated revised code. | Immutable release-specific records include source hashes. Export chooses a successful matching record and preserves the historical one. |
| External inference did not explicitly reapply the training backend policy. | Apply deterministic seeding/backend settings before each selected model evaluation. |
| Interrupted ZIP writing could leave a partial final filename. | Write and verify a temporary ZIP, then publish immutably. Simulated interruption/retry is tested. |
| Export checks did not reconcile per-run predictions or frozen case identities. | Reconcile per-run and aggregate predictions with the held-out manifest's cases, labels, and groups, in addition to SHA-256 and metric recomputation. |

Test scratch files now live on local runtime storage rather than Drive; logs and release-specific validation evidence remain persistent. This reduces avoidable Drive I/O without changing the experiment.

## Scientific protocol

The approved five methods, three seeds, ten epochs, fixed Shenzhen split, external Montgomery cohort, architecture, pretrained weights, optimizer, threshold, model-selection rule, and primary contrast are unchanged. Equal-channel collapse restores the intended native grayscale reading. Clipping only corrects floating-point interpolation overshoot. Neither repair establishes an accuracy improvement.

The historical notebooks and reports, earlier delivery archives, and verified checkpoints/results are preserved. This source revision must not be spliced into a partially trained earlier experiment: source-contract checks intentionally reject that operation.

## Executed validation

- Full automated suite: **138 passed** in 62.95 seconds on Windows / Python 3.12.14 / Torch 2.13.0+cpu. No skips or failures in that run.
- Full-project Ruff check: passed, including the generated notebook.
- Four official samples, one normal and one TB-labeled from each cohort: **20/20** preprocessing checks passed across all five methods at 384 × 384. Native pixels and depth were preserved at decoding.
- Random-weight EfficientNet-V2-S on two Shenzhen images: **5/5** forward checks passed at batch shape 2 × 3 × 384 × 384, with finite outputs. No optimizer step, pretrained download, performance metric, or Montgomery model inference was performed.
- The existing automated CPU backward and exact epoch-resume tests are included in the 138 passing tests.
- Full notebook: all **9 code cells passed twice**, including preparation, five synthetic training arms, frozen evaluation, export, and independent verification. The repeat run preserved all **10 epoch checkpoints and the existing results ZIP**, checked by SHA-256. Dependency installation was skipped in the already prepared local environment.
- Synthetic results package: **67 files and 5 runs** verified against checksums, frozen case identities, and recalculated metrics.
- The returned user notebook still matches its original SHA-256. The synthetic comparison and learning-curve images were visually inspected; labels are readable and clearly state that they are software checks, not research results.

Machine-readable official-image checks contain filenames, source URLs, hashes, shapes, and numerical checks, but no images. Test logs and notebook status records are bundled under `validation/reviewed_release/`. Synthetic results are not research results.

## Limits and handoff at the time of this software review

At this software-review checkpoint, only four official images had been downloaded for compatibility checks, not the complete cohorts. Full acquisition, duplicate validation across all 800 cases, Colab/Drive behavior, GPU memory feasibility, and the 15-run research suite still needed the user's execution. The failed notebook then available proved CUDA import in that earlier session, not successful training of the revised release.

The handoff instructed the owner to open the new notebook and upload the matching source ZIP, then run setup, software checks, acquisition/preparation, and preflight before training. No manual pip recovery cell was needed. Existing Drive data/logs and verified acquisition receipts were preserved, and Drive quota had to be checked against the checkpoint estimate.

The requested return was `results_for_review.zip`, the executed notebook, interruption/error logs if any, and checkpoint paths/checksums, with large weights kept separate. The owner has since supplied the results archive and executed notebook, enabling Phase 4. This historical software review involved no public repository operations or CV/LinkedIn claims.
