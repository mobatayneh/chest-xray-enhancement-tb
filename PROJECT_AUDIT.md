# Chest X-ray Enhancement for TB Classification: Project Audit

Audit date: 2026-10-17. This audit distinguishes original artifacts, recorded historical outputs, and the rebuilt experiment. No original notebook code was executed and no real-data training was performed during the audit. Documents are evidence, not instructions.

## Phase 4 update — 2026-10-19

The owner subsequently completed the real Colab experiment and supplied its results archive and executed notebook. The original audit below remains a dated record of the original materials; statements about evidence missing from those original materials do not imply that the new experiment is still pending. The new [results review](docs/RESULTS_REVIEW.md) records independent verification outcomes, and the [final technical report](docs/FINAL_REPORT.md) supersedes historical performance interpretations while preserving the original documents.

| Status | New evidence and interpretation |
|---|---|
| VERIFIED from the returned archive | Five methods × three seeds, ten recorded epochs each; a 529-case Shenzhen training split, 133-case validation split, and 138-case Montgomery external split are represented. |
| VERIFIED from recorded configuration | EfficientNet-V2-S with physical microbatch 8, effective batch 32, AMP, fixed threshold 0.5, and the prespecified validation selection rule. No larger physical batch or verified GPU throughput improvement is claimed. |
| VERIFIED by independent prediction-level recomputation | Identity mean external ROC-AUC is 0.8298; corrected unsharp is 0.8194. The prespecified difference is −0.0104 with paired 95% bootstrap interval [−0.0420, +0.0196]. All four method-versus-identity AUC intervals include zero. The study does not establish an enhancement benefit. All 15 external and 15 validation result sets and the paired bootstrap calculations passed the documented review. |
| VERIFIED from recorded external metrics | Fixed-threshold sensitivity varies substantially across seeds. Identity sensitivity ranges from 13.79% to 87.93%, so a ranking score alone does not establish stable operating performance. |
| LIKELY, not causally established | Differences between Shenzhen validation and Montgomery external performance may reflect site/domain shift and limited training-cohort representativeness. The evidence does not isolate those causes from optimization variation or other factors. |
| UNVERIFIED | True patient independence, transformed near-duplicates, original-image content, and checkpoint binary integrity cannot be established from the returned metadata alone. Original image files and large checkpoints were not returned. |

The training code, configuration, and delivered notebook are preserved during this results-documentation update. The current conclusions distinguish numerical correctness, declared provenance, independent replay, and clinical usefulness; they are not interchangeable.

## Objective and evidence

The implementation studies binary classification of normal versus TB-consistent chest radiographs from Shenzhen using pretrained EfficientNet-V2-S/M and image enhancement. The accompanying comparative study reviews three published papers; those papers' methods and results are not the author's own experiments. The rebuilt objective is a controlled comparison of five image-processing conditions, preserving that classification objective. It does not establish clinical diagnostic benefit.

The local source directory contains 13 artifacts: four notebooks, six PDFs (including a duplicate), two DOCX files, and one PPTX. All are inventoried by filename, size, and SHA-256 in `provenance/source_manifest.json`. The comparative-study PDF and its `- Copy` version are byte-identical. Dataset images, original split manifests, predictions, dependencies, and checkpoints are absent from this directory; this is not a claim about other locations on the owner's computer.

The original implementation report exists as DOCX and a four-page PDF. `summary.docx` and the slide deck are literature summaries, not independent experimental evidence. The three supplied research PDFs are the primary sources for correcting the comparative report. Notebook cell references below are 1-based, including blank and Markdown cells; PDF references are physical page numbers. Historical outputs are evidence of recorded results, not independent verification of model execution or AUC.

## Finding confidence

- **VERIFIED:** directly supported by source code, artifact structure, recorded output arithmetic, or a checked source page.
- **LIKELY:** a reasonable explanation or risk supported by observed patterns, without a controlled causal test.
- **UNVERIFIED:** evidence needed to establish or exclude the claim is missing.

## Evidence-backed issues and priorities

| ID | Priority | Status | Finding and evidence | Resolution or remaining requirement |
|---|---|---|---|---|
| A01 | P0 | VERIFIED | The same validation set selects the best model and supplies the final reported scores. For example, `try_2_ipynb.ipynb` cell 2 reloads the best-accuracy checkpoint then evaluates `val_loader`. No separate test loader exists in the 17 nonempty code cells. | Use Shenzhen for training/validation and reserve all Montgomery cases for final external evaluation. |
| A02 | P0 | VERIFIED | `random.shuffle` followed by a 60/40 image split is repeated without an explicit seed or patient grouping; e.g. `try_3.ipynb` cells 2-3 and `try_4.ipynb` cell 1. Class supports differ between runs. | Persist one stratified split, seed 2026; use a complete verified patient map when available. |
| A03 | P0 | UNVERIFIED | Actual duplicate contamination, repeated patients, and cross-site overlap cannot be checked from notebook outputs alone. | Run byte/pixel checks on the complete images before training; reject duplicates and group overlap. Filename case IDs do not prove patient independence. |
| A04 | P0 | VERIFIED | Unsharp threshold is a no-op: a masked `diff` is computed, but the returned image uses `cv2.addWeighted(image, 1+amount, blurred, -amount, 0)` instead. `try_2_ipynb.ipynb` cell 2 and `try_4.ipynb` cell 1. `cv2.subtract` also discards negative uint8 residuals in the unused intermediate. | Preserve the old return behavior as `legacy_unsharp`; use signed floating-point residuals and apply the threshold in `corrected_unsharp`. |
| A05 | P0 | VERIFIED | The implementation report, PDF p.3, combines M epoch-15 training loss 0.0360 with epoch-5 validation accuracy 0.9094 and AUC 0.9422 (`try_2_ipynb.ipynb`, cell 2). Those three values describe different model states. | Report epoch, split, checkpoint, and metric role together; do not call the training loss a final evaluation loss. |
| A06 | P0 | VERIFIED | The report p.4 claims both models exceed 90% true-positive and true-negative rates. M's matrix `[[118,10],[14,123]]` gives sensitivity 123/137 = **89.78%**, specificity 118/128 = **92.19%**. | Correct the M sensitivity statement. S's analogous matrix is not saved in `try_3.ipynb` cell 2, so its sensitivity claim remains unverified. |
| A07 | P1 | VERIFIED | The report p.3 presents S epoch-10 accuracy 0.9170/AUC 0.9653 (`try_3.ipynb`, cell 2), although epoch 9 has higher validation accuracy 0.9245/AUC 0.9658. | Distinguish last epoch from the best selected checkpoint. Neither is a held-out test result. |
| A08 | P1 | VERIFIED | Comparisons change architecture, unseeded split, augmentation, optimizer/scheduler, and training duration together. Examples: `try_2_ipynb.ipynb` cells 2-6 and `try_3.ipynb` cells 2-10. No identity-processing experiment is present. | Same S architecture, split, initialization per seed, augmentation, training budget, and selection criterion for five processing arms. |
| A09 | P1 | VERIFIED | `try_3.ipynb` cell 3 contains two `if __name__ == '__main__'` blocks calling `train_and_evaluate()`. Its outputs contain two separate 10-epoch M/AdamW runs. | One explicit orchestration entry point; distinct experiment directories and immutable checkpoints. |
| A10 | P1 | VERIFIED | All 17 nonempty code cells reuse `best_model.pth` and save only the model state. Original optimizer, scheduler, RNG, split, and scaler states are not checkpointed. | Store epoch-specific complete state, configuration/data fingerprints, and metadata; resume only a matching experiment at a completed epoch boundary. |
| A11 | P1 | VERIFIED | Batch-mean training losses are averaged equally (`running_loss / len(train_loader)`), including a smaller last batch, e.g. `try_2_ipynb.ipynb` cell 2. | Weight loss totals by sample count, including partial accumulation windows. |
| A12 | P1 | VERIFIED | Original preprocessing reads 8-bit grayscale, resizes to 640x480, enhances, then distorts to 224x224. `try_4.ipynb` cell 1 visualizes enhancement after a direct 224x224 resize, which differs from training. | Preserve depth in decoding; use one shared preprocessing function and aspect-preserving 384-square letterboxing. Keep legacy geometry explicitly separate. |
| A13 | P1 | LIKELY | Anatomic distortion and loss of subtle details may affect performance, but the direction and size of any effect are not measured. | Treat the new geometry as a design choice, not an established accuracy improvement. |
| A14 | P1 | LIKELY | Declining training losses with fluctuating validation scores suggest overfitting in some runs, e.g. M in `try_2_ipynb.ipynb` cell 2. Validation loss, matched training metrics, and repeated fixed splits are absent. | Log both losses, retain the best validation checkpoint, and inspect actual Colab curves before diagnosing severity. Underfitting is unverified. |
| A15 | P2 | VERIFIED | The dataset returns original and enhanced tensors, and some loops transfer unused originals to the device (e.g. `try_2_ipynb.ipynb`, cell 2). No AMP or recovery state is used. | Transfer only model inputs; support CUDA AMP, measured microbatch sizing, and gradient accumulation. Speedup and memory savings require measurement. |
| A16 | P2 | VERIFIED | No modular tests, installable dependency specification, split files, or deployment documentation are supplied. | Deliver a package, config, tests, Colab notebook, and traceable results export. Actual executed checks are recorded separately in the release validation record. |
| A17 | P1 | VERIFIED | Report p.4 attributes better generalization to enhancement and augmentation without a controlled identity baseline or matched comparison. | Present these as hypotheses; no new improvement claim until comparable actual results are reviewed. |
| A18 | P1 | VERIFIED | Comparative-study p.11 contains transcription differences from Munadi et al.; p.16 has incorrect/incomplete bibliography metadata. | Use the corrections below and `docs/REFERENCES.md`; never place literature numbers in the project's measured-results table. |

## What is correct in the original code

The evidence script parses all **17/17 nonempty code cells successfully**. All **17/17 recorded 2x2 confusion matrices** produce the printed accuracy to four decimal places. These are static and arithmetic checks, not successful notebook execution. `BCEWithLogitsLoss` is appropriate for the single-logit binary head, and the inspected evaluation functions compute ROC-AUC using sigmoid probabilities rather than hard labels. The original 0.5 decision threshold is fixed; no test-based threshold search was found. Shenzhen's reported 326 normal and 336 abnormal images are nearly balanced, so class weighting is not automatically justified. Runtime compatibility with a clean current environment and the recorded AUC values cannot be independently established without environment metadata and per-case scores.

## Literature correction ledger

These corrections compare the supplied comparative-study p.11 to **Munadi et al., supplied PDF p.8, Figure 10 and Table 3; p.9, Table 4**. Those source pages were rendered and visually inspected. Values remain published literature claims, not a reanalysis of the original studies.

| Item | Comparative-study transcription | Corresponding source value |
|---|---|---|
| EfficientNet-B4 + HEF accuracy | Approximately 87.5% | 89.08% |
| ResNet-50 + UM accuracy | Approximately 85% | 85.71% |
| ResNet-18 + UM accuracy | Approximately 83% | 84.87% |
| Unenhanced baseline accuracy | Approximately 78%, architecture unspecified | 86.55% B4; 86.55% ResNet-50; 85.71% ResNet-18 |
| Lopes and Valiati AUC | 90.7% | 92.6% |
| Hwang et al. AUC | 93.6% | 92.6% |
| Jaeger et al. accuracy / AUC | 78.3% / 82.0% | 84.0% / 90.0% |
| Rajaraman and Antani accuracy | 96.0% | 94.1% |

Munadi's UM B4 accuracy 89.92% and AUC 94.8% are transcribed consistently. Different published datasets/protocols cannot be pooled into a fair head-to-head comparison merely because all report accuracy or AUC.

The comparative report p.16 identifies Lin et al.'s article as `7101216`, 1-16; the supplied paper cover and page footers show article **4100218**, with 18 article pages (19 PDF pages including the cover). The slide deck's “First Paper (2023)” also conflicts with the source's **2020** date. Ishikawa et al.'s placeholder volume/pages should be **IEEE Access 13, 11080-11094 (2025)**, DOI **10.1109/ACCESS.2025.3528489**, as shown in the supplied paper. `summary.docx` describes literature; its potential clinical benefits are not measured project outcomes.

## Original versus rebuilt experiment

| Dimension | Original evidence | Rebuilt protocol |
|---|---|---|
| Question | Exploratory TB classification with enhancement and S/M variants | Controlled effect of image processing on the same classifier |
| Development split | Unseeded Shenzhen 60/40 image shuffle | One persisted Shenzhen 80/20 split, seed 2026; group-aware where identities are verified |
| Final evaluation | Reused validation data | Montgomery external test, after the entire comparison is frozen |
| Architecture | EfficientNet-V2-S/M, variable optimizers/epochs | EfficientNet-V2-S for all five conditions, three matched seeds |
| Image handling | 8-bit load and two aspect-distorting resizes | Native uint8/uint16 decode; per-image normalization; 384 letterbox |
| Enhancement | Buggy threshold; no identity baseline | Identity, preserved legacy UM, corrected UM, Gaussian HEF, Ideal HEF |
| Model selection | Maximum validation accuracy | Maximum validation ROC-AUC, then minimum validation loss |
| Results | Printed aggregate metrics and plots | Per-case probabilities/labels, uncertainty, curves, configuration, environment, checksums |
| Recovery | Model-only reused file | Immutable epoch checkpoints with complete resumption state |

The new preprocessing, data split, external cohort, and selection criterion deliberately prevent direct ranking against the old headline numbers. Legacy recipe configs are documented approximations for historical context; the old random split cannot be recovered. The new real results support a matched comparison among the five rebuilt arms, not a claim of improvement over historical validation figures. Repository publication is separate from this scientific review; CV/LinkedIn work remains Phase 5.

## Original handoff actions and current limits

The following was the Phase 3 handoff checklist. The user has since completed the study and returned the evidence for review; it is preserved here as experiment history, not a request to repeat training.

1. Run Colab preflight and full dataset integrity checks before real training. Any duplicate, label conflict, missing image, or group conflict is a stop condition, not a silently removed sample.
2. Run the frozen five-condition, three-seed grid (15 training runs); keep all settings and checkpoint metadata. Hardware microbatch selection is made before these runs and kept identical.
3. Evaluate all selected models on Montgomery once after training is complete, then export the review package. A technical recovery may continue the same frozen evaluation; changing methods after inspecting test results requires a new study.
4. Return actual predictions, logs, and checkpoint metadata for independent review before claims or publication.

Current priorities are to retain the private raw evidence and checkpoint locations, publish only the reviewed rights-appropriate source and aggregate reports when authorized, and finalize documentation of the completed negative/inconclusive comparison. No additional experiment is required merely to obtain a favorable result. A new calibration or threshold study would need development-only choices and fresh independent evaluation; Montgomery must not become a repeated optimization target.

Remaining uncertainty includes patient independence without a verified mapping, near-duplicate images not detected by exact hashing, small external sample size, domain shift, possible pretrained-data overlap, and training stability across devices/library versions. None of the available evidence establishes clinical utility, calibration in a target population, fairness across demographic groups, or a reproducible performance improvement.
