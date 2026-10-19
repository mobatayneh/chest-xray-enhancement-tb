# Original Methodology and Experimental History

This reconstruction describes the supplied artifacts. It does not reinterpret literature results as project results and does not claim to reproduce unavailable historical checkpoints.

## Materials and objective

The four original notebooks are `try_1.ipynb`, `try_2_ipynb.ipynb`, `try_3.ipynb`, and `try_4.ipynb`. Their 17 nonempty code cells are largely self-contained experimental scripts. The implementation report identifies the objective as normal versus TB classification on Shenzhen. The comparative report, its byte-identical PDF copy, `summary.docx`, and the PPTX summarize three external papers. They are context and attribution sources, not additional project experiments.

## Data and preprocessing

The notebooks call KaggleHub for `raddar/tuberculosis-chest-xrays-shenzhen`. Recorded output states 662 PNG images: 336 TB and 326 normal. Labels are parsed from the final filename token. Image/label pairs are shuffled by Python's random generator without a recorded seed, then split at `int(0.6 * n)`: 397 training and 265 validation images. No saved patient mapping, stratification, split membership, duplicate audit, or independent test cohort is present.

OpenCV reads grayscale with 8-bit conversion, resizes to 640x480, applies enhancement, then resizes to 224x224. Grayscale is expanded to three channels and ImageNet mean/std normalization is used. Augmentation probabilistically selects gamma adjustment, noise, Gaussian blur, contrast, or rotation; parameters differ between experiments. No augmentation-independent ablation establishes a causal benefit. The original display pipeline in `try_4.ipynb` uses a different order of enhancement/resizing than the model input pipeline.

Unsharp masking uses a 5x5 Gaussian blur, sigma 1, usually amount 1.5, and a nominal threshold 10. The thresholded residual is unused by the returned `addWeighted` expression. Gaussian and ideal high-frequency emphasis variants appear in `try_2_ipynb.ipynb` cells 4 and 5. Their fixed cutoff is tied to the original 640x480 image grid; transferring a raw cutoff to another resolution is not a parameter-equivalent change.

## Training and metrics

Torchvision ImageNet-pretrained EfficientNet-V2-S or V2-M is adapted to a single output logit and trained with BCEWithLogitsLoss. The common SGD setting is learning rate 0.01, momentum 0.9, weight decay 0.0005; AdamW is also tried. CosineAnnealingLR and ReduceLROnPlateau are both used. Runs last 10 or 15 epochs, with batch size 32. The highest validation accuracy writes `best_model.pth`; later evaluation normally reloads that same model on the same validation loader. Only model weights are saved.

Accuracy and the classification report use a 0.5 sigmoid threshold; ROC-AUC uses probabilities. Confusion matrices are ordered normal/TB. Training loss is an unweighted average of batch means. No saved per-case probability file permits independent AUC recomputation. The machine-readable evidence records epoch metrics as printed, not full-precision recomputations.

## Notebook map

| Artifact | Nonempty cells | Relevant history |
|---|---:|---|
| `try_1.ipynb` | 1, 3 | M with unsharp; cosine and plateau variants |
| `try_2_ipynb.ipynb` | 2-6 | M/UM/plateau; S/UM/cosine; M/Gaussian HEF/cosine; S/Ideal HEF/cosine; another S/UM run |
| `try_3.ipynb` | 2-10 | S/M, SGD/AdamW, 10/15-epoch variations; cell 3 runs twice; cell 2 supplies the report's S epoch-10 values |
| `try_4.ipynb` | 1 | S/UM/cosine, 10 epochs, additional visualizations; best validation accuracy 0.8717 and AUC 0.9302 |

Run ordering within notebook cells is observable; wall-clock chronology between notebooks and complete experimentation history are not recoverable from file naming. The highest visible score is not evidence of an unbiased model improvement.

## Representative historical settings

`configs/legacy_s.json` follows the S, SGD, cosine, 224 geometry, 10-epoch family. `configs/legacy_m.json` follows the M, SGD, plateau, 224 geometry, 15-epoch family. Both request accuracy-based selection and legacy unsharp behavior. They run against a newly prepared manifest; the original 60/40 memberships are unavailable. They use the rebuilt deterministic rotation-only augmentation instead of reconstructing every historical gamma/noise/blur variation. Thus they are **archival recipe approximations, not exact reproductions**, and are not part of the main 15-run comparison. A complete historical rerun requires the original notebooks plus the original dependencies, seeds, inputs, and split memberships, several of which are missing.

| Adapter | Architecture | Epochs | Optimizer / scheduler | Selection |
|---|---|---:|---|---|
| `legacy_s.json` | EfficientNet-V2-S | 10 | SGD / cosine | Maximum validation accuracy |
| `legacy_m.json` | EfficientNet-V2-M | 15 | SGD / plateau on validation accuracy | Maximum validation accuracy |

Both adapters use ImageNet initialization, learning rate 0.01, momentum 0.9, weight decay 0.0005, nominal effective batch 32, and fixed seed 42. They preserve the historical 640x480 enhancement followed by 224x224 resizing. Modern checkpoint safety, sample-weighted losses, AMP, and microbatch accumulation remain enabled, so these are diagnostic adapters rather than a way to authenticate the historical printed metrics. Run an adapter in its own suite directory and do not mix its outputs into the primary balanced comparison.

Full source hashes and extracted results are in `provenance/`; specific corrections and confidence labels are in `PROJECT_AUDIT.md`.
