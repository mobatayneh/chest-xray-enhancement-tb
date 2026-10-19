# Data Card

## Source and intended use

Shenzhen and Montgomery are NLM-distributed, de-identified chest radiograph research datasets. The source labels distinguish normal images from abnormalities consistent with TB. “TB” in code is shorthand for the supplied positive class, not proof of microbiological diagnosis. This project is a research comparison of image enhancement and binary classification, not a clinical screening service.

| Cohort | Source population | Expected images | Normal / abnormal | Assigned role |
|---|---|---:|---:|---|
| Shenzhen | Shenzhen No. 3 Hospital, China | 662 | 326 / 336 | Fixed 80/20 training/validation split |
| Montgomery | Montgomery County TB program, Maryland, USA | 138 | 80 / 58 | External test, all cases |

Counts and cohort descriptions are from the [NLM resource page](https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html), checked 2026-10-17. They match the split evidence returned from the completed Colab experiment on 2026-10-19. The original notebooks used a Kaggle mirror; the rebuilt pipeline prefers NLM downloads or a user-provided copy with recorded hashes. The two sources cannot be assumed byte-identical without comparison.

## Observed experiment allocation

| Recorded split | Cohort | Cases | Normal | TB-consistent |
|---|---|---:|---:|---:|
| Training | Shenzhen | 529 | 261 | 268 |
| Validation | Shenzhen | 133 | 65 | 68 |
| External test | Montgomery | 138 | 80 | 58 |

The returned `data_validation.json` records stratified case splitting with seed 2026, no supplied patient map, and no exact byte or decoded-pixel duplicates. These are recorded preparation findings. Review of the saved manifest can check membership, counts, labels, and hash consistency; without the original image bytes it cannot independently repeat decoding or duplicate detection. Patient independence and transformed near-duplicates remain unverified. See [the results review](RESULTS_REVIEW.md) for the precise verification scope.

## Access and restrictions

Use the [Shenzhen source directory](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html) and [Montgomery source directory](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html). Read the supplied source terms before downloading. The [Montgomery README, p.2](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf) requests no redistribution outside the research group/organization and asks for attribution. Shenzhen supplies `NLM-ChinaCXRSet-ReadMe.docx` in its source directory; retain it with the private dataset copy.

No image, raw clinical reading, mask, annotation archive, or third-party PDF is distributed with this code. Do not infer a permissive dataset license from public download access, from a research paper's CC-BY license, or from a future code license. Cite the dataset source and publications listed in `REFERENCES.md`. Public distribution of predictions/identifiers also needs a separate release review; the review export is for private verification.

## Required local schema

Provide separate cohort roots, each containing exactly one `CXR_png` directory. The original grayscale PNG filenames encode cohort, case number, and label, such as `CHNCXR_0001_0.png` and `MCUCXR_0001_0.png`. A suffix of `0` denotes normal and `1` denotes the supplied abnormal/TB-consistent class. Images must decode as nonconstant uint8 or uint16 grayscale arrays. Exactly equal RGB channels are accepted and collapsed to the original intensity plane without losing bit depth; genuinely unequal color channels, alpha channels, invalid labels, malformed names, damaged files, and unexpected counts are rejected. Montgomery's source README describes 12-bit grayscale stored in PNG; decoding must retain the integer depth instead of forcing 8-bit conversion.

The immutable manifest records cohort, sample ID, relative path, label, group ID, split, byte SHA-256, decoded-pixel SHA-256, dimensions, and dtype. It contains no absolute personal directory. The full SHA-256 and schema checks run before training, and inputs are verified again when loading a prepared manifest. Exact duplicate content within or across cohorts is a stop condition. Near-duplicate, differently cropped or differently compressed/rescaled copies may evade exact hashes and remain a limitation.

## Splitting and patient identity

Without a verified patient table, the code performs a stratified case split with seed 2026; filename IDs are case IDs, not validated patient identities. With `patient_map.csv`, supply `dataset,sample_id,patient_id` for every image, using a single reliable patient identity namespace across cohorts. The splitter keeps groups together and seeks an approximately 80/20 allocation with both classes; it refuses impossible splits and cross-cohort group overlap. Do not manufacture IDs or infer distinct patients from distinct filenames.

The external cohort is kept out of fitting, preprocessing-parameter tuning, threshold selection, model selection, and hyperparameter search. Integrity checks may read all images and ground-truth labels to confirm the input schema; they do not calculate model performance. At implementation time, never-used status is an owner-confirmed protocol assumption from the approved plan, not something code can prove.

## Limits

Patient identities, acquisition chronology, verified demographic coverage, and disease-severity strata are not established by the supplied local materials. These cohorts are small and differ in acquisition setting; external performance may reflect site shift. The nearly balanced development set does not match routine screening prevalence, so precision, calibration, and clinical thresholds cannot be generalized directly. Test confidence intervals quantify sampling uncertainty conditional on this protocol and do not prove clinical usefulness.
