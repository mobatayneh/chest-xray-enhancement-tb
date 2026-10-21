# References and Attribution

## Dataset sources

1. Jaeger S, Candemir S, Antani S, Wang YX, Lu PX, Thoma G. Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. *Quantitative Imaging in Medicine and Surgery*. 2014;4(6):475-477. [DOI:10.3978/j.issn.2223-4292.2014.11.20](https://doi.org/10.3978/j.issn.2223-4292.2014.11.20).
2. [NLM CHRB resources](https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html), with links to Shenzhen and Montgomery datasets, checked 2026-10-17.
3. [Montgomery source README](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf), pp.1-2: image/label description, usage request, and attribution. It requests citations to Candemir et al., *IEEE TMI* 2014;33(2):577-590, [DOI:10.1109/TMI.2013.2290491](https://doi.org/10.1109/TMI.2013.2290491), and Jaeger et al., *IEEE TMI* 2014;33(2):233-245, [DOI:10.1109/TMI.2013.2284099](https://doi.org/10.1109/TMI.2013.2284099).
4. [Shenzhen source directory](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html), including `NLM-ChinaCXRSet-ReadMe.docx`. Dataset rights are separate from paper and code licenses.

## Supplied research papers

5. Munadi K, Muchtar K, Maulina N, Pradhan B. Image Enhancement for Tuberculosis Detection Using Deep Learning. *IEEE Access*. 2020;8:217897-217907. [DOI:10.1109/ACCESS.2020.3041867](https://doi.org/10.1109/ACCESS.2020.3041867). Source of the literature UM/HEF/CLAHE comparison with EfficientNet-B4 and ResNet variants, not the project's EfficientNet-V2 experiment.
6. Lin CH, Wu JX, Li CM, Chen PY, Pai NS, Kuo YC. Enhancement of Chest X-Ray Images to Improve Screening Accuracy Rate Using Iterated Function System and Multilayer Fractional-Order Machine Learning Classifier. *IEEE Photonics Journal*. 2020;12(4), article **4100218**, pp.1-18. [DOI:10.1109/JPHOT.2020.3013193](https://doi.org/10.1109/JPHOT.2020.3013193). The supplied comparative report's article number and slide deck's 2023 date are corrected here.
7. Ishikawa R, Yuzawa T, Fukiage T, Kagesawa M, Watsuji T, Oishi T. Visibility Enhancement of Lesion Regions in Chest X-Ray Images With Image Fidelity Preservation. *IEEE Access*. 2025;13:11080-11094. [DOI:10.1109/ACCESS.2025.3528489](https://doi.org/10.1109/ACCESS.2025.3528489). This image-visibility objective and its metrics differ from binary TB classification; no implementation of this paper is claimed.

Bibliographic details above were checked against the supplied papers' front matter and printed page numbers. Their PDFs and figures are not redistributed in the source package. Published figures/tables are cited as reported evidence, not independently reproduced results.

## Model and software

8. Tan M, Le Q. EfficientNetV2: Smaller Models and Faster Training. *Proceedings of ICML*. 2021;139:10096-10106. [PMLR paper](https://proceedings.mlr.press/v139/tan21a.html).
9. [Torchvision EfficientNet-V2-S documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html). The experiment records the exact installed Torch/Torchvision versions and explicit pretrained weight enum; architecture/pretrained weights are third-party contributions.
10. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html), [automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html), and [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html). Runtime dependencies and actual versions are recorded separately.

## Contribution boundaries

Original course-project materials are attributed to Mohammad Albatayneh as identified in their reports. Dataset collection/annotation, published methods, pretrained weights, and underlying libraries retain their respective authorship. The reconstruction supplies software engineering, auditing, and documentation; it does not retroactively establish individual authorship of every historical cell or a new research contribution. Contributor/license choices require owner review before publication.
