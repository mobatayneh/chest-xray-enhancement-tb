# Independent Results Review

Review date: **2026-10-19**. Status: **PASS for numerical consistency, completeness, and recorded provenance; full model/data replay remains unverified**.

This review uses the actual user-returned `results_for_review.zip` and executed Colab notebook. It does not rerun training, alter the frozen experiment, or replace original evidence. The training implementation, configuration templates, and delivered clean Colab notebook retain their original bytes.

## Evidence identity

Results ZIP SHA-256: `655b3c6be8f14eb507004d5aed7acd19e6e690692185b0c1d5ae4f2bddea44e6`.

The public [evidence directory](../reports/colab_2026-10-19/README.md) contains aggregate results and generated figures. The original ZIP, executed notebook, case-level labels/predictions, split identities, and source images are not published. [Input provenance](../reports/colab_2026-10-19/input_provenance.json) records the input hashes. SHA-256 establishes consistency and detects changed bytes; it does not independently authenticate every claimed historical action.

## Checks actually performed

| Check | Outcome | Scope |
|---|---|---|
| ZIP structure and SHA-256 coverage | PASS: 168 members; 167 checksummed files | All members except the checksum inventory itself; includes all four provenance additions |
| External predictions | PASS: 15 complete runs, 2,070 rows | 138 identical Montgomery cases per run, 80 normal and 58 TB-consistent; not 2,070 independent patients |
| Validation predictions | PASS: 15 runs, 1,995 rows | Same 133 Shenzhen validation cases per run, 65 normal and 68 TB-consistent |
| Independent metric recomputation | PASS | All external and validation metric dictionaries, including confusion matrices and classwise scores; sklearn implementation independent of the exported summary |
| Aggregate reports | PASS | Per-run JSON, comparison CSV, seed mean/sample SD, and saved threshold decisions agree to tolerance 1e-10 |
| Paired bootstrap | PASS: 2,000 replicates | Independent weighted pairwise ranking-statistic implementation reproduces AUC intervals and method-minus-identity intervals using seed 2026 |
| Development split | PASS | Reproduced stratified seed-2026 529/133 split; test cohort remains all 138 Montgomery cases; no recorded case/group overlap |
| Hash duplicates in manifest | PASS | No repeated byte/pixel hashes in the exported manifest; full original image bytes were not independently reread |
| Training history | PASS: 150 rows | Every expected run contains epochs 1 through 10; finite recorded metrics and consistent configuration |
| Model selection | PASS: 15/15 | Highest validation ROC-AUC, then lower validation loss, then earliest epoch; selected checkpoint identity agrees across metadata |
| Source and settings | PASS | 32 notebook-pinned package files, including all 13 runtime Python files, match the delivered source; configuration/environment contracts agree |
| Manifest and preparation digests | PASS | Original manifest paths reconstructed from recorded layout; original preparation roots recovered from notebook output; both original hashes reproduced |
| Executed notebook | PASS with chronology limitation | 20 cells, 10 code cells; schema and syntax valid; no saved error outputs; 137 saved epoch lines agree with histories |
| Colab software checks | Recorded PASS: 137 passed, 1 skipped | 31.20 seconds; offline pip-bootstrap wheel test skipped on the Colab Python version |
| Review-verifier self-tests | PASS: 3 tests | Known counts and invalid decisions, bootstrap ties versus sklearn, discrepancy detection |
| Final publication-copy regression | PASS: 141 tests in 61.01 seconds | 138 project tests plus 3 review-tool tests; local CPU software checks only |

The notebook preserves a resumed run. Earlier training output was replaced when the cell was rerun: 13 completed epochs are present in histories but absent from the final saved training log. The added local-image-cache helper and the `clean-v2` output directory are documented execution changes, not changes to the classifier or physical batch. The notebook's `language_info.version` says 3.12, while actual interpreter outputs and experiment metadata consistently identify Python **3.13.15**; use the latter.

Two different source hashes are expected: the training/checkpoint contract and suite/software checks use different fingerprint algorithms over the same 13 runtime Python files. Both reproduce successfully; their different strings are not evidence of code drift.

## Verified conclusion

Corrected unsharp mean external ROC-AUC is **0.819361**, compared with **0.829777** without enhancement. The primary mean difference is **-0.010417**, paired 95% interval **[-0.042029, +0.019616]**. This does not establish improvement, harm, or equivalence. Each exploratory enhancement-versus-identity interval also includes zero. See the [final report](FINAL_REPORT.md) for all methods, seeds, metrics, and interpretation.

The numerical values in the returned results did not require correction. Corrections are to old claims, stale project status, and interpretation: validation performance is not external-test performance, a larger point estimate is not confirmed superiority, and seed replication does not increase the number of independent cases.

## What remains unverified

- The large `.pt` files were not returned. Their 150 recorded checksums and 15 selected identities are internally consistent, but their bytes were not independently rehashed and inference was not replayed from weights.
- Complete image cohorts were not returned. Recorded hashes and split arithmetic can be checked; image corruption, duplicate hashing, and label correspondence cannot be independently repeated against the original images here.
- Groups are filename-derived case IDs. Repeated patients, transformed near-duplicates, demographic performance, pretrained-data overlap, and prospective clinical validity are not established.
- A saved evaluation lock supports the recorded protocol. It cannot prove the absence of unrecorded earlier external-test access. The user's declaration that Montgomery was previously unused remains part of the protocol provenance.
- Three seeds and one small external site give limited uncertainty coverage. Bootstrap intervals condition on these fitted models and this development cohort, rather than quantifying every possible training-data variation.

## Reproduce the review

With authorized access to the private returned ZIP and the documented dependencies:

```bash
python -m cxr_enhance verify /path/to/results_for_review.zip
```

This built-in check verifies all member hashes and external metric consistency without loading model weights. The independently executed verification record and notebook findings are retained with the aggregate evidence. A complete computational reproduction additionally requires authorized dataset access and the selected checkpoint files. Do not tune thresholds or preprocessing against the now-observed Montgomery results.

The independent audit implementation and its three self-tests are also retained. Run it with normal Python (without optimization flags that disable assertions) and a new output directory:

```bash
python reports/colab_2026-10-19/review_tools/verify_returned_results.py /path/to/results_for_review.zip --source src/cxr_enhance --output /path/to/new-review
python -m pytest reports/colab_2026-10-19/review_tools/test_verifier.py -q
```

This audit is specific to the frozen 15-run October 2026 study, not a general-purpose validator for arbitrary datasets. Its standalone archive audit cannot reconstruct stripped local paths; the supplementary notebook review supplies that provenance check.
