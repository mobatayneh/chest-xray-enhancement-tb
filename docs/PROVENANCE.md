# Provenance and Preservation

Original files are retained unchanged in their supplied location. They are not copied into this distributable source package. This preserves the owner's research history while excluding medical images, embedded example figures, third-party papers, and personal absolute paths from a future code release.

## Completed Colab experiment, 2026-10-19

The owner supplied `results_for_review.zip` and the executed `04_colab_improved_training.ipynb` after completing the 15-run study. The private archive is the primary evidence for the new results; it is retained unchanged. The reports distinguish these measured external-test results from historical notebook outputs and synthetic software checks. [RESULTS_REVIEW.md](RESULTS_REVIEW.md) records integrity checks and review boundaries, and [FINAL_REPORT.md](FINAL_REPORT.md) presents the corrected scientific interpretation.

The review verified all 167 checksummed archive members, recomputed all 15 external and 15 selected-checkpoint validation result sets, reproduced the fixed split and paired bootstrap analysis, and checked 150 epoch records against the selection rule. All 13 runtime Python files and 32 pinned release files match the recorded source evidence. The executed notebook preserves resumed execution with 137 retained matching epoch log entries; the missing earlier 13 logs do not replace the complete exported history. These checks support the recorded experiment without implying access to unreturned image or checkpoint bytes.

The public-facing evidence subset under `reports/colab_2026-10-19/` supports the aggregate results without distributing the raw images, private case-level predictions/identifiers, large checkpoints, or personal runtime paths. A derived report is not a replacement for the original archive. Archive digests and any transformations used for the publication subset must remain recorded in the review evidence.

Checkpoint metadata supplies filenames, selected epochs, and digests. The checkpoint binaries and source images were not returned, so digest consistency in metadata is not an independent hash check of those binaries, nor proof of a successful replay. Likewise, a recorded source or environment hash is evidence of the declared run contract, not a guarantee that every historical action can be reconstructed. No training-code, configuration, or notebook edits are part of the Phase 4 documentation update.

## Evidence files

- `provenance/source_manifest.json`: all 13 source filenames, byte sizes, SHA-256 hashes, and byte-identical artifact groups.
- `provenance/notebook_audit.json`: original source-cell hashes, cell indices, AST syntax status, presence of recorded errors, and parsed historical epoch metrics.
- `provenance/historical_metrics.csv`: all 17 recorded confusion matrices with recomputed accuracy, sensitivity, and specificity. These are historical validation outputs, not a test leaderboard or current experiment results.

`scripts/audit_originals.py` reads originals without importing or executing their code. Run it using a local source directory and a **new** evidence output directory:

```bash
python scripts/audit_originals.py --source-dir /path/to/private-originals --output-dir /path/to/new-audit-evidence
```

It refuses to write inside the original directory and refuses to overwrite an existing evidence file. Archive each new audit separately; compare its hashes to the delivered inventory. File mtimes are not treated as experiment chronology. Source code hashes identify inspected cells; stored notebook outputs may originate from an older execution than the visible code, which cannot be excluded without execution/checkpoint provenance.

The comparative-study PDF and its `- Copy` file have equal hashes, so they count as two preserved artifacts but one distinct source document. The 17 AST passes establish syntactic validity only. Arithmetic agreement with a confusion matrix establishes consistency of a printed accuracy only; it cannot validate labels, unseen patient overlap, checkpoint provenance, AUC, or real training execution.

## Evidence hierarchy

1. Supplied notebook source and stored outputs support statements about historical implementation and printed results.
2. Supplied reports/Word/slides express claims that must be checked against that source; they are not proof that an experiment ran.
3. Supplied research papers support literature attribution and reported comparisons, not the project author's measured performance.
4. Source-provider documentation supports dataset provenance/expected counts/usage instructions. Returned preparation metadata records local checks; independently rechecking image contents requires the private images.
5. The returned Colab manifest, predictions, environment metadata, histories, and checksums support the rebuilt experiment within the bounds of the results review. Saved predictions allow metric recomputation. Checkpoint metadata alone cannot replace actual model weights for inference replay.

## Publication boundary

No code license is selected without owner authorization. A license for future original code cannot relicense source datasets, pretrained weights, papers, or collaborators' work. Keep original notebooks and reports in private archival storage unless their embedded data, rights, attribution, and personal information have been explicitly reviewed. A public repository should link to authorized dataset access rather than host dataset copies.

The implementation delivery completed Phase 3 before the owner ran training. Phase 4 now reviews the returned experiment and prepares the requested `chest-xray-enhancement-tb` repository. Repository publication is a separate owner-authorized action, and no publication is implied by the local documentation. Git branches/commits/pushes and CV/LinkedIn updates are not evidence of experimental validity. CV/LinkedIn work remains Phase 5 after repository finalization.
