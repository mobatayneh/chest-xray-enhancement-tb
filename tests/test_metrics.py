import numpy as np
import pandas as pd
import pytest

from cxr_enhance.metrics import classification_metrics, paired_bootstrap


def test_known_confusion_and_auc():
    result = classification_metrics([0, 0, 1, 1], [.1, .6, .4, .9])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["roc_auc"] == .75
    assert result["accuracy"] == .5
    assert result["sensitivity"] == .5
    assert result["specificity"] == .5
    assert result["brier"] == pytest.approx(.185)


@pytest.mark.parametrize("labels,scores", [([0, 0], [.2, .3]), ([0, 2], [.1, .9]),
                                         ([0, 1], [.1, np.nan]), ([0, 1], [.1, 1.1])])
def test_invalid_evaluation_rejected(labels, scores):
    with pytest.raises(ValueError):
        classification_metrics(labels, scores)


def test_paired_bootstrap_preserves_cases_across_seeds():
    import json

    rows = [{"method": method, "seed": seed, "sample_id": str(index), "group_id": str(index),
             "label": label, "probability": score}
            for method in ("identity", "corrected_unsharp") for seed in (42, 43, 44)
            for index, (label, score) in enumerate(zip([0, 0, 1, 1], [.1, .6, .4, .9]))]
    frame = pd.DataFrame(rows)
    result = paired_bootstrap(frame, samples=30)
    assert result["n_cases"] == 4
    json.dumps(result, allow_nan=False)
    assert result["methods"]["corrected_unsharp"]["paired_difference_ci95"] == [0.0, 0.0]
    assert result == paired_bootstrap(frame, samples=30)
    with pytest.raises(ValueError, match="same evaluation cases"):
        paired_bootstrap(frame.iloc[1:], samples=10)
