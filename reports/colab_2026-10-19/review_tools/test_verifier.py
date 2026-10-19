"""Cross-check independently implemented verification mathematics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from verify_returned_results import compare, independent_bootstrap, independent_metrics


def test_metrics_check_known_counts_and_reject_bad_classifications():
    predictions = pd.DataFrame({"label": [0, 0, 1, 1], "probability": [0.1, 0.6, 0.9, 0.4], "prediction": [0, 1, 1, 0]})
    result = independent_metrics(predictions, 0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["roc_auc"] == 0.75
    assert result["sensitivity"] == result["specificity"] == 0.5
    predictions.loc[0, "prediction"] = 1
    with pytest.raises(AssertionError):
        independent_metrics(predictions, 0.5)


def test_pairwise_bootstrap_matches_direct_sklearn_with_ties():
    labels = np.array([0, 0, 0, 1, 1])
    score_vectors = {
        ("identity", 42): [0.2, 0.5, 0.7, 0.5, 0.8],
        ("identity", 43): [0.1, 0.4, 0.9, 0.8, 0.4],
        ("corrected_unsharp", 42): [0.2, 0.8, 0.4, 0.4, 0.7],
        ("corrected_unsharp", 43): [0.3, 0.5, 0.8, 0.5, 0.9],
    }
    rows = [{"method": method, "seed": seed, "sample_id": f"case_{index}", "group_id": f"group_{index}",
             "label": int(labels[index]), "probability": probability}
            for (method, seed), probabilities in score_vectors.items() for index, probability in enumerate(probabilities)]
    frame = pd.DataFrame(rows)
    actual = independent_bootstrap(frame, 101, 2026)
    generator = np.random.default_rng(2026)
    distributions = {method: [] for method in frame.method.unique()}
    for _ in range(101):
        negatives = generator.choice(["group_0", "group_1", "group_2"], 3)
        positives = generator.choice(["group_3", "group_4"], 2)
        selected = [int(group[-1]) for group in [*negatives, *positives]]
        for method in distributions:
            distributions[method].append(np.mean([
                roc_auc_score(labels[selected], np.asarray(score_vectors[(method, seed)])[selected]) for seed in (42, 43)
            ]))
    for method, values in distributions.items():
        assert np.allclose(actual[method]["roc_auc_ci95"], np.quantile(values, [0.025, 0.975]))
        paired = np.asarray(values) - np.asarray(distributions["identity"])
        assert np.allclose(actual[method]["paired_difference_ci95"], np.quantile(paired, [0.025, 0.975]))


def test_nested_metric_comparison_detects_changes():
    differences = []
    compare({"accuracy": 0.8, "matrix": [[3, 1], [2, 4]]}, {"accuracy": 0.9, "matrix": [[3, 1], [2, 4]]}, "run", differences)
    assert len(differences) == 1
    assert differences[0]["context"] == "run.accuracy"
