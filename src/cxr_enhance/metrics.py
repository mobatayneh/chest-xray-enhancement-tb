"""Binary metrics and paired uncertainty without treating seeds as patients."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, log_loss, precision_recall_fscore_support, roc_auc_score,
)


def classification_metrics(labels, probabilities, threshold: float = 0.5) -> dict:
    truth = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or scores.ndim != 1 or len(truth) != len(scores) or not len(truth):
        raise ValueError("Labels and probabilities must be nonempty, aligned one-dimensional arrays")
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary ground truth and finite probabilities")
    if np.any((scores < 0) | (scores > 1)) or not 0 <= threshold <= 1:
        raise ValueError("Probabilities and threshold must lie in [0, 1]")
    if len(np.unique(truth)) != 2:
        raise ValueError("Both labeled classes are required for the evaluation protocol")
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    precision, recall, fscore, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "n": int(len(truth)), "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "precision": float(precision[1]), "recall": float(recall[1]), "f1": float(fscore[1]),
        "macro_f1": float(fscore.mean()), "weighted_f1": float(np.average(fscore, weights=support)),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "brier": float(brier_score_loss(truth, scores)),
        "log_loss": float(log_loss(truth, scores, labels=[0, 1])),
        "confusion_matrix": matrix.tolist(),
        "per_class": {name: {"precision": float(precision[index]), "recall": float(recall[index]),
                              "f1": float(fscore[index]), "support": int(support[index])}
                      for index, name in enumerate(("normal", "tb_consistent"))},
    }


def paired_bootstrap(predictions: pd.DataFrame, samples: int = 2000, seed: int = 2026) -> dict:
    """Resample groups jointly for every method and training seed; report mean AUC differences."""
    required = {"method", "seed", "sample_id", "group_id", "label", "probability"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing prediction fields: {required - set(predictions.columns)}")
    if predictions.duplicated(["method", "seed", "sample_id"]).any():
        raise ValueError("Duplicate prediction identities")
    if predictions.groupby("sample_id")[["label", "group_id"]].nunique().to_numpy().max() != 1:
        raise ValueError("Labels or groups disagree between runs")
    methods = sorted(predictions.method.unique())
    if "identity" not in methods:
        raise ValueError("Paired comparisons require an identity baseline")
    seeds = sorted(int(value) for value in predictions.seed.unique())
    identities = sorted(predictions.sample_id.unique())
    arrays = {}
    for method in methods:
        matrices = []
        for training_seed in seeds:
            selected = predictions[(predictions.method == method) & (predictions.seed == training_seed)]
            if set(selected.sample_id) != set(identities):
                raise ValueError("All runs must predict exactly the same evaluation cases")
            matrices.append(selected.set_index("sample_id").loc[identities, "probability"].to_numpy())
        arrays[method] = np.asarray(matrices)
    reference = predictions.drop_duplicates("sample_id").set_index("sample_id").loc[identities]
    truth = reference.label.to_numpy()
    groups = reference.group_id.to_numpy()
    group_members = {group: np.flatnonzero(groups == group) for group in np.unique(groups)}
    group_classes = {group: np.unique(truth[index]) for group, index in group_members.items()}
    homogeneous = all(len(classes) == 1 for classes in group_classes.values())
    strata = [list(group_members)]
    if homogeneous:
        strata = [[group for group in group_members if group_classes[group][0] == label] for label in (0, 1)]
    generator = np.random.default_rng(seed)
    distributions = {method: [] for method in methods}
    attempts = 0
    while len(distributions["identity"]) < samples and attempts < samples * 20:
        attempts += 1
        selected_groups = [group for stratum in strata for group in generator.choice(stratum, len(stratum))]
        indices = np.concatenate([group_members[group] for group in selected_groups])
        if len(np.unique(truth[indices])) < 2:
            continue
        for method in methods:
            distributions[method].append(float(np.mean([
                roc_auc_score(truth[indices], scores[indices]) for scores in arrays[method]
            ])))
    if len(distributions["identity"]) != samples:
        raise ValueError("Insufficient valid bootstrap replicates")
    baseline = np.asarray(distributions["identity"])
    result = {}
    for method in methods:
        values = np.asarray(distributions[method])
        difference = values - baseline
        point = float(np.mean([roc_auc_score(truth, scores) for scores in arrays[method]]))
        baseline_point = float(np.mean([roc_auc_score(truth, scores) for scores in arrays["identity"]]))
        result[method] = {"mean_roc_auc": point, "roc_auc_ci95": np.quantile(values, [.025, .975]).tolist(),
                          "difference_vs_identity": point - baseline_point,
                          "paired_difference_ci95": np.quantile(difference, [.025, .975]).tolist()}
    return {"replicates": samples, "seed": seed, "unit": "group_id", "stratified": homogeneous,
            "training_seeds": seeds, "n_cases": len(identities), "n_groups": len(group_members),
            "interpretation": "Case-sampling uncertainty conditional on these fitted seeds; not clinical validation.",
            "primary_contrast": "corrected_unsharp minus identity", "methods": result}
