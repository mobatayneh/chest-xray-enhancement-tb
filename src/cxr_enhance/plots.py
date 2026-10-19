"""Portable plots derived only from saved labels, scores, and histories."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def plot_evaluation(predictions: pd.DataFrame, history: pd.DataFrame, destination: Path,
                    synthetic: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.bbox": "tight"})
    truth = predictions.label.to_numpy()
    scores = predictions.probability.to_numpy()
    labels = predictions.prediction.to_numpy()
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(truth, labels, labels=[0, 1],
                                            display_labels=["Normal", "TB-consistent"], ax=axis,
                                            colorbar=False, cmap="Blues")
    axis.set_title("SYNTHETIC software check" if synthetic else "Held-out external cases")
    figure.savefig(destination / "confusion_matrix.png")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    if synthetic:
        figure.suptitle("SYNTHETIC software check - not research results")
    RocCurveDisplay.from_predictions(truth, scores, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(truth, scores, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axes[1].axhline(float(np.mean(truth)), color="gray", linestyle="--", label="Prevalence")
    axes[0].set_title("ROC curve")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(destination / "roc_pr_curves.png")
    plt.close(figure)
    fraction, means = calibration_curve(truth, scores, n_bins=5, strategy="uniform")
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(means, fraction, "o-", label="Observed")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axis.set(xlabel="Mean predicted probability", ylabel="Fraction labeled positive",
             title="SYNTHETIC calibration check" if synthetic else "Calibration (5 bins)")
    axis.legend()
    figure.savefig(destination / "calibration.png")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    if synthetic:
        figure.suptitle("SYNTHETIC software check - not research results")
    loss_columns = [column for column in history if "loss" in column and column != "learning_rate"]
    for column in loss_columns:
        axes[0].plot(history.epoch, history[column], marker="o", label=column.replace("_", " "))
    for column in history:
        if "roc_auc" in column or column in ("val_auc", "validation_auc"):
            axes[1].plot(history.epoch, history[column], marker="o", label=column.replace("_", " "))
    axes[0].set(xlabel="Completed epoch", ylabel="Per-example loss", title="Training and validation loss")
    axes[1].set(xlabel="Completed epoch", ylabel="ROC-AUC", title="Training and validation discrimination")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(destination / "learning_curves.png")
    plt.close(figure)


def plot_comparison(comparison: pd.DataFrame, destination: Path, synthetic: bool) -> None:
    figure, axis = plt.subplots(figsize=(9, 4))
    for position, (method, rows) in enumerate(comparison.groupby("method", sort=True)):
        axis.scatter(np.full(len(rows), position), rows.roc_auc, s=40)
        axis.plot([position - .15, position + .15], [rows.roc_auc.mean()] * 2, color="black")
    names = sorted(comparison.method.unique())
    axis.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names])
    axis.set(ylim=(0, 1), ylabel="External ROC-AUC per training seed",
             title="SYNTHETIC SOFTWARE CHECK - no research result" if synthetic else "Frozen external evaluation")
    axis.grid(axis="y", alpha=.2)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
