"""Validated, serializable experiment configuration."""

import json
from pathlib import Path

METHODS = ("identity", "legacy_unsharp", "corrected_unsharp", "gaussian_hef", "ideal_hef")
DEFAULT_CONFIG = {
    "model": "efficientnet_v2_s",
    "weights": "IMAGENET1K_V1",
    "image_size": 384,
    "epochs": 10,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "eta_min": 0.000001,
    "optimizer": "sgd",
    "scheduler": "cosine",
    "batch_size": 32,
    "microbatch_size": 8,
    "workers": 2,
    "amp": True,
    "rotation_degrees": 10.0,
    "rotation_probability": 0.5,
    "threshold": 0.5,
    "deterministic": True,
    "legacy_geometry": False,
    "selection_metric": "roc_auc",
    "methods": list(METHODS),
    "seeds": [42, 43, 44],
    "split_seed": 2026,
    "bootstrap_samples": 2000,
    "synthetic": False,
}


def validate_config(config: dict) -> dict:
    """Reject accidental protocol changes caused by misspelled or invalid settings."""
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **config}
    for key in ("image_size", "epochs", "batch_size", "microbatch_size", "bootstrap_samples"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if result["microbatch_size"] > result["batch_size"]:
        raise ValueError("microbatch_size cannot exceed batch_size")
    if result["batch_size"] % result["microbatch_size"]:
        raise ValueError("batch_size must be divisible by microbatch_size")
    if type(result["workers"]) is not int or result["workers"] < 0:
        raise ValueError("workers must be a nonnegative integer")
    if result["model"] not in ("efficientnet_v2_s", "efficientnet_v2_m", "tiny"):
        raise ValueError("Unsupported model")
    if result["model"] == "tiny" and not result["synthetic"]:
        raise ValueError("The tiny model is restricted to synthetic smoke tests")
    if result["synthetic"] and result["weights"] is not None:
        raise ValueError("Synthetic tests must not download pretrained weights")
    if result["weights"] not in (None, "IMAGENET1K_V1"):
        raise ValueError("Use an explicit pretrained weight identifier or null")
    if result["optimizer"] not in ("sgd", "adamw") or result["scheduler"] not in ("cosine", "plateau"):
        raise ValueError("Unsupported optimizer or scheduler")
    if result["selection_metric"] not in ("roc_auc", "accuracy"):
        raise ValueError("Unsupported checkpoint selection metric")
    for key in ("amp", "deterministic", "legacy_geometry", "synthetic"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("threshold", "rotation_probability"):
        if not 0 <= result[key] <= 1:
            raise ValueError(f"{key} must lie in [0, 1]")
    if result["threshold"] != 0.5:
        raise ValueError("The approved protocol fixes threshold at 0.5")
    if result["selection_metric"] != "roc_auc" and not result["legacy_geometry"]:
        raise ValueError("Accuracy selection is restricted to historical recipe adapters")
    for key in ("learning_rate", "weight_decay", "eta_min", "rotation_degrees"):
        if not isinstance(result[key], (int, float)) or not 0 <= result[key] < float("inf"):
            raise ValueError(f"{key} must be finite and nonnegative")
    if result["learning_rate"] <= 0 or result["eta_min"] > result["learning_rate"]:
        raise ValueError("Require 0 < learning_rate and eta_min <= learning_rate")
    if not 0 <= result["momentum"] < 1:
        raise ValueError("momentum must lie in [0, 1)")
    if not result["methods"] or len(set(result["methods"])) != len(result["methods"]):
        raise ValueError("methods must be nonempty and unique")
    if set(result["methods"]) - set(METHODS):
        raise ValueError("Unknown enhancement method")
    if not result["seeds"] or len(set(result["seeds"])) != len(result["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in result["seeds"]):
        raise ValueError("seeds must be unsigned 32-bit integers")
    if type(result["split_seed"]) is not int or not 0 <= result["split_seed"] < 2**32:
        raise ValueError("split_seed must be an unsigned 32-bit integer")
    return result


def load_config(path: str | Path | None = None) -> dict:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")) if path else {})
