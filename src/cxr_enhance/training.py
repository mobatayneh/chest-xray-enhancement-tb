"""Paired-arm training, epoch-boundary recovery, and resource preflight."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_m,
    efficientnet_v2_s,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .checkpoints import (
    canonical_hash,
    latest_checkpoint,
    load_checkpoint,
    local_lock,
    restore_rng,
    rng_state,
    save_checkpoint,
    sha256_file,
    source_hash,
    validate_contract,
    write_json,
)
from .filters import preprocess_image, read_image
from .data import validate_prepared_manifest
from .metrics import classification_metrics

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _device(config: dict[str, Any]) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This reproducibility protocol supports CPU and CUDA devices")
    return device


def environment_metadata(device: torch.device) -> dict[str, Any]:
    packages = {}
    for package in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "opencv-python-headless", "pillow", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "device_type": device.type, "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_threads": torch.get_num_threads(),
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
    }


def build_model(config: dict[str, Any]) -> nn.Module:
    model_name = config.get("model", "efficientnet_v2_s")
    weights = config.get("weights", "IMAGENET1K_V1")
    if weights not in {None, "IMAGENET1K_V1"}:
        raise ValueError("weights must be null or IMAGENET1K_V1")
    if model_name == "tiny":
        if not config.get("synthetic", False):
            raise ValueError("The tiny network is restricted to explicitly synthetic smoke tests")
        return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))
    if model_name == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if weights else None)
    elif model_name == "efficientnet_v2_m":
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1 if weights else None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


class ImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame | list[dict[str, Any]], data_roots: dict[str, Path],
                 config: dict[str, Any], method: str, seed: int = 0, epoch: int = 0,
                 training: bool = False) -> None:
        self.rows = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
        self.roots = {key: Path(value).resolve() for key, value in data_roots.items()}
        self.config, self.method = config, method
        self.seed, self.epoch, self.training = seed, epoch, training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        root = self.roots[row["dataset"]]
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path escapes dataset root: {row['sample_id']}")
        image = preprocess_image(read_image(path), self.method,
                                 image_size=int(self.config.get("image_size", 384)),
                                 legacy_geometry=bool(self.config.get("legacy_geometry", False)))
        tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
        if self.training:
            identifier = f"{self.seed}:{self.epoch}:{row['dataset']}:{row['sample_id']}"
            local_seed = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "big")
            generator = random.Random(local_seed)
            if generator.random() < self.config.get("rotation_probability", 0.5):
                degrees = float(self.config.get("rotation_degrees", 10))
                tensor = vision_functional.rotate(tensor, generator.uniform(-degrees, degrees),
                                                   interpolation=InterpolationMode.BILINEAR, fill=0)
        tensor = tensor.repeat(3, 1, 1)
        tensor = vision_functional.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor, torch.tensor(float(row["label"]), dtype=torch.float32)


def _loader(dataset: ImageDataset, config: dict[str, Any], seed: int,
            shuffle: bool, device: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    workers = int(config.get("workers", 2))
    return DataLoader(dataset, batch_size=int(config.get("microbatch_size", 8)),
                      shuffle=shuffle, num_workers=workers, generator=generator,
                      worker_init_fn=_worker_seed, pin_memory=device.type == "cuda",
                      persistent_workers=False, drop_last=False)


def _validate_config(config: dict[str, Any]) -> None:
    for key, default in (("epochs", 10), ("batch_size", 32), ("microbatch_size", 8), ("image_size", 384)):
        value = config.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("batch_size", 32) % config.get("microbatch_size", 8):
        raise ValueError("batch_size must be divisible by the fixed microbatch_size")
    if int(config.get("workers", 2)) < 0:
        raise ValueError("workers must be nonnegative")
    if not 0 <= config.get("rotation_probability", 0.5) <= 1:
        raise ValueError("rotation_probability must be between zero and one")
    if float(config.get("rotation_degrees", 10)) < 0:
        raise ValueError("rotation_degrees must be nonnegative")
    if config.get("selection_metric", "roc_auc") != "roc_auc" and not (
            config.get("legacy_geometry", False) and config.get("selection_metric") == "accuracy"):
        raise ValueError("The main protocol selects validation ROC-AUC; accuracy is limited to historical recipes")
    if float(config.get("threshold", 0.5)) != 0.5:
        raise ValueError("This protocol predeclares a fixed decision threshold of 0.5")
    if config.get("model") == "tiny" and not config.get("synthetic", False):
        raise ValueError("tiny model requires synthetic=true")


def _optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    settings = {"lr": float(config.get("learning_rate", 0.01)),
                "weight_decay": float(config.get("weight_decay", 0.0005))}
    name = config.get("optimizer", "sgd")
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=float(config.get("momentum", 0.9)), **settings)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **settings)
    raise ValueError(f"Unsupported optimizer: {name}")


def _scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Any:
    if config.get("scheduler", "cosine") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config.get("epochs", 10)), eta_min=float(config.get("eta_min", 1e-6)))
    if config["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    raise ValueError(f"Unsupported scheduler: {config['scheduler']}")


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
               config: dict[str, Any], optimizer: torch.optim.Optimizer | None = None,
               scaler: Any = None) -> tuple[float, list[int], list[float]]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    effective_batch = int(config.get("batch_size", 32))
    accumulation_steps = effective_batch // int(config.get("microbatch_size", 8))
    samples_seen, total_loss = 0, 0.0
    labels, probabilities = [], []
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for batch_index, (images, targets) in enumerate(loader):
        if batch_index % accumulation_steps == 0:
            window_size = min(effective_batch, len(loader.dataset) - samples_seen)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and config.get("amp", True)):
                logits = model(images).reshape(-1)
                summed_loss = criterion(logits, targets)
            if not torch.isfinite(summed_loss):
                raise FloatingPointError("Non-finite loss; checkpoint remains at the last complete epoch")
            if training:
                scaler.scale(summed_loss / window_size).backward()
                end_of_window = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
                if end_of_window:
                    clip = config.get("gradient_clip_norm")
                    if clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip), error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        total_loss += float(summed_loss.detach().cpu())
        samples_seen += len(targets)
        labels.extend(targets.detach().cpu().int().tolist())
        probabilities.extend(torch.sigmoid(logits.detach().float()).cpu().tolist())
    if samples_seen == 0:
        raise ValueError("Cannot run an epoch on an empty dataset")
    return total_loss / samples_seen, labels, probabilities


def predict_rows(model: nn.Module, rows: pd.DataFrame | list[dict[str, Any]],
                 data_roots: dict[str, Path], config: dict[str, Any], method: str,
                 device: str | torch.device) -> pd.DataFrame:
    rows = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    _verify_training_files(rows, data_roots)
    device = torch.device(device)
    dataset = ImageDataset(rows, data_roots, config, method)
    loader = _loader(dataset, config, seed=0, shuffle=False, device=device)
    _, _, probabilities = _run_epoch(model, loader, device, config)
    result = rows[["dataset", "sample_id", "group_id", "label"]].copy()
    result["probability"] = probabilities
    result["prediction"] = (result["probability"] >= float(config.get("threshold", 0.5))).astype(int)
    return result


def _verify_training_files(rows: pd.DataFrame, roots: dict[str, Path]) -> None:
    for row in rows.to_dict("records"):
        root = Path(roots[row["dataset"]]).resolve()
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid or missing image path: {row['sample_id']}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Image changed since preparation: {row['sample_id']}")


def _persist_run_summary(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    history = pd.DataFrame(state["history"])
    history.to_csv(run_dir / "training_history.csv", index=False)
    predictions = pd.DataFrame(state["best_predictions"])
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    best_path = run_dir / "checkpoints" / f"epoch_{state['best_epoch']:04d}.pt"
    metadata = {
        "method": state["contract"]["method"], "seed": state["contract"]["seed"],
        "synthetic": bool(state["config"].get("synthetic", False)),
        "completed_epochs": state["epoch"], "planned_epochs": state["config"].get("epochs", 10),
        "best_epoch": state["best_epoch"], "best_checkpoint": best_path.relative_to(run_dir).as_posix(),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_validation_metrics": state["best_validation_metrics"],
        "best_validation_loss": state["best_loss"], "contract": state["contract"],
        "selection_metric": state["config"].get("selection_metric", "roc_auc"),
        "selection_rule": "Highest validation selection metric, then lowest validation loss, then earliest epoch",
    }
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((run_dir / "checkpoints").glob("epoch_*.json"))
               if path.with_suffix(".pt").exists()]
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "checkpoint_metadata.json", {"selected": metadata, "epochs": records})
    write_json(run_dir / "latest.json", {"epoch": state["epoch"],
                                       "path": f"checkpoints/epoch_{state['epoch']:04d}.pt"})
    write_json(run_dir / "best.json", {"epoch": state["best_epoch"],
                                     "path": f"checkpoints/epoch_{state['best_epoch']:04d}.pt"})
    return metadata


def train_run(config: dict[str, Any], manifest_path: Path, data_roots: dict[str, Path],
              run_dir: Path, method: str, seed: int, resume: bool = False,
              stop_after_epoch: int | None = None) -> dict[str, Any]:
    """Allow only one training writer per run on the current runtime."""
    with local_lock(Path(run_dir), "training"):
        return _train_run(config, manifest_path, data_roots, run_dir, method, seed, resume, stop_after_epoch)


def _train_run(config: dict[str, Any], manifest_path: Path, data_roots: dict[str, Path],
               run_dir: Path, method: str, seed: int, resume: bool = False,
               stop_after_epoch: int | None = None) -> dict[str, Any]:
    """Train only Shenzhen train/validation rows and save every completed epoch immutably."""
    if stop_after_epoch is not None and (type(stop_after_epoch) is not int or stop_after_epoch < 1):
        raise ValueError("stop_after_epoch must be a positive integer")
    run_dir = Path(run_dir)
    for directory in (run_dir.resolve(), *run_dir.resolve().parents):
        if (directory / "evaluation_lock.json").exists():
            raise RuntimeError("Training is forbidden after external evaluation has begun")
    _validate_config(config)
    device = _device(config)
    seed_everything(seed, bool(config.get("deterministic", True)))
    environment = environment_metadata(device)
    manifest_path, run_dir = Path(manifest_path), Path(run_dir)
    if (manifest_path.parent / "data_validation.json").exists() or not config.get("synthetic", False):
        validate_prepared_manifest(manifest_path, config)
    manifest = pd.read_csv(manifest_path, keep_default_na=False, dtype={"sample_id": str, "group_id": str})
    required = {"dataset", "sample_id", "path", "group_id", "label", "split", "sha256"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(manifest.columns))}")
    eligible = manifest[manifest["split"].isin(["train", "validation"])].copy()
    if eligible.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("Duplicate training/validation sample identifiers")
    if eligible["sha256"].duplicated().any():
        raise ValueError("Duplicate image content in training/validation manifest")
    if not config.get("synthetic", False) and set(eligible["dataset"].str.lower()) != {"shenzhen"}:
        raise ValueError("Only Shenzhen can enter training or validation in this protocol")
    if eligible.groupby("group_id")["split"].nunique().gt(1).any():
        raise ValueError("A patient/group crosses the training-validation boundary")
    training_rows = eligible[eligible["split"] == "train"].reset_index(drop=True)
    validation_rows = eligible[eligible["split"] == "validation"].reset_index(drop=True)
    for split_rows in (training_rows, validation_rows):
        if set(split_rows["label"].unique()) != {0, 1}:
            raise ValueError("Training and validation must each contain valid labels 0 and 1")
    _verify_training_files(eligible, data_roots)
    contract = {
        "config_hash": canonical_hash(config), "manifest_hash": sha256_file(manifest_path),
        "prepared_report_hash": sha256_file(manifest_path.parent / "data_validation.json")
        if (manifest_path.parent / "data_validation.json").exists() else None,
        "source_hash": source_hash(), "environment_hash": canonical_hash(environment),
        "method": method, "seed": seed,
    }
    prior = latest_checkpoint(run_dir)
    if prior and not resume:
        raise FileExistsError("Run already contains checkpoints; use resume=True with the same frozen inputs")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "experiment_config.json"
    for path, expected in ((run_dir / "resume_contract.json", contract),
                           (config_path, config), (run_dir / "environment.json", environment)):
        if path.exists():
            actual = json.loads(path.read_text(encoding="utf-8"))
            validate_contract(actual, expected)
        else:
            write_json(path, expected, overwrite=False)
    model = build_model({**config, "weights": None} if prior else config).to(device)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.get("amp", True))
    best_score, best_loss, best_epoch, start_epoch = -math.inf, math.inf, 0, 1
    history, best_predictions, best_metrics = [], [], {}
    if prior:
        state = load_checkpoint(prior, device)
        validate_contract(state["contract"], contract)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
        start_epoch = state["epoch"] + 1
        history, best_predictions = state["history"], state["best_predictions"]
        best_score, best_loss, best_epoch = state["best_score"], state["best_loss"], state["best_epoch"]
        best_metrics = state["best_validation_metrics"]
        _persist_run_summary(run_dir, state)
    final_epoch = min(int(config.get("epochs", 10)), stop_after_epoch or int(config.get("epochs", 10)))
    validation_loader = _loader(ImageDataset(validation_rows, data_roots, config, method), config,
                                seed=seed, shuffle=False, device=device)
    for epoch in range(start_epoch, final_epoch + 1):
        started = time.perf_counter()
        train_dataset = ImageDataset(training_rows, data_roots, config, method, seed, epoch, training=True)
        train_loader = _loader(train_dataset, config, seed=seed + epoch, shuffle=True, device=device)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_labels, train_probs = _run_epoch(model, train_loader, device, config, optimizer, scaler)
        validation_loss, validation_labels, validation_probs = _run_epoch(model, validation_loader, device, config)
        threshold = float(config.get("threshold", 0.5))
        train_metrics = classification_metrics(train_labels, train_probs, threshold=threshold)
        validation_metrics = classification_metrics(validation_labels, validation_probs, threshold=threshold)
        validation_auc = validation_metrics["roc_auc"]
        if validation_auc is None or not math.isfinite(float(validation_auc)):
            raise ValueError("Validation ROC-AUC must be finite for model selection")
        if config.get("scheduler", "cosine") == "plateau":
            scheduler.step(validation_metrics["accuracy"])
        else:
            scheduler.step()
        history.append({"epoch": epoch, "learning_rate": learning_rate,
                        "train_loss": train_loss, "validation_loss": validation_loss,
                        "train_roc_auc": train_metrics["roc_auc"], "validation_roc_auc": validation_auc,
                        "train_accuracy": train_metrics["accuracy"],
                        "validation_accuracy": validation_metrics["accuracy"],
                        "seconds": time.perf_counter() - started})
        selection_score = validation_metrics[config.get("selection_metric", "roc_auc")]
        if (selection_score > best_score) or (selection_score == best_score and validation_loss < best_loss):
            best_score, best_loss, best_epoch = selection_score, validation_loss, epoch
            best_metrics = validation_metrics
            predictions = validation_rows[["dataset", "sample_id", "group_id", "label"]].copy()
            predictions["probability"] = validation_probs
            predictions["prediction"] = (predictions["probability"] >= threshold).astype(int)
            best_predictions = predictions.to_dict("records")
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
            "config": config, "contract": contract, "environment": environment, "history": history,
            "best_score": best_score, "best_loss": best_loss, "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics, "best_predictions": best_predictions,
        }
        save_checkpoint(run_dir, epoch, state)
        metadata = _persist_run_summary(run_dir, state)
        LOGGER.info("%s seed=%s epoch=%s train_loss=%.6f validation_loss=%.6f validation_auc=%.6f",
                    method, seed, epoch, train_loss, validation_loss, validation_auc)
    if prior and start_epoch > final_epoch:
        metadata = _persist_run_summary(run_dir, state)
    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def load_selected_model(run_dir: Path, device: str | torch.device) -> tuple[nn.Module, dict[str, Any]]:
    run_dir, device = Path(run_dir), torch.device(device)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "checkpoints" / f"epoch_{metadata['best_epoch']:04d}.pt"
    if sha256_file(checkpoint) != metadata["best_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint checksum differs from run metadata")
    state = load_checkpoint(checkpoint, device)
    validate_contract(state["contract"], metadata["contract"])
    if state["contract"]["source_hash"] != source_hash():
        raise ValueError("Source code changed since training; evaluation requires the frozen source")
    if state["contract"]["method"] != metadata["method"] or state["contract"]["seed"] != metadata["seed"]:
        raise ValueError("Selected model metadata disagrees with checkpoint method or seed")
    if canonical_hash(config) != state["contract"]["config_hash"]:
        raise ValueError("Saved configuration was changed after training")
    model = build_model({**config, "weights": None}).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, metadata


def preflight(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Measure synthetic forward/backward memory and choose one microbatch for all arms."""
    _validate_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(config)
    seed_everything(2026, bool(config.get("deterministic", True)))
    maximum = int(config.get("microbatch_size", 8))
    candidates = [candidate for candidate in (8, 4, 2, 1)
                  if candidate <= maximum and int(config.get("batch_size", 32)) % candidate == 0]
    if device.type == "cpu":
        candidates = [1]
    model = build_model({**config, "weights": None}).to(device)
    size = int(config.get("image_size", 384))
    successful = None
    attempts = []
    for microbatch in candidates:
        try:
            model.train()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            images = torch.zeros((microbatch, 3, size, size), device=device)
            targets = torch.zeros(microbatch, device=device)
            for iteration in range(2):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and config.get("amp", True)):
                    loss = nn.functional.binary_cross_entropy_with_logits(model(images).reshape(-1), targets)
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            free_bytes = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None
            parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
            optimizer_bytes = parameter_bytes * (2 if config.get("optimizer", "sgd") == "adamw" else 1)
            if free_bytes is not None and free_bytes < optimizer_bytes + 512 * 1024**2:
                raise torch.OutOfMemoryError("Insufficient reserve for optimizer state and evaluation")
            successful = {"microbatch_size": microbatch, "forward_backward_seconds": elapsed,
                          "peak_cuda_allocated_bytes": peak, "optimizer_state_estimate_bytes": optimizer_bytes}
            attempts.append({"microbatch_size": microbatch, "status": "passed"})
            break
        except torch.OutOfMemoryError:
            attempts.append({"microbatch_size": microbatch, "status": "out_of_memory"})
            model.zero_grad(set_to_none=True)
            if "images" in locals():
                del images
            if "loss" in locals():
                del loss
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if successful is None:
        raise RuntimeError("No safe microbatch found; use a larger-memory runtime")
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    checkpoint_bytes = parameter_bytes + successful["optimizer_state_estimate_bytes"]
    runs = len(config.get("methods", ["identity"])) * len(config.get("seeds", [42]))
    disk_free = shutil.disk_usage(output_dir).free
    checkpoint_total = checkpoint_bytes * int(config.get("epochs", 10)) * runs
    result = {
        **successful, "synthetic_preflight_only": True, "real_training_completed": False,
        "environment": environment_metadata(device), "attempts": attempts,
        "effective_batch_size": int(config.get("batch_size", 32)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes_estimate_per_epoch": checkpoint_bytes,
        "checkpoint_bytes_estimate_all_runs": checkpoint_total,
        "disk_free_bytes": disk_free, "disk_capacity_sufficient_for_checkpoints": disk_free > checkpoint_total * 1.2,
        "estimated_train_compute_seconds_per_epoch": successful["forward_backward_seconds"] * math.ceil(
            int(config.get("estimated_training_samples", 529)) / successful["microbatch_size"]),
        "estimate_limitations": "Synthetic warmed-up forward/backward only; excludes data loading, preprocessing, validation, optimizer steps and checkpoint I/O. Storage excludes data and metadata. Check Drive quota separately.",
        "batchnorm_note": "Gradient accumulation does not enlarge BatchNorm statistics; use the same microbatch across all arms.",
    }
    write_json(output_dir / "preflight.json", result)
    del model, images, targets, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
