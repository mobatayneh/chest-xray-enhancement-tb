from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import cxr_enhance.training as training
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cxr_enhance.checkpoints import load_checkpoint, sha256_file
from cxr_enhance.training import (
    ImageDataset,
    _run_epoch,
    _scheduler,
    build_model,
    load_selected_model,
    predict_rows,
    preflight,
    seed_everything,
    train_run,
)


@pytest.fixture
def synthetic_case(tmp_path: Path) -> tuple[dict, Path, dict[str, Path]]:
    root = tmp_path / "images"
    root.mkdir()
    generator = np.random.default_rng(2026)
    records = []
    for index in range(12):
        image = generator.integers(0, 256, (21, 29), dtype=np.uint8)
        filename = f"synthetic_{index}.png"
        Image.fromarray(image).save(root / filename)
        records.append({"dataset": "synthetic", "sample_id": f"id{index}", "path": filename,
                        "label": index % 2, "group_id": f"group{index}",
                        "split": "train" if index < 8 else "validation",
                        "sha256": sha256_file(root / filename)})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    config = {"model": "tiny", "synthetic": True, "weights": None, "device": "cpu",
              "epochs": 2, "batch_size": 4, "microbatch_size": 2, "workers": 0,
              "image_size": 32, "learning_rate": 0.01, "deterministic": True,
              "rotation_probability": 0.5, "rotation_degrees": 10, "amp": False,
              "methods": ["identity"], "seeds": [42]}
    return config, manifest, {"synthetic": root}


def test_epoch_boundary_resume_matches_uninterrupted(synthetic_case, tmp_path: Path, monkeypatch) -> None:
    config, manifest, roots = synthetic_case
    torch.set_num_threads(1)

    def stochastic_builder(settings):
        return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Dropout(0.3),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))

    monkeypatch.setattr(training, "build_model", stochastic_builder)
    complete_dir, resumed_dir = tmp_path / "complete", tmp_path / "resumed"
    complete = train_run(config, manifest, roots, complete_dir, "identity", 42)
    train_run(config, manifest, roots, resumed_dir, "identity", 42, stop_after_epoch=1)
    first_hash = sha256_file(resumed_dir / "checkpoints" / "epoch_0001.pt")
    resumed = train_run(config, manifest, roots, resumed_dir, "identity", 42, resume=True)
    assert first_hash == sha256_file(resumed_dir / "checkpoints" / "epoch_0001.pt")
    full_state = load_checkpoint(complete_dir / "checkpoints" / "epoch_0002.pt")
    resumed_state = load_checkpoint(resumed_dir / "checkpoints" / "epoch_0002.pt")
    for key, value in full_state["model"].items():
        torch.testing.assert_close(value, resumed_state["model"][key], rtol=0, atol=0)
    assert complete["best_epoch"] == resumed["best_epoch"]
    assert complete["best_validation_metrics"] == resumed["best_validation_metrics"]
    assert full_state["scheduler"] == resumed_state["scheduler"]
    for parameter_id, optimizer_state in full_state["optimizer"]["state"].items():
        for name, value in optimizer_state.items():
            torch.testing.assert_close(value, resumed_state["optimizer"]["state"][parameter_id][name], rtol=0, atol=0)
    for full_epoch, resumed_epoch in zip(full_state["history"], resumed_state["history"]):
        assert {key: value for key, value in full_epoch.items() if key != "seconds"} == {
            key: value for key, value in resumed_epoch.items() if key != "seconds"}
    model, metadata = load_selected_model(resumed_dir, "cpu")
    rows = pd.read_csv(manifest).query("split == 'validation'")
    predictions = predict_rows(model, rows, roots, config, "identity", "cpu")
    saved = pd.read_csv(resumed_dir / "validation_predictions.csv")
    np.testing.assert_allclose(predictions["probability"], saved["probability"], rtol=0, atol=1e-15)
    assert metadata["synthetic"] is True


def test_rejects_changed_config_data_and_accidental_restart(synthetic_case, tmp_path: Path) -> None:
    config, manifest, roots = synthetic_case
    torch.set_num_threads(1)
    run_dir = tmp_path / "run"
    train_run(config, manifest, roots, run_dir, "identity", 42, stop_after_epoch=1)
    with pytest.raises(FileExistsError, match="resume=True"):
        train_run(config, manifest, roots, run_dir, "identity", 42)
    with pytest.raises(ValueError, match="config_hash"):
        train_run({**config, "learning_rate": 0.02}, manifest, roots, run_dir, "identity", 42, resume=True)
    with pytest.raises(ValueError, match="seed"):
        train_run(config, manifest, roots, run_dir, "identity", 43, resume=True)
    with (roots["synthetic"] / "synthetic_0.png").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed since preparation"):
        train_run(config, manifest, roots, run_dir, "identity", 42, resume=True)


def test_accumulation_uses_actual_incomplete_window_size() -> None:
    torch.set_num_threads(1)
    images = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    baseline = nn.Linear(1, 1, bias=False)
    baseline.weight.data.fill_(0.1)
    accumulated = nn.Linear(1, 1, bias=False)
    accumulated.load_state_dict(baseline.state_dict())
    optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01)
    loss_sum = 0.0
    for start in (0, 4):
        optimizer.zero_grad()
        logits = baseline(images[start:start + 4]).reshape(-1)
        losses = nn.functional.binary_cross_entropy_with_logits(logits, labels[start:start + 4], reduction="none")
        loss_sum += losses.detach().sum().item()
        losses.mean().backward()
        optimizer.step()
    actual_loss, _, _ = _run_epoch(
        accumulated, DataLoader(TensorDataset(images, labels), batch_size=2), torch.device("cpu"),
        {"batch_size": 4, "microbatch_size": 2, "amp": False},
        torch.optim.SGD(accumulated.parameters(), lr=0.01), torch.amp.GradScaler("cuda", enabled=False))
    torch.testing.assert_close(accumulated.weight, baseline.weight, atol=1e-7, rtol=0)
    assert actual_loss == pytest.approx(loss_sum / 5)


def test_augmentation_is_sample_epoch_seed_deterministic(synthetic_case) -> None:
    config, manifest, roots = synthetic_case
    rows = pd.read_csv(manifest)
    config = {**config, "rotation_probability": 1.0}
    first = ImageDataset(rows, roots, config, "identity", seed=42, epoch=1, training=True)
    second = ImageDataset(rows, roots, config, "identity", seed=42, epoch=1, training=True)
    next_epoch = ImageDataset(rows, roots, config, "identity", seed=42, epoch=2, training=True)
    torch.testing.assert_close(first[0][0], second[0][0], atol=0, rtol=0)
    assert not torch.equal(first[0][0], next_epoch[0][0])


def test_preflight_is_explicitly_synthetic(synthetic_case, tmp_path: Path) -> None:
    config, _, _ = synthetic_case
    torch.set_num_threads(1)
    result = preflight(config, tmp_path / "preflight")
    assert result["microbatch_size"] == 1
    assert result["real_training_completed"] is False
    assert result["synthetic_preflight_only"] is True
    assert result["parameter_count"] > 0


def test_actual_efficientnet_s_forward_backward_without_download() -> None:
    torch.set_num_threads(1)
    seed_everything(42)
    model = build_model({"model": "efficientnet_v2_s", "weights": None})
    model.train()
    inputs = torch.randn(2, 3, 32, 32)
    outputs = model(inputs).reshape(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(outputs, torch.tensor([0.0, 1.0]))
    loss.backward()
    assert outputs.shape == (2,)
    assert torch.isfinite(loss)
    assert model.classifier[-1].weight.grad is not None
    assert torch.isfinite(model.classifier[-1].weight.grad).all()


def test_tiny_cannot_be_used_for_research() -> None:
    with pytest.raises(ValueError, match="synthetic"):
        build_model({"model": "tiny", "weights": None})


def test_training_is_blocked_after_suite_evaluation(synthetic_case, tmp_path: Path) -> None:
    config, manifest, roots = synthetic_case
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "evaluation_lock.json").write_text("{}")
    with pytest.raises(RuntimeError, match="external evaluation"):
        train_run(config, manifest, roots, suite / "runs" / "identity__seed_42", "identity", 42)


def test_prediction_rejects_changed_file(synthetic_case) -> None:
    config, manifest, roots = synthetic_case
    rows = pd.read_csv(manifest).query("split == 'validation'")
    model = build_model(config)
    with (roots["synthetic"] / rows.iloc[0]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed since preparation"):
        predict_rows(model, rows, roots, config, "identity", "cpu")


def test_resume_and_selection_reject_source_or_environment_changes(synthetic_case, tmp_path: Path, monkeypatch) -> None:
    config, manifest, roots = synthetic_case
    torch.set_num_threads(1)
    run_dir = tmp_path / "run"
    train_run(config, manifest, roots, run_dir, "identity", 42, stop_after_epoch=1)
    with monkeypatch.context() as patch:
        patch.setattr(training, "source_hash", lambda: "changed")
        with pytest.raises(ValueError, match="source_hash"):
            train_run(config, manifest, roots, run_dir, "identity", 42, resume=True)
        with pytest.raises(ValueError, match="Source code changed"):
            load_selected_model(run_dir, "cpu")
    actual_environment = training.environment_metadata(torch.device("cpu"))
    monkeypatch.setattr(training, "environment_metadata", lambda device: {**actual_environment, "python": "changed"})
    with pytest.raises(ValueError, match="environment_hash"):
        train_run(config, manifest, roots, run_dir, "identity", 42, resume=True)


def test_resume_constructs_model_without_downloading_weights(synthetic_case, tmp_path: Path, monkeypatch) -> None:
    config, manifest, roots = synthetic_case
    torch.set_num_threads(1)
    config = {**config, "weights": "IMAGENET1K_V1"}
    requested_weights = []

    def fake_builder(settings):
        requested_weights.append(settings.get("weights"))
        return build_model({**settings, "weights": None})

    monkeypatch.setattr(training, "build_model", fake_builder)
    run_dir = tmp_path / "run"
    train_run(config, manifest, roots, run_dir, "identity", 42, stop_after_epoch=1)
    train_run(config, manifest, roots, run_dir, "identity", 42, resume=True)
    assert requested_weights == ["IMAGENET1K_V1", None]


def test_legacy_plateau_matches_accuracy_schedule() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = _scheduler(optimizer, {"scheduler": "plateau"})
    assert scheduler.mode == "max"
    assert scheduler.factor == 0.5
    assert scheduler.patience == 2
    for accuracy in (0.8, 0.7, 0.7, 0.7):
        scheduler.step(accuracy)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_legacy_selection_uses_accuracy(synthetic_case, tmp_path: Path, monkeypatch) -> None:
    config, manifest, roots = synthetic_case
    torch.set_num_threads(1)
    config = {**config, "legacy_geometry": True, "selection_metric": "accuracy", "scheduler": "plateau"}
    validation_probabilities = iter(([0.6, 0.8, 0.7, 0.9], [0.1, 0.8, 0.7, 0.6]))

    def fake_epoch(model, loader, device, settings, optimizer=None, scaler=None):
        probabilities = [0.1, 0.9, 0.1, 0.9] if optimizer else next(validation_probabilities)
        return 1.0, [0, 1, 0, 1], probabilities

    monkeypatch.setattr(training, "_run_epoch", fake_epoch)
    metadata = train_run(config, manifest, roots, tmp_path / "legacy", "identity", 42)
    assert metadata["best_epoch"] == 2
    assert metadata["best_validation_metrics"]["accuracy"] == 0.75
    assert metadata["best_validation_metrics"]["roc_auc"] == 0.75
