import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import torch
import cxr_enhance.training as training

from cxr_enhance.evaluation import evaluate_suite, export_results, initialize_suite, verify_export
from cxr_enhance.smoke import run_smoke
from cxr_enhance.utils import sha256_file


@pytest.fixture(scope="module")
def completed_smoke(tmp_path_factory):
    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("end-to-end") / "synthetic"
    status = run_smoke(directory)
    assert status["synthetic_only"] is True
    assert status["real_training_completed"] is False
    return directory


def rewrite_archive(source: Path, destination: Path, changed_name: str, payload: bytes,
                    refresh_checksums: bool = False) -> None:
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[changed_name] = payload
    if refresh_checksums:
        members["SHA256SUMS.txt"] = "".join(
            f"{hashlib.sha256(value).hexdigest()}  {name}\n"
            for name, value in sorted(members.items()) if name != "SHA256SUMS.txt"
        ).encode()
    with zipfile.ZipFile(destination, "x") as archive:
        for name, value in members.items():
            archive.writestr(name, value)


def test_all_arms_export_recompute_and_repeat_evaluation(completed_smoke):
    directory = completed_smoke
    archive = directory / "synthetic_results_for_review.zip"
    verified = verify_export(archive)
    assert verified["verified_runs"] == 5
    assert verified["verified_files"] > 50
    suite = directory / "suite"
    roots = {name: directory / "synthetic_data" / name for name in ("shenzhen", "montgomery")}
    completion = suite / "evaluation" / "completion.json"
    checksum = sha256_file(completion)
    assert evaluate_suite(suite, directory / "prepared" / "manifest.csv", roots) == completion.parent
    assert sha256_file(completion) == checksum
    with pytest.raises(FileExistsError, match="replace"):
        export_results(suite, archive)
    with zipfile.ZipFile(archive) as package:
        assert not any(name.endswith((".pt", ".pth")) or "CXR_png" in name for name in package.namelist())
        assert json.loads(package.read("metrics.json"))["synthetic"] is True


def test_suite_rejects_configuration_change(completed_smoke):
    suite = completed_smoke / "suite"
    config = json.loads((suite / "suite.json").read_text())["config"]
    with pytest.raises(ValueError, match="configuration"):
        initialize_suite(suite, {**config, "epochs": 20}, completed_smoke / "prepared" / "manifest.csv")


def test_evaluation_rejects_incomplete_training(completed_smoke):
    suite = completed_smoke / "suite"
    path = suite / "runs" / "identity__seed_42" / "run_metadata.json"
    original = path.read_bytes()
    record = json.loads(original)
    record["completed_epochs"] = 1
    path.write_text(json.dumps(record))
    try:
        with pytest.raises(ValueError, match="incomplete"):
            evaluate_suite(suite, completed_smoke / "prepared" / "manifest.csv", {})
    finally:
        path.write_bytes(original)


def test_checksum_detects_modified_bytes(completed_smoke, tmp_path):
    archive = completed_smoke / "synthetic_results_for_review.zip"
    changed = tmp_path / "corrupt.zip"
    rewrite_archive(archive, changed, "evaluation_report.md", b"modified")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_export(changed)


def test_recompute_detects_wrong_metrics_even_with_updated_checksums(completed_smoke, tmp_path):
    archive = completed_smoke / "synthetic_results_for_review.zip"
    with zipfile.ZipFile(archive) as package:
        comparison = pd.read_csv(io.BytesIO(package.read("model_comparison.csv")))
    comparison.loc[0, "roc_auc"] = -1
    changed = tmp_path / "false-metric.zip"
    rewrite_archive(archive, changed, "model_comparison.csv", comparison.to_csv(index=False).encode(), True)
    with pytest.raises(ValueError, match="Recomputed metric"):
        verify_export(changed)


def test_recompute_detects_saved_run_metric_change(completed_smoke, tmp_path):
    archive = completed_smoke / "synthetic_results_for_review.zip"
    name = "runs/identity__seed_42/metrics.json"
    with zipfile.ZipFile(archive) as package:
        metrics = json.loads(package.read(name))
    metrics["per_class"]["normal"]["precision"] = -1
    changed = tmp_path / "false-class-metric.zip"
    rewrite_archive(archive, changed, name, json.dumps(metrics).encode(), True)
    with pytest.raises(ValueError, match="Saved run metrics"):
        verify_export(changed)


def test_recompute_requires_complete_experiment_matrix(completed_smoke, tmp_path):
    archive = completed_smoke / "synthetic_results_for_review.zip"
    with zipfile.ZipFile(archive) as package:
        comparison = pd.read_csv(io.BytesIO(package.read("model_comparison.csv")))
    changed = tmp_path / "missing-run.zip"
    rewrite_archive(archive, changed, "model_comparison.csv", comparison.iloc[1:].to_csv(index=False).encode(), True)
    with pytest.raises(ValueError, match="complete configured experiment matrix"):
        verify_export(changed)


def test_evaluation_reapplies_training_determinism(completed_smoke, tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    shutil.copytree(completed_smoke / "suite", suite)
    (suite / "evaluation" / "completion.json").unlink()
    seeds = []
    original_seed = training.seed_everything

    def record_seed(seed, deterministic):
        seeds.append((seed, deterministic))
        original_seed(seed, deterministic)

    monkeypatch.setattr(training, "seed_everything", record_seed)
    roots = {name: completed_smoke / "synthetic_data" / name for name in ("shenzhen", "montgomery")}
    evaluate_suite(suite, completed_smoke / "prepared" / "manifest.csv", roots)
    assert seeds == [(42, True)] * 5


def test_interrupted_export_does_not_publish_partial_archive(completed_smoke, tmp_path, monkeypatch):
    destination = tmp_path / "review.zip"
    original_write = zipfile.ZipFile.write
    calls = 0

    def interrupted_write(archive, *arguments, **keywords):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Simulated interrupted storage write")
        return original_write(archive, *arguments, **keywords)

    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile, "write", interrupted_write)
        with pytest.raises(OSError, match="interrupted storage"):
            export_results(completed_smoke / "suite", destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".pending-export-*.zip"))
    export_results(completed_smoke / "suite", destination)
    assert verify_export(destination)["verified_runs"] == 5


@pytest.mark.parametrize("field,replacement", [("label", 1), ("group_id", "wrong-group"),
                                               ("sample_id", "wrong-case")])
def test_predictions_must_match_held_out_manifest(completed_smoke, tmp_path, field, replacement):
    source = completed_smoke / "synthetic_results_for_review.zip"
    with zipfile.ZipFile(source) as archive:
        manifest = pd.read_csv(io.BytesIO(archive.read("split_manifest.csv")))
    first_test = manifest.index[manifest["split"] == "test"][0]
    if field == "label":
        replacement = 1 - int(manifest.loc[first_test, "label"])
    manifest.loc[first_test, field] = replacement
    destination = tmp_path / f"incorrect-{field}.zip"
    rewrite_archive(source, destination, "split_manifest.csv", manifest.to_csv(index=False).encode(), True)
    with pytest.raises(ValueError, match="held-out manifest"):
        verify_export(destination)


def test_per_run_predictions_must_match_aggregate(completed_smoke, tmp_path):
    source = completed_smoke / "synthetic_results_for_review.zip"
    name = "runs/identity__seed_42/predictions.csv"
    with zipfile.ZipFile(source) as archive:
        predictions = pd.read_csv(io.BytesIO(archive.read(name)))
    predictions.loc[0, "probability"] = 0.123456789
    destination = tmp_path / "incorrect-per-run.zip"
    rewrite_archive(source, destination, name, predictions.to_csv(index=False).encode(), True)
    with pytest.raises(ValueError, match="Per-run predictions"):
        verify_export(destination)


def test_export_uses_source_matched_versioned_software_validation(completed_smoke, tmp_path):
    suite = tmp_path / "suite"
    shutil.copytree(completed_smoke / "suite", suite)
    source_hash = json.loads((suite / "suite.json").read_text())["source_sha256"]
    historical = suite / "software_validation.json"
    historical.write_text(json.dumps({"stdout": "Historical tests"}))
    record = {"source_hash": source_hash, "package_id": "current-package",
              "kind": "software_tests_only", "exit_code": 0, "stdout": "Current tests passed"}
    (suite / "software_validation_current-package.json").write_text(json.dumps(record))
    (suite / "software_validation_old-package.json").write_text(json.dumps({**record, "source_hash": "old"}))
    original = historical.read_bytes()
    destination = export_results(suite, tmp_path / "review.zip")
    with zipfile.ZipFile(destination) as archive:
        assert json.loads(archive.read("provenance/software_validation.json")) == record
    assert historical.read_bytes() == original
    assert verify_export(destination)["verified_runs"] == 5


@pytest.mark.parametrize("changed", [{"source_hash": "old"}, {"exit_code": 1}])
def test_export_rejects_unmatched_or_failed_versioned_validation(completed_smoke, tmp_path, changed):
    suite = tmp_path / "suite"
    shutil.copytree(completed_smoke / "suite", suite)
    source_hash = json.loads((suite / "suite.json").read_text())["source_sha256"]
    record = {"source_hash": source_hash, "kind": "software_tests_only", "exit_code": 0, **changed}
    (suite / "software_validation.json").write_text(json.dumps({"stdout": "Historical tests"}))
    (suite / "software_validation_mismatch.json").write_text(json.dumps(record))
    destination = tmp_path / "review.zip"
    with pytest.raises(ValueError, match="software-validation record matches"):
        export_results(suite, destination)
    assert not destination.exists()
