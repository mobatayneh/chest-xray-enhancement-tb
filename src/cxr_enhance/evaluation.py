"""Freeze complete experiments before external evaluation and export inspectable results."""

import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import torch

from .metrics import classification_metrics, paired_bootstrap
from .checkpoints import _publish_immutable, local_lock
from .data import validate_prepared_manifest
from .plots import plot_comparison, plot_evaluation
from .utils import canonical_hash, environment_info, sha256_file, source_hash, write_json


def run_name(method: str, seed: int) -> str:
    return f"{method}__seed_{seed}"


def initialize_suite(directory: Path, config: dict, manifest: Path) -> dict:
    validate_prepared_manifest(manifest, config)
    directory.mkdir(parents=True, exist_ok=True)
    expected = {"config": config, "manifest_sha256": sha256_file(manifest), "source_sha256": source_hash(),
                "data_validation_sha256": sha256_file(manifest.parent / "data_validation.json"),
                "runs": [run_name(method, seed) for method in config["methods"] for seed in config["seeds"]]}
    path = directory / "suite.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("Suite configuration, manifest, or source differs. Use a new suite directory.")
    else:
        write_json(path, expected)
    return expected


def evaluate_suite(directory: Path, manifest: Path, data_roots: dict[str, Path]) -> Path:
    from .training import load_selected_model, predict_rows, seed_everything
    from .checkpoints import canonical_hash as checkpoint_hash, source_hash as checkpoint_source_hash

    suite = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
    initialize_suite(directory, suite["config"], manifest)
    configuration = suite["config"]
    run_records = {}
    training_environments = set()
    for name in suite["runs"]:
        run_directory = directory / "runs" / name
        record = json.loads((run_directory / "run_metadata.json").read_text(encoding="utf-8"))
        if record["completed_epochs"] != configuration["epochs"]:
            raise ValueError(f"Training is incomplete: {name}")
        saved_config = json.loads((run_directory / "experiment_config.json").read_text(encoding="utf-8"))
        contract = record["contract"]
        if saved_config != configuration or contract["config_hash"] != checkpoint_hash(configuration):
            raise ValueError(f"Run configuration differs from the suite: {name}")
        if contract["manifest_hash"] != sha256_file(manifest):
            raise ValueError(f"Run used a different manifest: {name}")
        if contract["prepared_report_hash"] != suite["data_validation_sha256"]:
            raise ValueError(f"Run used different preparation evidence: {name}")
        if contract["source_hash"] != checkpoint_source_hash():
            raise ValueError(f"Source changed after training: {name}")
        if name != run_name(contract["method"], contract["seed"]):
            raise ValueError(f"Run identity mismatch: {name}")
        training_environments.add(contract["environment_hash"])
        checkpoint = Path(record["best_checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = run_directory / checkpoint
        digest = sha256_file(checkpoint)
        if digest != record["best_checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint changed: {name}")
        run_records[name] = {"checkpoint_sha256": digest, "best_epoch": record["best_epoch"],
                             "checkpoint_location": f"runs/{name}/{record['best_checkpoint']}",
                             "contract": contract, "selection_rule": record["selection_rule"]}
    if len(training_environments) != 1:
        raise ValueError("Training environments differ across the controlled comparison")
    frozen = {"suite_sha256": canonical_hash(suite), "selected_checkpoints": run_records}
    lock_path = directory / "evaluation_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != frozen:
            raise ValueError("External evaluation is frozen to different models or configuration")
    else:
        write_json(lock_path, frozen)
    output = directory / "evaluation"
    output.mkdir(exist_ok=True)
    completion = output / "completion.json"
    if completion.exists():
        saved = json.loads(completion.read_text(encoding="utf-8"))
        if saved["freeze_sha256"] != canonical_hash(frozen):
            raise ValueError("Completed evaluation does not match the current frozen experiment")
        for name, digest in saved["files"].items():
            if sha256_file(output / name) != digest:
                raise ValueError(f"Evaluation artifact changed: {name}")
        return output
    frame = pd.read_csv(manifest, dtype={"sample_id": str, "group_id": str}, keep_default_na=False)
    test_rows = frame[frame.split == "test"].reset_index(drop=True)
    if test_rows.empty or set(test_rows.dataset) != {"montgomery"}:
        raise ValueError("External evaluation requires the reserved Montgomery cases")
    prediction_frames = []
    comparisons = []
    histories = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for method in configuration["methods"]:
        for seed in configuration["seeds"]:
            seed_everything(seed, configuration["deterministic"])
            name = run_name(method, seed)
            run_directory = directory / "runs" / name
            model, _metadata = load_selected_model(run_directory, device)
            predicted = predict_rows(model, test_rows, data_roots, configuration, method, device)
            predicted["method"] = method
            predicted["seed"] = seed
            predicted["run_id"] = name
            predicted["split"] = "test"
            metrics = classification_metrics(predicted.label, predicted.probability, configuration["threshold"])
            prediction_frames.append(predicted)
            comparisons.append({"run_id": name, "method": method, "seed": seed,
                                **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))}})
            run_output = output / "runs" / name
            run_output.mkdir(parents=True, exist_ok=True)
            write_json(run_output / "metrics.json", metrics, overwrite=True)
            predicted.to_csv(run_output / "predictions.csv", index=False)
            history = pd.read_csv(run_directory / "training_history.csv")
            history["run_id"] = name
            histories.append(history)
            plot_evaluation(predicted, history, run_output, synthetic=configuration["synthetic"])
            validation = pd.read_csv(run_directory / "validation_predictions.csv",
                                     dtype={"sample_id": str, "group_id": str}, keep_default_na=False)
            validation.to_csv(run_output / "validation_predictions.csv", index=False)
            for artifact in ("environment.json", "run_metadata.json", "checkpoint_metadata.json"):
                value = json.loads((run_directory / artifact).read_text(encoding="utf-8"))
                write_json(run_output / artifact, value, overwrite=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    comparison = pd.DataFrame(comparisons)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output / "training_history.csv", index=False)
    if "identity" in configuration["methods"]:
        uncertainty = paired_bootstrap(predictions, configuration["bootstrap_samples"], configuration["split_seed"])
    else:
        uncertainty = {"n_cases": test_rows.sample_id.nunique(), "n_groups": test_rows.group_id.nunique(),
                       "methods": {}, "status": "Not computed: historical adapter has no identity comparator"}
    summary = comparison.groupby("method").roc_auc.agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(output / "seed_summary.csv", index=False)
    write_json(output / "metrics.json", {"synthetic": configuration["synthetic"], "paired_uncertainty": uncertainty,
                                         "runs": comparisons}, overwrite=True)
    write_json(output / "experiment_config.json", configuration, overwrite=True)
    write_json(output / "environment.json", environment_info(), overwrite=True)
    write_json(output / "checkpoint_metadata.json", frozen, overwrite=True)
    safe_columns = [column for column in frame if column != "path"]
    frame[safe_columns].to_csv(output / "split_manifest.csv", index=False)
    data_report = json.loads((manifest.parent / "data_validation.json").read_text(encoding="utf-8"))
    data_report.pop("roots", None)
    write_json(output / "data_validation.json", data_report, overwrite=True)
    plot_comparison(comparison, output / "model_comparison.png", configuration["synthetic"])
    primary = uncertainty["methods"].get("corrected_unsharp", {})
    report = ["# External Evaluation Report", "",
              "SYNTHETIC SOFTWARE CHECK ONLY. These values are not research findings." if configuration["synthetic"]
              else "Computed from frozen checkpoints on reserved Montgomery cases. Independent review is pending.",
              "", f"Cases: {uncertainty['n_cases']}. Groups: {uncertainty['n_groups']}.",
              f"Training seeds: {configuration['seeds']}. Threshold: {configuration['threshold']}.",
              "", "Primary contrast: corrected thresholded unsharp minus no enhancement.",
              f"Mean ROC-AUC difference: {primary.get('difference_vs_identity', 'not run')}.",
              f"Paired 95% interval: {primary.get('paired_difference_ci95', 'not run')}.",
              "", "Bootstrap resamples groups jointly across methods and seeds. Seed standard deviation is separate.",
              "This does not include uncertainty from alternative training cohorts. Secondary comparisons are exploratory.",
              "No clinical efficacy, universal enhancement benefit, or patient-independent validation is established.",
              "Shenzhen was repeatedly used in historical development. Montgomery has a different acquisition population.",
              "If patient metadata is unavailable, image/case groups do not prove independent patients.",
              "No threshold, preprocessing, or architecture may be revised using these external scores.",
              "", "Large checkpoints remain in the original suite runs; checksums identify the evaluated models."]
    (output / "evaluation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    files = {path.relative_to(output).as_posix(): sha256_file(path) for path in sorted(output.rglob("*"))
             if path.is_file() and path.name != "completion.json"}
    write_json(completion, {"freeze_sha256": canonical_hash(frozen), "files": files})
    return output


def export_results(directory: Path, destination: Path) -> Path:
    with local_lock(destination, "export"):
        return _export_results(directory, destination)


def _export_results(directory: Path, destination: Path) -> Path:
    evaluation = directory / "evaluation"
    completion = json.loads((evaluation / "completion.json").read_text(encoding="utf-8"))
    for name, digest in completion["files"].items():
        if sha256_file(evaluation / name) != digest:
            raise ValueError(f"Integrity failure: {name}")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing results package: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = dict(completion["files"])
    files["completion.json"] = sha256_file(evaluation / "completion.json")
    provenance = {"provenance/" + name: directory / name
                  for name in ("suite.json", "evaluation_lock.json", "preflight.json")
                  if (directory / name).exists()}
    validation = _software_validation_artifact(directory)
    if validation is not None:
        provenance["provenance/software_validation.json"] = validation
    complete_checksums = {**files, **{name: sha256_file(path) for name, path in provenance.items()}}
    checksum_text = "".join(f"{digest}  {name}\n" for name, digest in sorted(complete_checksums.items()))
    descriptor, pending_name = tempfile.mkstemp(prefix=".pending-export-", suffix=".zip", dir=destination.parent)
    os.close(descriptor)
    pending = Path(pending_name)
    try:
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(files):
                archive.write(evaluation / name, name)
            archive.writestr("SHA256SUMS.txt", checksum_text)
            for name, path in provenance.items():
                archive.write(path, name)
        verify_export(pending)
        _publish_immutable(pending, destination)
    finally:
        pending.unlink(missing_ok=True)
    return destination


def _software_validation_artifact(directory: Path) -> Path | None:
    versioned = sorted(directory.glob("software_validation_*.json"))
    if versioned:
        suite = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
        matching = []
        for path in versioned:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record.get("source_hash") == suite["source_sha256"]
                    and record.get("kind") == "software_tests_only" and record.get("exit_code") == 0):
                matching.append(path)
        if not matching:
            raise ValueError("No successful software-validation record matches the suite source hash")
        return matching[-1]
    original = directory / "software_validation.json"
    return original if original.is_file() else None


def verify_export(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        import hashlib

        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        if sum(item.file_size for item in archive.infolist()) > 512 * 1024**2:
            raise ValueError("Review package exceeds the compact-artifact size limit")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe ZIP member")
        checksums = archive.read("SHA256SUMS.txt").decode().splitlines()
        checksum_names = [line.split("  ", 1)[1] for line in checksums]
        if len(checksum_names) != len(set(checksum_names)) or set(checksum_names) != set(names) - {"SHA256SUMS.txt"}:
            raise ValueError("Checksum inventory does not cover exactly the archive members")
        for line in checksums:
            digest, name = line.split("  ", 1)
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {name}")
        import io

        identity_types = {"sample_id": str, "group_id": str}
        scores = pd.read_csv(io.BytesIO(archive.read("predictions.csv")), dtype=identity_types, keep_default_na=False)
        manifest = pd.read_csv(io.BytesIO(archive.read("split_manifest.csv")),
                               dtype=identity_types, keep_default_na=False)
        comparison = pd.read_csv(io.BytesIO(archive.read("model_comparison.csv")))
        config = json.loads(archive.read("experiment_config.json"))
        aggregate = json.loads(archive.read("metrics.json"))
        expected_runs = {run_name(method, seed) for method in config["methods"] for seed in config["seeds"]}
        if set(comparison.run_id) != expected_runs:
            raise ValueError("Results do not contain the complete configured experiment matrix")
        if len(aggregate["runs"]) != len(expected_runs):
            raise ValueError("Aggregate metrics do not contain every run")
        aggregate_runs = {row["run_id"]: row for row in aggregate["runs"]}
        test_rows = manifest[manifest["split"] == "test"]
        if (test_rows.empty or set(test_rows["dataset"]) != {"montgomery"}
                or test_rows["sample_id"].duplicated().any()):
            raise ValueError("Exported manifest must identify unique held-out Montgomery cases")
        reference_cases = test_rows.set_index("sample_id")[["group_id", "label"]].sort_index()
        if comparison.run_id.duplicated().any() or scores.duplicated(["run_id", "sample_id"]).any():
            raise ValueError("Duplicate run or case predictions")
        if set(comparison.run_id) != set(scores.run_id):
            raise ValueError("Comparison and prediction runs disagree")
        for row in comparison.to_dict("records"):
            selected = scores[scores.run_id == row["run_id"]]
            if row["run_id"] != run_name(row["method"], row["seed"]):
                raise ValueError("Comparison run identity disagrees with method/seed")
            if (set(selected.method) != {row["method"]} or set(selected.seed) != {row["seed"]}
                    or set(selected.dataset) != {"montgomery"} or set(selected.split) != {"test"}):
                raise ValueError("Prediction run identity or external split is invalid")
            case_record = selected.set_index("sample_id")[["group_id", "label"]].sort_index()
            if not case_record.equals(reference_cases):
                raise ValueError("Predictions do not match held-out manifest cases, labels and groups")
            saved_predictions = pd.read_csv(io.BytesIO(archive.read(f"runs/{row['run_id']}/predictions.csv")),
                                            dtype=identity_types, keep_default_na=False)
            if set(saved_predictions.columns) != set(selected.columns):
                raise ValueError("Per-run and aggregate prediction schemas disagree")
            saved_predictions = saved_predictions[selected.columns].sort_values("sample_id").reset_index(drop=True)
            if not saved_predictions.equals(selected.sort_values("sample_id").reset_index(drop=True)):
                raise ValueError("Per-run predictions disagree with aggregate predictions")
            if row["threshold"] != config["threshold"]:
                raise ValueError("Saved threshold differs from experiment configuration")
            calculated = classification_metrics(selected.label, selected.probability, row["threshold"])
            for metric, value in calculated.items():
                if isinstance(value, (dict, list)):
                    continue
                if not math.isfinite(row[metric]) or abs(calculated[metric] - row[metric]) > 1e-10:
                    raise ValueError(f"Recomputed metric differs: {row['run_id']} {metric}")
                if not _metrics_equal(aggregate_runs[row["run_id"]][metric], value):
                    raise ValueError(f"Aggregate metric differs: {row['run_id']} {metric}")
            saved = json.loads(archive.read(f"runs/{row['run_id']}/metrics.json"))
            if not _metrics_equal(saved, calculated):
                raise ValueError("Saved run metrics disagree with predictions")
            expected_predictions = (selected.probability.to_numpy() >= row["threshold"]).astype(int)
            if not (selected.prediction.to_numpy() == expected_predictions).all():
                raise ValueError("Saved predictions disagree with probabilities and threshold")
    return {"verified_files": len(checksums), "verified_runs": len(comparison), "archive_sha256": sha256_file(path)}


def _metrics_equal(saved: object, calculated: object) -> bool:
    if isinstance(calculated, dict):
        return (isinstance(saved, dict) and saved.keys() == calculated.keys()
                and all(_metrics_equal(saved[key], value) for key, value in calculated.items()))
    if isinstance(calculated, list):
        return (isinstance(saved, list) and len(saved) == len(calculated)
                and all(_metrics_equal(first, second) for first, second in zip(saved, calculated)))
    if isinstance(calculated, (int, float)):
        return isinstance(saved, (int, float)) and abs(saved - calculated) <= 1e-10
    return saved == calculated
