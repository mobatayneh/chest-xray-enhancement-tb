"""Read-only independent verification of the returned October 2026 experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value: object, compact: bool = False) -> str:
    options = {"separators": (",", ":")} if compact else {}
    return digest(json.dumps(value, sort_keys=True, allow_nan=False, **options).encode())


def independent_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    labels = frame.label.to_numpy(dtype=int)
    probabilities = frame.probability.to_numpy(dtype=float)
    assert set(labels) == {0, 1}
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    classified = probabilities >= threshold
    assert np.array_equal(classified.astype(int), frame.prediction.to_numpy())
    confusion = metrics.confusion_matrix(labels, classified, labels=[0, 1])
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        labels, classified, labels=[0, 1], zero_division=0
    )
    return {
        "n": len(labels), "threshold": threshold,
        "accuracy": metrics.accuracy_score(labels, classified),
        "balanced_accuracy": metrics.balanced_accuracy_score(labels, classified),
        "roc_auc": metrics.roc_auc_score(labels, probabilities),
        "average_precision": metrics.average_precision_score(labels, probabilities),
        "precision": precision[1], "recall": recall[1], "f1": fscore[1],
        "macro_f1": metrics.f1_score(labels, classified, average="macro"),
        "weighted_f1": metrics.f1_score(labels, classified, average="weighted"),
        "sensitivity": confusion[1, 1] / confusion[1].sum(),
        "specificity": confusion[0, 0] / confusion[0].sum(),
        "brier": metrics.brier_score_loss(labels, probabilities),
        "log_loss": metrics.log_loss(labels, probabilities, labels=[0, 1]),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            name: {"precision": precision[index], "recall": recall[index],
                   "f1": fscore[index], "support": int(support[index])}
            for index, name in enumerate(("normal", "tb_consistent"))
        },
    }


def compare(actual: object, expected: object, context: str, differences: list) -> None:
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                differences.append({"context": f"{context}.{key}", "error": "missing"})
            else:
                compare(actual[key], value, f"{context}.{key}", differences)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            differences.append({"context": context, "error": "length mismatch"})
        else:
            for index, value in enumerate(expected):
                compare(actual[index], value, f"{context}[{index}]", differences)
    elif isinstance(expected, (int, float, np.integer, np.floating)) and not isinstance(expected, bool):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
            differences.append({"context": context, "actual": float(actual), "expected": float(expected)})
    elif actual != expected:
        differences.append({"context": context, "actual": actual, "expected": expected})


def independent_bootstrap(predictions: pd.DataFrame, replicates: int, seed: int) -> dict:
    reference = predictions.drop_duplicates("sample_id").sort_values("sample_id").set_index("sample_id")
    assert reference.group_id.nunique() == len(reference)
    group_to_sample = dict(zip(reference.group_id, reference.index))
    groups = sorted(group_to_sample)
    negative = [group for group in groups if reference.loc[group_to_sample[group], "label"] == 0]
    positive = [group for group in groups if reference.loc[group_to_sample[group], "label"] == 1]
    negative_lookup = {group: index for index, group in enumerate(negative)}
    positive_lookup = {group: index for index, group in enumerate(positive)}
    negative_counts = np.zeros((replicates, len(negative)), dtype=int)
    positive_counts = np.zeros((replicates, len(positive)), dtype=int)
    generator = np.random.default_rng(seed)
    for replicate in range(replicates):
        drawn_negative = generator.choice(negative, len(negative))
        drawn_positive = generator.choice(positive, len(positive))
        negative_counts[replicate] = np.bincount(
            [negative_lookup[group] for group in drawn_negative], minlength=len(negative)
        )
        positive_counts[replicate] = np.bincount(
            [positive_lookup[group] for group in drawn_positive], minlength=len(positive)
        )
    distributions = {}
    points = {}
    for method, method_frame in predictions.groupby("method"):
        seed_distributions = []
        seed_points = []
        for _, seed_frame in method_frame.groupby("seed"):
            lookup = seed_frame.set_index("group_id").probability
            negative_scores = lookup.loc[negative].to_numpy()
            positive_scores = lookup.loc[positive].to_numpy()
            ranking = (positive_scores[:, None] > negative_scores[None, :]).astype(float)
            ranking += 0.5 * (positive_scores[:, None] == negative_scores[None, :])
            values = np.sum((positive_counts @ ranking) * negative_counts, axis=1)
            values /= len(negative) * len(positive)
            seed_distributions.append(values)
            seed_points.append(float(ranking.mean()))
        distributions[method] = np.mean(seed_distributions, axis=0)
        points[method] = float(np.mean(seed_points))
    return {
        method: {
            "mean_roc_auc": points[method],
            "roc_auc_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "difference_vs_identity": points[method] - points["identity"],
            "paired_difference_ci95": np.quantile(values - distributions["identity"], [0.025, 0.975]).tolist(),
        }
        for method, values in distributions.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    archive_bytes = arguments.archive.read_bytes()
    differences = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in name
            assert not re.match(r"^[A-Za-z]:", name)
        checksums = {}
        for line in archive.read("SHA256SUMS.txt").decode().splitlines():
            checksum, name = line.split("  ", 1)
            assert name not in checksums and re.fullmatch(r"[0-9a-f]{64}", checksum)
            checksums[name] = checksum
        assert set(checksums) == set(names) - {"SHA256SUMS.txt"}
        for name, checksum in checksums.items():
            assert digest(archive.read(name)) == checksum, name

        def read_json(name: str) -> dict:
            return json.loads(archive.read(name))

        def read_csv(name: str) -> pd.DataFrame:
            return pd.read_csv(io.BytesIO(archive.read(name)), keep_default_na=False)

        configuration = read_json("experiment_config.json")
        suite = read_json("provenance/suite.json")
        frozen = read_json("checkpoint_metadata.json")
        complete = read_json("completion.json")
        data_validation = read_json("data_validation.json")
        aggregate = read_json("metrics.json")
        manifest = read_csv("split_manifest.csv")
        history = read_csv("training_history.csv")
        predictions = read_csv("predictions.csv")
        comparison = read_csv("model_comparison.csv")
        summary = read_csv("seed_summary.csv")
        expected_runs = {f"{method}__seed_{seed}" for method in configuration["methods"] for seed in configuration["seeds"]}
        assert not configuration["synthetic"] and not aggregate["synthetic"]
        assert suite["config"] == configuration
        assert set(suite["runs"]) == expected_runs == set(comparison.run_id) == set(predictions.run_id)
        assert len(comparison) == len(aggregate["runs"]) == len(expected_runs) == 15
        assert set(history.run_id) == expected_runs == set(frozen["selected_checkpoints"])
        assert frozen == read_json("provenance/evaluation_lock.json")
        assert frozen["suite_sha256"] == canonical(suite)
        assert complete["freeze_sha256"] == canonical(frozen)
        for name, checksum in complete["files"].items():
            assert checksums[name] == checksum
        assert set(complete["files"]) == {
            name for name in names if name not in {"SHA256SUMS.txt", "completion.json"} and not name.startswith("provenance/")
        }
        assert not manifest[["dataset", "sample_id"]].duplicated().any()
        assert not manifest[["sha256"]].duplicated().any()
        assert not manifest[["pixel_sha256"]].duplicated().any()
        assert manifest.groupby("group_id").split.nunique().max() == 1
        assert manifest.groupby("sample_id").label.nunique().max() == 1
        assert (manifest.sample_id.str.rsplit("_", n=1).str[-1].astype(int) == manifest.label).all()
        assert (manifest.loc[manifest.split == "test", "dataset"] == "montgomery").all()
        assert (manifest.loc[manifest.split != "test", "dataset"] == "shenzhen").all()
        counts = {split: {str(int(label)): int(count) for label, count in frame.label.value_counts().sort_index().items()}
                  for split, frame in manifest.groupby("split")}
        assert counts == data_validation["counts"]
        assert len(manifest) == 800 and len(predictions) == 2070
        shenzhen = manifest.loc[manifest.dataset == "shenzhen"].reset_index(drop=True)
        _, validation_indices = train_test_split(
            np.arange(len(shenzhen)), test_size=0.2, random_state=configuration["split_seed"], stratify=shenzhen.label
        )
        assert set(shenzhen.loc[validation_indices, "sample_id"]) == set(shenzhen.loc[shenzhen.split == "validation", "sample_id"])
        assert not predictions[["run_id", "sample_id"]].duplicated().any()
        assert set(predictions.split) == {"test"}
        assert predictions.groupby("sample_id")[["label", "group_id"]].nunique().max().max() == 1
        source_files = {path.name: digest(path.read_bytes()) for path in sorted(arguments.source.glob("*.py"))}
        source_digest = hashlib.sha256()
        for path in sorted(arguments.source.rglob("*.py")):
            source_digest.update(path.relative_to(arguments.source).as_posix().encode())
            source_digest.update(path.read_bytes())
        assert canonical(source_files) == suite["source_sha256"]
        run_results = []
        checkpoint_bytes = 0
        all_checkpoint_hashes = []
        for run_id in sorted(expected_runs):
            metadata = read_json(f"runs/{run_id}/run_metadata.json")
            checkpoints = read_json(f"runs/{run_id}/checkpoint_metadata.json")
            environment = read_json(f"runs/{run_id}/environment.json")
            selected = frozen["selected_checkpoints"][run_id]
            contract = metadata["contract"]
            assert run_id == f"{metadata['method']}__seed_{metadata['seed']}"
            assert contract["method"] == metadata["method"] and contract["seed"] == metadata["seed"]
            assert checkpoints["selected"] == metadata
            assert contract == selected["contract"]
            assert contract["config_hash"] == canonical(configuration, compact=True)
            assert contract["source_hash"] == source_digest.hexdigest()
            assert contract["environment_hash"] == canonical(environment, compact=True)
            assert contract["manifest_hash"] == suite["manifest_sha256"] == data_validation["manifest_sha256"]
            assert contract["prepared_report_hash"] == suite["data_validation_sha256"]
            assert metadata["completed_epochs"] == metadata["planned_epochs"] == configuration["epochs"] == 10
            assert metadata["best_checkpoint_sha256"] == selected["checkpoint_sha256"]
            assert metadata["best_epoch"] == selected["best_epoch"]
            run_history = history.loc[history.run_id == run_id]
            assert sorted(run_history.epoch.tolist()) == list(range(1, 11))
            assert sorted(item["epoch"] for item in checkpoints["epochs"]) == list(range(1, 11))
            checkpoint_bytes += sum(item["bytes"] for item in checkpoints["epochs"])
            all_checkpoint_hashes.extend(item["sha256"] for item in checkpoints["epochs"])
            selected_sidecar = next(item for item in checkpoints["epochs"] if item["epoch"] == metadata["best_epoch"])
            assert selected_sidecar["sha256"] == metadata["best_checkpoint_sha256"]
            assert metadata["best_checkpoint"] == "checkpoints/" + selected_sidecar["path"]
            chosen = run_history.sort_values(["validation_roc_auc", "validation_loss", "epoch"], ascending=[False, True, True]).iloc[0]
            assert int(chosen.epoch) == metadata["best_epoch"]
            assert math.isclose(chosen.validation_loss, metadata["best_validation_loss"], abs_tol=1e-10)
            assert np.isfinite(run_history.select_dtypes(include="number").to_numpy()).all()
            expected_rates = [configuration["eta_min"] + (configuration["learning_rate"] - configuration["eta_min"]) *
                              (1 + math.cos(math.pi * (epoch - 1) / 10)) / 2 for epoch in run_history.epoch]
            assert np.allclose(run_history.learning_rate, expected_rates, rtol=1e-10, atol=1e-10)
            external = predictions.loc[predictions.run_id == run_id].reset_index(drop=True)
            assert set(external.method) == {metadata["method"]} and set(external.seed) == {metadata["seed"]}
            pd.testing.assert_frame_equal(external, read_csv(f"runs/{run_id}/predictions.csv"), check_exact=False, atol=1e-14)
            validation = read_csv(f"runs/{run_id}/validation_predictions.csv")
            for frame, split in [(external, "test"), (validation, "validation")]:
                cohort = manifest.loc[manifest.split == split].set_index("sample_id")
                assert set(frame.sample_id) == set(cohort.index)
                assert not frame.sample_id.duplicated().any()
                for field in ("label", "group_id", "dataset"):
                    assert np.array_equal(frame[field].to_numpy(), cohort.loc[frame.sample_id, field].to_numpy())
            external_metrics = independent_metrics(external, configuration["threshold"])
            validation_metrics = independent_metrics(validation, configuration["threshold"])
            compare(read_json(f"runs/{run_id}/metrics.json"), external_metrics, f"{run_id}.external", differences)
            compare(metadata["best_validation_metrics"], validation_metrics, f"{run_id}.validation", differences)
            flat_external = {key: value for key, value in external_metrics.items() if not isinstance(value, (list, dict))}
            compare(comparison.loc[comparison.run_id == run_id].iloc[0].to_dict(), flat_external, f"{run_id}.csv", differences)
            aggregate_run = next(item for item in aggregate["runs"] if item["run_id"] == run_id)
            compare(aggregate_run, flat_external, f"{run_id}.aggregate", differences)
            compare({"roc_auc": chosen.validation_roc_auc, "accuracy": chosen.validation_accuracy},
                    {key: validation_metrics[key] for key in ("roc_auc", "accuracy")}, f"{run_id}.selected_history", differences)
            run_results.append({"run_id": run_id, "method": metadata["method"], "seed": metadata["seed"],
                                "best_epoch": metadata["best_epoch"], "validation": validation_metrics,
                                "external": external_metrics, "recorded_epoch_seconds": float(run_history.seconds.sum()),
                                "last_train_loss": float(run_history.iloc[-1].train_loss),
                                "last_validation_loss": float(run_history.iloc[-1].validation_loss)})
        assert len(set(all_checkpoint_hashes)) == 150
        methods = []
        for method, rows in comparison.groupby("method"):
            summary_row = summary.loc[summary.method == method].iloc[0]
            compare(summary_row.to_dict(), {"mean": rows.roc_auc.mean(), "std": rows.roc_auc.std(ddof=1), "count": 3},
                    f"{method}.seed_summary", differences)
            methods.append({"method": method, **{f"{metric}_{statistic}": float(getattr(rows[metric], statistic)())
                            for metric in ("roc_auc", "average_precision", "accuracy", "sensitivity", "specificity", "f1", "brier")
                            for statistic in ("mean", "std")}})
        bootstrap = independent_bootstrap(predictions, configuration["bootstrap_samples"], configuration["split_seed"])
        compare(aggregate["paired_uncertainty"]["methods"], bootstrap, "paired_bootstrap", differences)
        report = {
            "archive_sha256": digest(archive_bytes), "archive_files": len(names), "checksum_verified_files": len(checksums),
            "status": "PASS" if not differences else "FAIL", "differences": differences,
            "independent_implementation": "sklearn metrics; pairwise ranking-statistic bootstrap with joint stratified case draws",
            "review_environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
            "counts": counts, "unique_cases": len(manifest), "external_cases": 138, "external_predictions": len(predictions),
            "validation_cases": 133, "validation_predictions": 15 * 133, "completed_runs": len(run_results), "completed_epochs": len(history),
            "fixed_split_reproduced": True,
            "physical_batch": configuration["microbatch_size"], "effective_batch": configuration["batch_size"],
            "declared_checkpoint_bytes": checkpoint_bytes, "recorded_epoch_seconds": float(history.seconds.sum()),
            "source_sha256": suite["source_sha256"], "checkpoint_source_sha256": source_digest.hexdigest(),
            "methods": methods, "paired_bootstrap": bootstrap, "runs": run_results,
            "limitations": [
                "Large .pt weights and original images are absent: checkpoint and image digests are internally consistent metadata, not independently rehashed binaries.",
                "Export removes original manifest paths and data-report roots: original raw file hashes cannot be reproduced from the sanitized CSV/report, but all references agree.",
                "Group IDs are one case per image; patient independence and near-duplicate absence remain unverified.",
                "Bootstrap intervals condition on these three trained seeds and the same training cohort; seeds are not extra patients.",
                "All four enhancement-versus-identity AUC difference intervals include zero. No enhancement improvement is established.",
                "Secondary contrasts are exploratory without multiplicity correction; fixed threshold0.5 exhibits substantial seed variability.",
                "Recorded epoch seconds exclude checkpoint/save overhead and are not full end-to-end elapsed runtime.",
                "SHA256 verifies consistency and corruption detection, not independent authenticity of self-reported experiment provenance.",
            ],
        }
    assert arguments.archive.read_bytes() == archive_bytes
    arguments.output.mkdir(parents=True, exist_ok=True)
    target = arguments.output / "verification.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(methods).to_csv(arguments.output / "verified_method_summary.csv", index=False)
    markdown = ["# Independent Returned-Results Verification", "", f"Status: **{report['status']}**.",
                f"Archive SHA-256: `{report['archive_sha256']}`.",
                f"Verified {len(checksums)} checksums; 15 external and 15 validation metric records; 150 history rows; 2,000 joint paired bootstrap replicates.",
                "", "| Method | External ROC-AUC mean ± seed SD | Accuracy | Sensitivity | Specificity |", "|---|---:|---:|---:|---:|"]
    for item in methods:
        markdown.append(f"| {item['method']} | {item['roc_auc_mean']:.6f} ± {item['roc_auc_std']:.6f} | {item['accuracy_mean']:.6f} | {item['sensitivity_mean']:.6f} | {item['specificity_mean']:.6f} |")
    primary = bootstrap["corrected_unsharp"]
    markdown.extend(["", f"Primary AUC difference: {primary['difference_vs_identity']:.9f}; paired 95% interval: {primary['paired_difference_ci95']}.",
                     "No enhancement benefit is established by the predeclared primary comparison.", "", "## Limitations", ""])
    markdown.extend("- " + item for item in report["limitations"])
    (arguments.output / "VERIFICATION.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "archive_sha256", "checksum_verified_files", "completed_runs", "completed_epochs", "differences", "methods", "recorded_epoch_seconds")}, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
