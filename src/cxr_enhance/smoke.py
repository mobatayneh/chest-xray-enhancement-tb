"""Clearly synthetic integration fixtures; never a substitute for chest radiograph experiments."""

from pathlib import Path

import cv2
import numpy as np

from .config import validate_config
from .data import prepare_dataset
from .evaluation import evaluate_suite, export_results, initialize_suite, run_name, verify_export
from .training import preflight, train_run
from .utils import write_json


def create_synthetic_data(destination: Path) -> dict[str, Path]:
    roots = {}
    generator = np.random.default_rng(9182)
    for dataset, prefix, count in (("shenzhen", "CHNCXR", 30), ("montgomery", "MCUCXR", 10)):
        root = Path(destination) / dataset
        directory = root / "CXR_png"
        directory.mkdir(parents=True, exist_ok=False)
        for index in range(count):
            label = index % 2
            height, width = 48 + index % 7, 39 + index % 5
            row_grid, column_grid = np.mgrid[:height, :width]
            texture = generator.integers(0, 800, (height, width))
            shape = ((row_grid > height // 2) if label else (column_grid > width // 2)) * 1000
            image = (texture + row_grid * 13 + column_grid * 5 + shape).astype(np.uint16)
            path = directory / f"{prefix}_{index + 1:04d}_{label}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write synthetic fixture: {path}")
        roots[dataset] = root
    return roots


def smoke_config() -> dict:
    return validate_config({"model": "tiny", "weights": None, "synthetic": True,
                            "image_size": 64, "epochs": 2, "seeds": [42],
                            "batch_size": 4, "microbatch_size": 2, "workers": 0,
                            "amp": False, "bootstrap_samples": 30})


def run_smoke(destination: Path) -> dict:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new smoke directory to preserve previous verification evidence")
    destination.mkdir(parents=True)
    roots = create_synthetic_data(destination / "synthetic_data")
    manifest = prepare_dataset(roots["shenzhen"], roots["montgomery"], destination / "prepared", allow_partial=True)
    configuration = smoke_config()
    suite_dir = destination / "suite"
    measured = preflight(configuration, suite_dir)
    configuration["microbatch_size"] = measured["microbatch_size"]
    initialize_suite(suite_dir, configuration, manifest)
    for method in configuration["methods"]:
        train_run(configuration, manifest, roots, suite_dir / "runs" / run_name(method, 42), method, 42)
    evaluate_suite(suite_dir, manifest, roots)
    archive = export_results(suite_dir, destination / "synthetic_results_for_review.zip")
    status = {"synthetic_only": True, "real_training_completed": False,
              "methods": configuration["methods"], "epochs_per_run": configuration["epochs"],
              "export_verification": verify_export(archive)}
    write_json(destination / "smoke_status.json", status)
    return status
