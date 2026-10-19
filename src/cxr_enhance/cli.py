"""Command-line entry points shared by local workflows and the Colab notebook."""

import argparse
import json
import logging
from pathlib import Path

from .config import load_config, validate_config
from .utils import write_json


def _data_roots(manifest: Path, shenzhen: str | None, montgomery: str | None) -> dict[str, Path]:
    report = json.loads((manifest.parent / "data_validation.json").read_text(encoding="utf-8"))
    roots = report.get("roots", {})
    if shenzhen:
        roots["shenzhen"] = shenzhen
    if montgomery:
        roots["montgomery"] = montgomery
    if set(roots) != {"shenzhen", "montgomery"}:
        raise ValueError("Both dataset roots must be supplied or recorded during preparation")
    return {name: Path(path) for name, path in roots.items()}


def run_preflight(config_path: Path, output: Path) -> dict:
    from .training import preflight

    config = load_config(config_path)
    resolved = output / "resolved_config.json"
    if resolved.exists():
        saved = load_config(resolved)
        if {**config, "microbatch_size": saved["microbatch_size"]} != saved:
            raise ValueError("This preflight belongs to another configuration. Use a new directory.")
        return json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    result = preflight(config, output)
    if not result["disk_capacity_sufficient_for_checkpoints"]:
        raise OSError("Insufficient estimated disk space for immutable epoch checkpoints")
    config["microbatch_size"] = result["microbatch_size"]
    write_json(resolved, validate_config(config))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproducible CXR enhancement research; no clinical deployment.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Acquire only official CXR PNG files; no redistribution")
    download.add_argument("--dataset", choices=["shenzhen", "montgomery"], required=True)
    download.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Validate both cohorts and freeze a development split")
    prepare.add_argument("--shenzhen", type=Path, required=True)
    prepare.add_argument("--montgomery", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--patient-map", type=Path)
    prepare.add_argument("--synthetic", action="store_true", help="Permit small synthetic fixtures; not benchmark data")
    prepare.add_argument("--split-seed", type=int, default=2026)
    flight = commands.add_parser("preflight", help="Measure synthetic memory/compute and freeze one microbatch")
    flight.add_argument("--config", type=Path, required=True)
    flight.add_argument("--output", type=Path, required=True)
    for command in ("train", "evaluate"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--manifest", type=Path, required=True)
        subparser.add_argument("--suite", type=Path, required=True)
        subparser.add_argument("--shenzhen")
        subparser.add_argument("--montgomery")
        if command == "train":
            subparser.add_argument("--config", type=Path, required=True)
            subparser.add_argument("--resume", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--suite", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="Verify checksums and independently recompute saved metrics")
    verify.add_argument("archive", type=Path)
    smoke = commands.add_parser("smoke", help="Run all five arms using only synthetic images and a tiny network")
    smoke.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=arguments.log_level, format="%(asctime)s %(levelname)s %(message)s")
    if arguments.command == "download":
        from .data import download_official

        print(download_official(arguments.dataset, arguments.output))
    elif arguments.command == "prepare":
        from .data import prepare_dataset

        print(prepare_dataset(arguments.shenzhen, arguments.montgomery, arguments.output,
                              split_seed=arguments.split_seed, allow_partial=arguments.synthetic,
                              patient_map=arguments.patient_map))
    elif arguments.command == "preflight":
        print(json.dumps(run_preflight(arguments.config, arguments.output), indent=2))
    elif arguments.command == "train":
        from .evaluation import initialize_suite, run_name
        from .training import train_run

        if (arguments.suite / "evaluation_lock.json").exists():
            raise ValueError("External evaluation has started; training this suite is permanently frozen")
        config = load_config(arguments.config)
        initialize_suite(arguments.suite, config, arguments.manifest)
        roots = _data_roots(arguments.manifest, arguments.shenzhen, arguments.montgomery)
        for method in config["methods"]:
            for seed in config["seeds"]:
                metadata = train_run(config, arguments.manifest, roots,
                                     arguments.suite / "runs" / run_name(method, seed), method, seed,
                                     resume=arguments.resume)
                print(json.dumps(metadata, indent=2))
    elif arguments.command == "evaluate":
        from .evaluation import evaluate_suite

        roots = _data_roots(arguments.manifest, arguments.shenzhen, arguments.montgomery)
        print(evaluate_suite(arguments.suite, arguments.manifest, roots))
    elif arguments.command == "export":
        from .evaluation import export_results, verify_export

        print(export_results(arguments.suite, arguments.output))
        print(json.dumps(verify_export(arguments.output), indent=2))
    elif arguments.command == "verify":
        from .evaluation import verify_export

        print(json.dumps(verify_export(arguments.archive), indent=2))
    elif arguments.command == "smoke":
        from .smoke import run_smoke

        print(json.dumps(run_smoke(arguments.output), indent=2))
