"""Build the delivered Colab notebook from inspectable, deterministic cell sources."""

import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def build_notebook():
    root = Path(__file__).resolve().parents[1]
    package_paths = sorted([*root.glob("src/cxr_enhance/*.py"), *root.glob("configs/*.json"),
                            *root.glob("scripts/*.py"), *root.glob("tests/*.py"),
                            *root.glob("requirements*.txt"), root / "pyproject.toml"])
    package_hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in package_paths}
    cells = [
        markdown("""
        # Controlled Chest X-ray Enhancement Experiments

        **Phase 3 delivery — real-data training and independent results review are pending.**

        Research question: does a specified enhancement change normal versus TB-consistent
        classification under one fixed EfficientNet-V2-S training protocol?

        **Three steps:**
        1. Select a GPU runtime with Python 3.12 or 3.13. Keep the delivered
           `chest-xray-enhancement-tb.zip` available and run the setup/upload cells.
        2. Read the source data terms, set the persistent Drive paths, and run validation
           and preflight before the 15-run experiment. Check your Drive storage quota.
        3. Run the frozen external evaluation and export cells. Return the ZIP, this
           notebook saved with outputs, logs, and the selected checkpoint locations.

        Five conditions × seeds 42/43/44 × ten epochs. Shenzhen has a fixed 80/20
        development split (seed 2026); Montgomery is external test data only.
        The primary contrast is corrected unsharp minus identity in ROC-AUC.
        No threshold, model, or preprocessing choice may be changed using test scores.

        See `PROJECT_AUDIT.md`, `docs/RESEARCH_PROTOCOL.md`, and `docs/COLAB_GUIDE.md`
        in the package. No dataset or clinical usefulness is claimed by this notebook.
        """),
        markdown("""
        ## Runtime configuration

        Research mode is the default. The package is uploaded only when a local project
        directory is unavailable. Re-running cells reuses the same prepared data and run
        directory; checkpoints are not overwritten. Use **one Colab runtime per destination**:
        local process locks do not coordinate two different Colab VMs.

        For local notebook verification, set environment variables before execution:
        `CXR_MODE=smoke`, `CXR_SKIP_INSTALL=1`, `CXR_PROJECT_DIR=/absolute/package/path`,
        `CXR_WORK_ROOT=/absolute/new/scratch/path`. Smoke mode uses only deterministic
        synthetic images and a tiny model; its scores are not research results.
        """),
        code("""
        import datetime
        import hashlib
        import io
        import json
        import os
        from pathlib import Path, PurePosixPath
        import shutil
        import stat
        import subprocess
        import sys
        import tempfile
        import uuid
        import zipfile

        if not (3, 12) <= sys.version_info[:2] < (3, 14):
            raise RuntimeError("Use Python 3.12 or 3.13 for the pinned environment.")
        MODE = os.environ.get("CXR_MODE", "research")
        if MODE not in {"research", "smoke"}:
            raise ValueError("CXR_MODE must be research or smoke.")
        SKIP_INSTALL = os.environ.get("CXR_SKIP_INSTALL", "0") == "1"
        IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
        PROJECT_INPUT = os.environ.get("CXR_PROJECT_DIR")
        WORK_INPUT = os.environ.get("CXR_WORK_ROOT")
        DOWNLOAD_IF_MISSING = True
        PATIENT_MAP_INPUT = os.environ.get("CXR_PATIENT_MAP", "")
        EXPECTED_PACKAGE_HASHES = __EXPECTED_PACKAGE_HASHES__
        PACKAGE_ID = hashlib.sha256(json.dumps(EXPECTED_PACKAGE_HASHES, sort_keys=True).encode()).hexdigest()[:16]
        SOURCE_CODE_HASH = hashlib.sha256(json.dumps(
            {Path(name).name: value for name, value in EXPECTED_PACKAGE_HASHES.items()
             if name.startswith("src/cxr_enhance/")}, sort_keys=True).encode()).hexdigest()
        print({"mode": MODE, "python": sys.version, "in_colab": IN_COLAB,
               "install_skipped": SKIP_INSTALL, "package_id": PACKAGE_ID})
        """),
        markdown("""
        ## Load the source package and mount persistent storage

        In Colab, upload the **source ZIP** when prompted. Do not upload a results ZIP here.
        Extraction accepts a single `chest-xray-enhancement-tb/` root and rejects unsafe
        paths, links, and duplicate members. The package contains code, not private images.
        Drive holds data, split evidence, logs, and large checkpoints.

        Use the ZIP delivered alongside this notebook. Its code/configuration hashes
        must match this release. Older extracted packages are preserved separately;
        opening a revised notebook cannot silently reuse their code.
        """),
        code("""
        if PROJECT_INPUT:
            PROJECT_DIR = Path(PROJECT_INPUT).expanduser().resolve()
        else:
            if not IN_COLAB:
                raise RuntimeError("Set CXR_PROJECT_DIR to the delivered package when running outside Colab.")
            release_root = Path("/content") / f"cxr-source-{PACKAGE_ID}"
            release_root.mkdir(parents=True, exist_ok=True)
            PROJECT_DIR = release_root / "chest-xray-enhancement-tb"
            if not PROJECT_DIR.exists():
                from google.colab import files

                uploaded = files.upload()
                if len(uploaded) != 1:
                    raise ValueError("Select only the source ZIP delivered with this notebook.")
                archive_name, archive_bytes = next(iter(uploaded.items()))
                with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                    members = archive.infolist()
                    names = [member.filename for member in members]
                    if len(names) != len(set(names)) or sum(member.file_size for member in members) > 100_000_000:
                        raise ValueError("Duplicate members or unexpectedly large source archive.")
                    for member in members:
                        relative = PurePosixPath(member.filename)
                        mode = member.external_attr >> 16
                        if (relative.is_absolute() or ".." in relative.parts or "\\\\" in member.filename
                                or ":" in member.filename or not relative.parts
                                or relative.parts[0] != "chest-xray-enhancement-tb"
                                or stat.S_ISLNK(mode)):
                            raise ValueError(f"Unsafe source ZIP member: {member.filename}")
                    for relative_name, expected_hash in EXPECTED_PACKAGE_HASHES.items():
                        member_name = f"chest-xray-enhancement-tb/{relative_name}"
                        if member_name not in names or hashlib.sha256(archive.read(member_name)).hexdigest() != expected_hash:
                            raise ValueError("This source ZIP does not match this notebook release. "
                                             "Upload the matching revised ZIP; older files are unchanged.")
                    with tempfile.TemporaryDirectory(prefix="extract-", dir=release_root) as staging:
                        staging_root = Path(staging)
                        for member in members:
                            destination = staging_root.joinpath(*PurePosixPath(member.filename).parts)
                            if member.is_dir():
                                destination.mkdir(parents=True, exist_ok=True)
                            else:
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                with destination.open("xb") as target:
                                    target.write(archive.read(member))
                        (staging_root / "chest-xray-enhancement-tb").rename(PROJECT_DIR)
        if not (PROJECT_DIR / "pyproject.toml").is_file() or not (PROJECT_DIR / "src/cxr_enhance").is_dir():
            raise FileNotFoundError("The project directory does not contain the delivered source package.")
        for relative_name, expected_hash in EXPECTED_PACKAGE_HASHES.items():
            package_file = PROJECT_DIR / relative_name
            if not package_file.is_file() or hashlib.sha256(package_file.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Source package mismatch: {relative_name}. Use the matching notebook and ZIP.")
        if MODE == "research" and not WORK_INPUT:
            if not IN_COLAB:
                raise RuntimeError("Set CXR_WORK_ROOT for a local research run.")
            from google.colab import drive

            drive.mount("/content/drive", force_remount=False)
            WORK_ROOT = Path("/content/drive/MyDrive/cxr-enhancement-tb")
        elif WORK_INPUT:
            WORK_ROOT = Path(WORK_INPUT).expanduser().resolve()
        else:
            raise RuntimeError("Set a unique CXR_WORK_ROOT for synthetic verification.")
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        SESSION_ID = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        LOG_DIR = WORK_ROOT / "execution_logs" / SESSION_ID
        LOG_DIR.mkdir(parents=True, exist_ok=False)
        DATA_ROOT = Path(os.environ.get("CXR_DATA_ROOT", str(WORK_ROOT / "data"))).expanduser().resolve()
        PREPARED_DIR = WORK_ROOT / "prepared"
        SUITE_DIR = WORK_ROOT / "runs" / ("balanced" if MODE == "research" else "smoke")
        RESULT_ZIP = WORK_ROOT / "results_for_review.zip"
        print({"project": str(PROJECT_DIR), "work_root": str(WORK_ROOT),
               "data_root": str(DATA_ROOT), "suite": str(SUITE_DIR), "logs": str(LOG_DIR)})
        """),
        markdown("""
        ## Install pinned dependencies in an isolated environment

        Colab maintains its own host dependency pins. The setup creates a separate
        environment at `/content/cxr-enhance-venv`, leaving the notebook kernel and
        `google.colab` packages intact. Every research command uses that environment's
        Python in a fresh subprocess. A new runtime recreates this local environment;
        Drive checkpoints remain unchanged. Local installation uses `WORK_ROOT/.venv`.

        The environment installs PyTorch 2.13.0/Torchvision 0.28.0, selecting the CUDA
        12.6 wheel channel when an NVIDIA device is reported, otherwise CPU. This is
        a documented target, not a claim of a previously completed GPU run. Preflight
        verifies the actual session. `CXR_SKIP_INSTALL=1` uses the current interpreter
        for local verification and does not run pip or mutate its environment.
        """),
        code("""
        SESSION_ID = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        setup_log_root = WORK_ROOT / "execution_logs"
        setup_log_root.mkdir(parents=True, exist_ok=True)
        LOG_DIR = Path(tempfile.mkdtemp(prefix=SESSION_ID + "-setup-", dir=setup_log_root))
        COMMAND_ENV = os.environ.copy()
        COMMAND_ENV["PYTHONPATH"] = str(PROJECT_DIR / "src") + os.pathsep + COMMAND_ENV.get("PYTHONPATH", "")
        COMMAND_ENV["PYTHONUNBUFFERED"] = "1"
        COMMAND_ENV["PYTHONHASHSEED"] = "2026"
        COMMAND_ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if MODE == "smoke":
            COMMAND_ENV["OMP_NUM_THREADS"] = "2"
            COMMAND_ENV["MKL_NUM_THREADS"] = "2"
        STAGE_RECORDS = []
        PIPELINE_PYTHON = sys.executable

        def run_command(arguments, stage):
            command = [str(argument) for argument in arguments]
            started = datetime.datetime.now(datetime.timezone.utc).isoformat()
            log_path = LOG_DIR / f"{len(STAGE_RECORDS):02d}_{stage}_{uuid.uuid4().hex[:12]}.log"
            print("Running:", command)
            return_code = None
            process = None
            try:
                with log_path.open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=PROJECT_DIR, env=COMMAND_ENV,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, encoding="utf-8", errors="replace")
                    for line in process.stdout:
                        print(line, end="")
                        log.write(line)
                        log.flush()
                    return_code = process.wait()
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                record = {"stage": stage, "command": command, "started_at": started,
                          "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "exit_code": return_code, "log_path": str(log_path),
                          "status": "interrupted_or_not_started" if return_code is None else
                                    "passed" if return_code == 0 else "failed"}
                STAGE_RECORDS.append(record)
                (LOG_DIR / "commands.json").write_text(json.dumps(STAGE_RECORDS, indent=2) + "\\n", encoding="utf-8")
            if return_code:
                raise RuntimeError(f"Stage {stage} failed ({return_code}); retain {log_path}")
            return record

        def cli(*arguments, stage):
            return run_command([PIPELINE_PYTHON, "-m", "cxr_enhance", *arguments], stage)

        def ensure_environment_pip(environment_python):
            probe = subprocess.run([str(environment_python), "-m", "pip", "--version"],
                                   capture_output=True, text=True, env=COMMAND_ENV)
            if probe.returncode == 0:
                print(probe.stdout.strip())
                return
            print("The isolated environment has no working pip; bootstrapping it.")
            try:
                run_command([environment_python, "-m", "ensurepip", "--upgrade"], "bootstrap_pip_ensurepip")
            except RuntimeError:
                print("ensurepip was unavailable; using the notebook host pip to manage only the isolated environment.")
                run_command([sys.executable, "-m", "pip", "--python", environment_python,
                             "install", "pip"], "bootstrap_pip_from_host")
            run_command([environment_python, "-m", "pip", "--version"], "verify_isolated_pip")

        nvidia = shutil.which("nvidia-smi")
        gpu_detected = False
        if nvidia:
            probe = subprocess.run([nvidia, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                   capture_output=True, text=True)
            print(probe.stdout or probe.stderr)
            gpu_detected = probe.returncode == 0 and bool(probe.stdout.strip())
        COMMAND_ENV["CXR_EXPECT_CUDA"] = "1" if gpu_detected and MODE == "research" else "0"
        if not SKIP_INSTALL:
            environment_root = Path("/content/cxr-enhance-venv") if IN_COLAB else WORK_ROOT / ".venv"
            environment_python = environment_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not environment_root.exists():
                try:
                    run_command([sys.executable, "-m", "venv", "--without-pip", environment_root],
                                "create_isolated_environment")
                except RuntimeError as error:
                    raise RuntimeError("Cannot create the isolated environment. Use a Python 3.12/3.13 "
                                       "runtime with venv available; preserve the setup log. "
                                       "No research dependency was installed into the notebook host.") from error
            if not environment_python.is_file():
                raise RuntimeError(f"Incomplete isolated environment at {environment_root}. "
                                   "Use a fresh runtime or a new local work directory; existing evidence is retained.")
            PIPELINE_PYTHON = str(environment_python)
            ensure_environment_pip(environment_python)
            wheel_channel = "cu126" if gpu_detected else "cpu"
            run_command([PIPELINE_PYTHON, "-m", "pip", "install", "torch==2.13.0", "torchvision==0.28.0",
                         "--index-url", f"https://download.pytorch.org/whl/{wheel_channel}"], "install_torch")
            run_command([PIPELINE_PYTHON, "-m", "pip", "install", "-r", PROJECT_DIR / "requirements-dev.txt"],
                        "install_dependencies")
            run_command([PIPELINE_PYTHON, "-m", "pip", "install", "--no-deps", "-e", PROJECT_DIR],
                        "install_project")
            run_command([PIPELINE_PYTHON, "-m", "pip", "check"], "isolated_dependency_check")
        print({"notebook_kernel_python": sys.executable, "pipeline_python": PIPELINE_PYTHON})
        runtime_probe = (
            "import json, os, torch, torchvision; from cxr_enhance.utils import environment_info; "
            "print(json.dumps(environment_info(), indent=2)); "
            "print({'cuda_available': torch.cuda.is_available(), 'cuda_build': torch.version.cuda}); "
            "\\nif os.environ.get('CXR_EXPECT_CUDA') == '1' and not torch.cuda.is_available():"
            "\\n    raise RuntimeError('NVIDIA hardware was detected but PyTorch cannot use CUDA. '"
            "'Check the GPU runtime, driver compatibility and isolated cu126 installation; '"
            "'retain this diagnostic log. Research training has not started.')"
        )
        run_command([PIPELINE_PYTHON, "-c", runtime_probe], "runtime_versions")
        """),
        markdown("""
        ## Software checks before data preparation

        These tests cover filtering, schemas, splits, metrics, checkpoint safety, and
        synthetic training/recovery. They do not train on clinical data or validate a
        medical claim. A unique temporary directory keeps the checks separate from
        experiment checkpoints. Any failure stops the notebook.
        """),
        code("""
        check_root = Path(tempfile.gettempdir()) / "cxr-software-checks"
        check_root.mkdir(parents=True, exist_ok=True)
        test_temp = Path(tempfile.mkdtemp(prefix="pytest-", dir=check_root))
        validation_record = run_command([PIPELINE_PYTHON, "-m", "pytest", "-q", "--basetemp", test_temp],
                                        "unit_tests")
        validation_record = {**validation_record, "kind": "software_tests_only",
                             "source_hash": SOURCE_CODE_HASH, "package_id": PACKAGE_ID,
                             "stdout": Path(validation_record["log_path"]).read_text(encoding="utf-8")}
        (LOG_DIR / "unit_tests.json").write_text(json.dumps(validation_record, indent=2) + "\\n", encoding="utf-8")
        SUITE_DIR.mkdir(parents=True, exist_ok=True)
        validation_path = SUITE_DIR / f"software_validation_{PACKAGE_ID}.json"
        if not validation_path.exists():
            with validation_path.open("x", encoding="utf-8") as stream:
                json.dump(validation_record, stream, indent=2)
        else:
            print("Same-release software-validation record retained; the latest check is in", LOG_DIR)
        """),
        markdown("""
        ## Locate or acquire data; freeze split membership

        Expected real inputs: 662 Shenzhen images and 138 Montgomery images, in separate
        `shenzhen/CXR_png` and `montgomery/CXR_png` directories under `DATA_ROOT`.
        Read the [NLM source terms](https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html)
        and package data card before acquisition. The datasets are not redistributed here.
        If a cohort is missing, `DOWNLOAD_IF_MISSING=True` acquires it from the official NLM
        source. Set it to `False` to require your existing private copy.

        Preparation rejects anomalies and duplicates; it does not silently drop samples.
        A complete reliable patient mapping can be supplied with `CXR_PATIENT_MAP`.
        Otherwise group IDs represent cases, and patient independence is unverified.
        Synthetic mode generates its own small labeled fixtures and marks all outputs.
        """),
        code("""
        if MODE == "smoke":
            DATA_ROOT = WORK_ROOT / "synthetic_data"
            config_path = WORK_ROOT / "synthetic_config.json"
            bootstrap = (
                "import json, sys; from pathlib import Path; "
                "from cxr_enhance.smoke import create_synthetic_data, smoke_config; "
                "root=Path(sys.argv[1]); target=Path(sys.argv[2]); "
                "create_synthetic_data(root) if not root.exists() else print('Reusing synthetic fixtures'); "
                "content=json.dumps(smoke_config(), indent=2, sort_keys=True)+chr(10); "
                "existing=target.read_text(encoding='utf-8') if target.exists() else None; "
                "assert existing is None or json.loads(existing)==smoke_config(), 'Synthetic config changed'; "
                "target.write_text(content, encoding='utf-8') if existing is None else None"
            )
            run_command([PIPELINE_PYTHON, "-c", bootstrap, DATA_ROOT, config_path], "synthetic_fixtures")
        else:
            config_path = PROJECT_DIR / "configs/balanced.json"
            for dataset in ("shenzhen", "montgomery"):
                cohort_root = DATA_ROOT / dataset
                managed_acquisition = ((cohort_root / "acquisition_receipts").exists()
                                       or (cohort_root / ".cxr_acquisition").exists())
                image_directories = list(cohort_root.rglob("CXR_png")) if cohort_root.exists() else []
                missing_images = not any(directory.is_dir() and any(directory.glob("*.png"))
                                         for directory in image_directories)
                if missing_images or (managed_acquisition and not (cohort_root / "acquisition_manifest.json").is_file()):
                    if not DOWNLOAD_IF_MISSING:
                        raise FileNotFoundError(f"Provide the complete private {dataset} dataset at {cohort_root}")
                    cli("download", "--dataset", dataset, "--output", cohort_root, stage=f"download_{dataset}")
        preparation_arguments = ["prepare", "--shenzhen", DATA_ROOT / "shenzhen",
                                 "--montgomery", DATA_ROOT / "montgomery", "--output", PREPARED_DIR]
        if MODE == "smoke":
            preparation_arguments.append("--synthetic")
        elif PATIENT_MAP_INPUT:
            preparation_arguments.extend(["--patient-map", Path(PATIENT_MAP_INPUT).expanduser().resolve()])
        cli(*preparation_arguments, stage="validate_and_prepare")
        MANIFEST = PREPARED_DIR / "manifest.csv"
        print((PREPARED_DIR / "data_validation.json").read_text(encoding="utf-8"))
        """),
        markdown("""
        ## Model preflight and storage estimate

        Preflight measures a forward/backward pass with the actual selected architecture,
        chooses one microbatch, and writes `resolved_config.json`. It does not fit real
        images. Timing excludes data loading, validation, checkpoint I/O, and Drive
        latency. Gradient accumulation does not enlarge BatchNorm's physical batch.

        All 150 epoch checkpoints from the research suite are retained; storage can be
        tens of gigabytes. Check your **Drive quota separately** from the filesystem free
        space. Preserve old checkpoints and choose sufficient storage before starting.
        """),
        code("""
        cli("preflight", "--config", config_path, "--output", SUITE_DIR, stage="preflight")
        RESOLVED_CONFIG = SUITE_DIR / "resolved_config.json"
        resolved = json.loads(RESOLVED_CONFIG.read_text(encoding="utf-8"))
        preflight_report = json.loads((SUITE_DIR / "preflight.json").read_text(encoding="utf-8"))
        print(json.dumps({"resolved_config": resolved, "preflight": preflight_report}, indent=2))
        if MODE == "research" and (resolved["synthetic"] or len(resolved["methods"]) != 5
                                   or resolved["seeds"] != [42, 43, 44] or resolved["epochs"] != 10):
            raise ValueError("Research mode requires the approved five-method, three-seed, ten-epoch configuration.")
        """),
        markdown("""
        ## Train or resume the approved suite

        Keep this code, environment, split, configuration, and destination unchanged.
        Resume replays only an interrupted partial epoch and continues from the last
        complete checkpoint. Environment or input mismatches intentionally stop recovery;
        do not edit metadata to bypass them. Logs are streamed and saved in `execution_logs`.

        After external evaluation starts, training is permanently frozen. A later
        **Run all** skips training and verifies the frozen evaluation/export instead.
        """),
        code("""
        if (SUITE_DIR / "evaluation_lock.json").exists():
            print("External evaluation is already frozen; skipping training.")
        else:
            cli("train", "--config", RESOLVED_CONFIG, "--manifest", MANIFEST,
                "--suite", SUITE_DIR, "--resume", stage="train_suite")
        """),
        markdown("""
        ## Evaluate frozen checkpoints on the external cohort

        All configured training runs must be complete. Evaluation freezes checkpoint
        hashes before computing Montgomery metrics. It saves probabilities and ground
        truth, per-seed metrics, paired bootstrap intervals, calibration, and learning
        curves. It never uses external results to select an epoch or threshold.

        A technical restart may repeat or complete this same frozen evaluation. Any
        scientific revision after viewing these results requires a separately reviewed
        study; Montgomery would no longer be an untouched test cohort for that revision.
        """),
        code("""
        import csv
        import html
        from IPython.display import HTML, Image, display

        cli("evaluate", "--manifest", MANIFEST, "--suite", SUITE_DIR, stage="external_evaluation")
        print((SUITE_DIR / "evaluation/evaluation_report.md").read_text(encoding="utf-8"))
        print("Plots and numeric outputs:", SUITE_DIR / "evaluation")
        with (SUITE_DIR / "evaluation/model_comparison.csv").open(encoding="utf-8", newline="") as stream:
            comparison_reader = csv.DictReader(stream)
            comparison_columns = comparison_reader.fieldnames or []
            comparison_rows = list(comparison_reader)
        header = "".join(f"<th>{html.escape(column)}</th>" for column in comparison_columns)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[column]))}</td>"
                                        for column in comparison_columns) + "</tr>"
                       for row in comparison_rows)
        display(HTML("<div style='overflow-x:auto'><table border='1' cellpadding='5'>"
                     f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"))
        display(Image(filename=str(SUITE_DIR / "evaluation/model_comparison.png")))
        first_run = f"{resolved['methods'][0]}__seed_{resolved['seeds'][0]}"
        print("Preview of first configured run (not selected by test performance):", first_run)
        for figure_name in ("learning_curves.png", "confusion_matrix.png"):
            display(Image(filename=str(SUITE_DIR / "evaluation/runs" / first_run / figure_name)))
        """),
        markdown("""
        ## Export and independently verify the compact results package

        The ZIP includes numeric predictions/labels, metrics, configuration/environment,
        training histories, model comparisons, plots, split evidence, checkpoint metadata,
        software-check evidence, and SHA-256 checksums. Large `.pt` checkpoints remain
        separately in Drive. If the ZIP exists, it is verified rather than overwritten.

        **Return:** `results_for_review.zip`, this executed notebook saved with outputs,
        any interruption/error logs, and selected checkpoint locations/checksums. Keep
        the review private; do not publish images, identifiers, or model artifacts.
        No improved-performance claim or CV/LinkedIn work precedes independent Phase 4
        review. Synthetic mode's exported metrics are software diagnostics only.
        """),
        code("""
        if RESULT_ZIP.exists():
            print("Preserving and verifying existing results package:", RESULT_ZIP)
        else:
            cli("export", "--suite", SUITE_DIR, "--output", RESULT_ZIP, stage="export_results")
        cli("verify", RESULT_ZIP, stage="verify_export")
        print({"results_zip": str(RESULT_ZIP), "mode": MODE,
               "checkpoint_root": str(SUITE_DIR / "runs"), "execution_logs": str(LOG_DIR),
               "next_step": "Return the ZIP, executed notebook, logs, and checkpoint locations for independent review."})
        if IN_COLAB and MODE == "research":
            from google.colab import files

            files.download(str(RESULT_ZIP))
        """),
    ]
    for cell in cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("__EXPECTED_PACKAGE_HASHES__", repr(package_hashes))
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": "04_colab_improved_training.ipynb", "provenance": []},
        "accelerator": "GPU",
    })
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"cxr-{index:02d}"
    nbformat.validate(notebook)
    return notebook


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "04_colab_improved_training.ipynb"
    nbformat.write(build_notebook(), destination)
    print(destination)


if __name__ == "__main__":
    main()
