"""Validate delivered notebook syntax and safe rerun behavior without real data."""

import ast
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import venv
import uuid
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "04_colab_improved_training.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def stage_source(notebook, marker: str) -> str:
    matches = [cell.source for cell in notebook.cells if cell.cell_type == "code" and marker in cell.source]
    assert len(matches) == 1
    return matches[0]


def test_notebook_schema_and_embedded_python_compile(notebook):
    nbformat.validate(notebook)
    embedded_sources = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id in {"bootstrap", "runtime_probe"}
                            for target in node.targets)):
                embedded_sources.append(ast.literal_eval(node.value))
    assert embedded_sources
    for source in embedded_sources:
        compile(source, "<notebook subprocess>", "exec")


def test_delivered_notebook_does_not_embed_execution_results(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []


@pytest.mark.parametrize("evaluation_started", [False, True])
def test_run_all_cannot_retrain_an_evaluation_frozen_suite(notebook, tmp_path, evaluation_started):
    if evaluation_started:
        (tmp_path / "evaluation_lock.json").write_text("{}", encoding="utf-8")
    calls = []
    namespace = {
        "SUITE_DIR": tmp_path,
        "RESOLVED_CONFIG": tmp_path / "resolved_config.json",
        "MANIFEST": tmp_path / "manifest.csv",
        "cli": lambda *arguments, **keywords: calls.append((arguments, keywords)),
    }
    exec(compile(stage_source(notebook, 'stage="train_suite"'), "<training cell>", "exec"), namespace)
    if evaluation_started:
        assert calls == []
    else:
        assert len(calls) == 1
        assert calls[0][0][0] == "train"
        assert "--resume" in calls[0][0]
        assert namespace["RESOLVED_CONFIG"] in calls[0][0]


@pytest.mark.parametrize("archive_exists", [False, True])
def test_run_all_verifies_existing_export_without_replacing_it(notebook, tmp_path, archive_exists):
    archive = tmp_path / "results_for_review.zip"
    original = b"preserved-review-package"
    if archive_exists:
        archive.write_bytes(original)
    calls = []
    namespace = {
        "RESULT_ZIP": archive,
        "SUITE_DIR": tmp_path / "suite",
        "LOG_DIR": tmp_path / "logs",
        "MODE": "smoke",
        "IN_COLAB": False,
        "cli": lambda *arguments, **keywords: calls.append(arguments),
    }
    exec(compile(stage_source(notebook, 'stage="verify_export"'), "<export cell>", "exec"), namespace)
    assert [arguments[0] for arguments in calls] == (["verify"] if archive_exists else ["export", "verify"])
    if archive_exists:
        assert archive.read_bytes() == original


def pip_bootstrap_function(notebook, namespace):
    tree = ast.parse(stage_source(notebook, '"runtime_versions"'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "ensure_environment_pip")
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<pip bootstrap>", "exec"), namespace)
    return namespace["ensure_environment_pip"]


@pytest.mark.parametrize("pip_present,ensurepip_fails,host_fails,expected", [
    (True, False, False, []),
    (False, False, False, ["bootstrap_pip_ensurepip", "verify_isolated_pip"]),
    (False, True, False, ["bootstrap_pip_ensurepip", "bootstrap_pip_from_host", "verify_isolated_pip"]),
    (False, True, True, ["bootstrap_pip_ensurepip", "bootstrap_pip_from_host"]),
])
def test_pip_bootstrap_recovers_and_propagates_failure(notebook, pip_present, ensurepip_fails,
                                                     host_fails, expected):
    calls = []
    target = Path("isolated-python")

    def runner(arguments, stage):
        calls.append((tuple(str(argument) for argument in arguments), stage))
        if (stage == "bootstrap_pip_ensurepip" and ensurepip_fails
                or stage == "bootstrap_pip_from_host" and host_fails):
            raise RuntimeError("Simulated bootstrap failure")

    namespace = {"subprocess": SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(
        returncode=0 if pip_present else 1, stdout="pip ready")),
        "sys": SimpleNamespace(executable="host-python"), "COMMAND_ENV": {}, "run_command": runner}
    bootstrap = pip_bootstrap_function(notebook, namespace)
    if host_fails:
        with pytest.raises(RuntimeError, match="bootstrap failure"):
            bootstrap(target)
    else:
        bootstrap(target)
    assert [stage for arguments, stage in calls] == expected
    host_calls = [arguments for arguments, stage in calls if stage == "bootstrap_pip_from_host"]
    if host_calls:
        assert host_calls == [("host-python", "-m", "pip", "--python", str(target), "install", "pip")]


def test_setup_retry_preserves_previous_logs(notebook, tmp_path):
    tree = ast.parse(stage_source(notebook, '"runtime_versions"'))
    prelude = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            break
        prelude.append(node)
    compiled = compile(ast.Module(body=prelude, type_ignores=[]), "<setup session>", "exec")
    namespace = {"datetime": datetime, "os": os, "tempfile": tempfile, "Path": Path,
                 "sys": sys, "WORK_ROOT": tmp_path, "PROJECT_DIR": tmp_path, "MODE": "smoke"}
    exec(compiled, namespace)
    first_directory = namespace["LOG_DIR"]
    previous_log = first_directory / "00_install_torch.log"
    previous_log.write_text("Original pip-missing failure", encoding="utf-8")
    exec(compiled, namespace)
    assert namespace["LOG_DIR"] != first_directory
    assert previous_log.read_text(encoding="utf-8") == "Original pip-missing failure"
    assert namespace["STAGE_RECORDS"] == []


def test_real_pipless_environments_recover_offline(notebook, tmp_path):
    ensurepip = pytest.importorskip("ensurepip", reason="Offline bootstrap wheel is unavailable on this Python")
    wheels = Path(ensurepip.__file__).parent / "_bundled"
    if not list(wheels.glob("pip-*.whl")):
        pytest.skip("Python distributor does not bundle an offline pip wheel")
    command_environment = {**os.environ, "PIP_NO_INDEX": "1", "PIP_FIND_LINKS": str(wheels),
                           "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    stages = []

    def runner(arguments, stage):
        stages.append(stage)
        if stage == "bootstrap_pip_ensurepip" and len(targets) == 2:
            raise RuntimeError("Simulated missing ensurepip in the second environment")
        result = subprocess.run([str(argument) for argument in arguments], env=command_environment,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    namespace = {"subprocess": subprocess, "sys": sys, "COMMAND_ENV": command_environment,
                 "run_command": runner}
    bootstrap = pip_bootstrap_function(notebook, namespace)
    targets = []
    for name in ("offline-ensurepip", "host-pip-recovery"):
        directory = tmp_path / name
        venv.EnvBuilder(with_pip=False).create(directory)
        target = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        targets.append(target)
        missing = subprocess.run([str(target), "-m", "pip", "--version"], env=command_environment,
                                 capture_output=True, text=True)
        assert missing.returncode != 0
        if len(targets) == 2:
            namespace["sys"] = SimpleNamespace(executable=str(targets[0]))
        bootstrap(target)
        location = subprocess.check_output([str(target), "-c", "import pip; print(pip.__file__)"],
                                           env=command_environment, text=True).strip()
        assert Path(location).resolve().is_relative_to(directory.resolve())
    assert stages == ["bootstrap_pip_ensurepip", "verify_isolated_pip", "bootstrap_pip_ensurepip",
                      "bootstrap_pip_from_host", "verify_isolated_pip"]


def test_interrupted_command_preserves_log_and_can_be_retried(notebook, tmp_path):
    processes = []

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.interrupted = not processes
            self.finished = False
            processes.append(self)
            self.stdout = self.output()

        def output(self):
            yield "Saved progress\n"
            if self.interrupted:
                raise KeyboardInterrupt

        def poll(self):
            return 0 if self.finished else None

        def terminate(self):
            self.finished = True

        def wait(self, timeout=None):
            self.finished = True
            return 0

    namespace = {"subprocess": SimpleNamespace(Popen=FakeProcess, PIPE=-1, STDOUT=-2),
                 "LOG_DIR": tmp_path, "PROJECT_DIR": tmp_path, "COMMAND_ENV": {},
                 "STAGE_RECORDS": [], "datetime": datetime, "uuid": uuid, "json": json}
    tree = ast.parse(stage_source(notebook, '"runtime_versions"'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_command")
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<command runner>", "exec"), namespace)
    with pytest.raises(KeyboardInterrupt):
        namespace["run_command"](["fake-python"], "train_suite")
    first = namespace["STAGE_RECORDS"][0]
    assert first["status"] == "interrupted_or_not_started"
    first_log = Path(first["log_path"])
    saved = first_log.read_bytes()
    second = namespace["run_command"](["fake-python"], "train_suite")
    assert second["status"] == "passed"
    assert first["log_path"] != second["log_path"]
    assert first_log.read_bytes() == saved
    assert len(json.loads((tmp_path / "commands.json").read_text())) == 2


def test_notebook_source_hash_matches_runtime_contract(notebook):
    from cxr_enhance.utils import source_hash

    namespace = {}
    exec(compile(stage_source(notebook, "EXPECTED_PACKAGE_HASHES ="), "<configuration>", "exec"), namespace)
    assert namespace["SOURCE_CODE_HASH"] == source_hash()


@pytest.mark.parametrize("changed", [False, True])
def test_package_validation_rejects_stale_source(notebook, tmp_path, changed):
    project = tmp_path / "project"
    source = project / "src/cxr_enhance"
    source.mkdir(parents=True)
    (project / "pyproject.toml").write_text("version='reviewed'", encoding="utf-8")
    (source / "__init__.py").write_text("", encoding="utf-8")
    hashes = {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (project / "pyproject.toml", source / "__init__.py")}
    if changed:
        (source / "__init__.py").write_text("stale = True", encoding="utf-8")
    namespace = {"PROJECT_INPUT": str(project), "EXPECTED_PACKAGE_HASHES": hashes,
                 "MODE": "smoke", "WORK_INPUT": str(tmp_path / "workflow"), "IN_COLAB": False,
                 "Path": Path, "hashlib": hashlib, "datetime": datetime, "os": os}
    compiled = compile(stage_source(notebook, "if PROJECT_INPUT:"), "<source bootstrap>", "exec")
    if changed:
        with pytest.raises(ValueError, match="Source package mismatch"):
            exec(compiled, namespace)
        assert not (tmp_path / "workflow").exists()
    else:
        exec(compiled, namespace)
        assert namespace["PROJECT_DIR"] == project


@pytest.mark.parametrize("state,download_required", [
    ("empty", True), ("interrupted", True), ("manual", False), ("complete", False),
])
def test_data_cell_recovers_missing_and_interrupted_acquisition(notebook, tmp_path, state, download_required):
    data_root = tmp_path / "data"
    shenzhen = data_root / "shenzhen"
    shenzhen.mkdir(parents=True)
    other_images = data_root / "montgomery/CXR_png"
    other_images.mkdir(parents=True)
    (other_images / "MCUCXR_0001_0.png").write_bytes(b"fixture")
    if state != "empty":
        image_directory = shenzhen / "CXR_png"
        image_directory.mkdir()
        (image_directory / "CHNCXR_0001_0.png").write_bytes(b"fixture")
    if state in {"interrupted", "complete"}:
        (shenzhen / "acquisition_receipts").mkdir()
    if state == "complete":
        (shenzhen / "acquisition_manifest.json").write_text("{}")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "data_validation.json").write_text("{}")
    calls = []
    namespace = {"MODE": "research", "DATA_ROOT": data_root, "PROJECT_DIR": tmp_path,
                 "PREPARED_DIR": prepared, "DOWNLOAD_IF_MISSING": True, "PATIENT_MAP_INPUT": "",
                 "cli": lambda *args, **kwargs: calls.append(args)}
    exec(compile(stage_source(notebook, 'stage="validate_and_prepare"'), "<data cell>", "exec"), namespace)
    downloads = [call for call in calls if call[0] == "download"]
    assert len(downloads) == int(download_required)
    assert calls[-1][0] == "prepare"


def test_software_evidence_is_bound_to_current_source_and_preserves_history(notebook, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    test_log = logs / "tests.log"
    test_log.write_text("All software checks passed", encoding="utf-8")
    suite = tmp_path / "suite"
    suite.mkdir()
    original = suite / "software_validation.json"
    original.write_text('{"source_hash": "old"}', encoding="utf-8")
    namespace = {"Path": Path, "tempfile": tempfile, "WORK_ROOT": tmp_path, "LOG_DIR": logs,
                 "SUITE_DIR": suite, "PIPELINE_PYTHON": sys.executable, "json": json,
                 "SOURCE_CODE_HASH": "reviewed-source", "PACKAGE_ID": "reviewed-release",
                 "run_command": lambda *args, **kwargs: {"exit_code": 0, "log_path": str(test_log)}}
    exec(compile(stage_source(notebook, '"software_tests_only"'), "<test cell>", "exec"), namespace)
    assert json.loads(original.read_text()) == {"source_hash": "old"}
    revised = json.loads((suite / "software_validation_reviewed-release.json").read_text())
    assert revised["source_hash"] == "reviewed-source"
    assert revised["exit_code"] == 0
