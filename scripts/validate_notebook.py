"""Execute every Colab cell with synthetic fixtures in an isolated, existing environment."""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--work-root", type=Path, help="Reuse a completed synthetic workflow to test Run all recovery")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    notebook = nbformat.read(root / "04_colab_improved_training.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for index, cell in enumerate(code_cells):
        ast.parse(cell.source, filename=f"notebook-code-{index}")
    kernel_directory = output / "jupyter" / "kernels" / "cxr-validation"
    kernel_directory.mkdir(parents=True)
    specification = {"argv": [str(arguments.kernel_python.resolve()), "-m", "ipykernel_launcher", "-f",
                               "{connection_file}"], "display_name": "CXR validation", "language": "python"}
    (kernel_directory / "kernel.json").write_text(json.dumps(specification), encoding="utf-8")
    environment = {"CXR_MODE": "smoke", "CXR_SKIP_INSTALL": "1", "CXR_PROJECT_DIR": str(root),
                   "CXR_WORK_ROOT": str(arguments.work_root.resolve() if arguments.work_root else output / "workflow"),
                   "JUPYTER_PATH": str(output / "jupyter"),
                   "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    for name in ("JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "JUPYTER_RUNTIME_DIR", "IPYTHONDIR",
                 "MPLCONFIGDIR", "TEMP", "TMP"):
        directory = output / "runtime" / name.lower()
        directory.mkdir(parents=True)
        environment[name] = str(directory)
    os.environ.update(environment)

    def report_cell(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Executing code cell at notebook position {cell_index + 1}", flush=True)

    client = NotebookClient(notebook, timeout=1200, kernel_name="cxr-validation",
                            resources={"metadata": {"path": str(root)}}, on_cell_start=report_cell)
    success = False
    try:
        client.execute()
        success = True
    finally:
        nbformat.write(notebook, output / "executed_synthetic_notebook.ipynb")
        errors = [{"cell": index + 1, "error": result.get("ename"), "message": result.get("evalue")}
                  for index, cell in enumerate(notebook.cells) for result in cell.get("outputs", [])
                  if result.output_type == "error"]
        status = {"success": success, "synthetic_only": True, "code_cells": len(code_cells),
                  "dependency_install_executed": False, "real_training_completed": False, "errors": errors}
        (output / "notebook_validation.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
