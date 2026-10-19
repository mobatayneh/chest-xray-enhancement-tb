"""Inventory original artifacts without executing notebook code or copying source data."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_text(cell: dict[str, Any]) -> str:
    return "\n".join(
        "".join(output.get("text", []))
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def audit_notebook(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    matrices = []
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+).*?Loss[: ]+([\d.]+).*?Acc[: ]+([\d.]+).*?AUC[: ]+([\d.]+)"
    )
    matrix_pattern = re.compile(r"\[\[\s*(\d+)\s+(\d+)\s*\]\s*\[\s*(\d+)\s+(\d+)\s*\]\]")
    final_pattern = re.compile(r"(?:Final\s+)?Acc(?:uracy)?\s*:\s*([\d.]+)")
    for cell_number, cell in enumerate(notebook["cells"], 1):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or not source.strip():
            continue
        try:
            ast.parse(source)
            syntax = "PASS"
        except SyntaxError as error:
            syntax = f"FAIL: {error.msg}, line {error.lineno}"
        text = output_text(cell)
        epochs = [
            {"epoch": int(match[0]), "planned_epochs": int(match[1]), "train_loss_printed": float(match[2]),
             "validation_accuracy_printed": float(match[3]), "validation_auc_printed": float(match[4])}
            for match in epoch_pattern.findall(text)
        ]
        cells.append({
            "cell": cell_number, "syntax": syntax, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "models": sorted(set(re.findall(r"efficientnet_v2_[sm]", source))),
            "bce_with_logits_present": "BCEWithLogitsLoss" in source,
            "has_seed_call": bool(re.search(r"manual_seed|random\.seed|np\.random\.seed", source)),
            "checkpoint_name_reused": "best_model.pth" in source,
            "epochs": epochs,
            "stored_errors": [output.get("ename", "unknown") for output in cell.get("outputs", [])
                              if output.get("output_type") == "error"],
        })
        for matrix_number, match in enumerate(matrix_pattern.finditer(text), 1):
            true_negative, false_positive, false_negative, true_positive = map(int, match.groups())
            count = true_negative + false_positive + false_negative + true_positive
            printed = final_pattern.findall(text[:match.start()])
            printed_accuracy = float(printed[-1]) if printed else None
            computed = (true_negative + true_positive) / count
            matrices.append({
                "notebook": path.name, "cell": cell_number, "matrix_in_cell": matrix_number,
                "tn": true_negative, "fp": false_positive, "fn": false_negative, "tp": true_positive,
                "sample_count": count, "accuracy_recomputed": computed,
                "accuracy_printed": printed_accuracy,
                "accuracy_matches_printed_4dp": printed_accuracy is not None and abs(computed - printed_accuracy) <= 0.00005,
                "tb_sensitivity_recomputed": true_positive / (true_positive + false_negative),
                "specificity_recomputed": true_negative / (true_negative + false_positive),
                "evidence_type": "saved notebook output; not rerun or independent test",
            })
    return {"notebook": path.name, "code_cells": cells}, matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve(strict=True)
    destination = args.output_dir.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Evidence output must be outside the originals directory")
    filenames = [path for path in sorted(source.iterdir(), key=lambda item: item.name.casefold())
                 if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".pptx", ".ipynb"}]
    inventory = [{"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in filenames]
    hashes: dict[str, list[str]] = {}
    for item in inventory:
        hashes.setdefault(item["sha256"], []).append(item["filename"])
    notebooks, matrices = [], []
    for path in filenames:
        if path.suffix == ".ipynb":
            notebook, notebook_matrices = audit_notebook(path)
            notebooks.append(notebook)
            matrices.extend(notebook_matrices)
    report = {
        "interpretation": "Static syntax and recorded-output audit only. No original code or training executed.",
        "cell_numbering": "1-based notebook cell index, including markdown and blank cells",
        "notebooks": notebooks,
        "summary": {"nonempty_code_cells": sum(len(item["code_cells"]) for item in notebooks),
                    "syntax_passes": sum(cell["syntax"] == "PASS" for item in notebooks for cell in item["code_cells"]),
                    "matrices": len(matrices),
                    "accuracy_matches": sum(item["accuracy_matches_printed_4dp"] for item in matrices)},
    }
    artifacts = {
        "source_manifest.json": {"files": inventory, "identical_file_groups": [names for names in hashes.values() if len(names) > 1]},
        "notebook_audit.json": report,
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        path = destination / name
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    with (destination / "historical_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        if matrices:
            writer = csv.DictWriter(stream, fieldnames=list(matrices[0]))
            writer.writeheader()
            writer.writerows(matrices)
    print(json.dumps({"files": len(inventory), **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
