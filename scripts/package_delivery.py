"""Package only reviewed source and documentation; exclude private data and checkpoints."""

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = arguments.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "chest-xray-enhancement-tb.zip"
    notebook_path = destination / "04_colab_improved_training.ipynb"
    checksums_path = destination / "SHA256SUMS.txt"
    if any(path.exists() for path in (archive_path, notebook_path, checksums_path)):
        raise FileExistsError("Delivery files already exist; choose another output directory.")
    directories = {"src", "configs", "docs", "provenance", "scripts", "tests", "validation"}
    extensions = {".py", ".md", ".toml", ".txt", ".json", ".csv", ".ipynb", ".log"}
    members = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" or part.endswith(".egg-info")
               for part in relative.parts[:-1]):
            continue
        if len(relative.parts) > 1 and relative.parts[0] not in directories:
            continue
        if path.suffix not in extensions and relative.as_posix() != ".gitignore":
            continue
        members[relative.as_posix()] = path.read_bytes()
    if "04_colab_improved_training.ipynb" not in members or "docs/VALIDATION.md" not in members:
        raise ValueError("Notebook and actual validation report are required before packaging.")
    source_checksums = "".join(f"{digest(content)}  {name}\n" for name, content in sorted(members.items()))
    members["SOURCE_SHA256SUMS.txt"] = source_checksums.encode()
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            entry = zipfile.ZipInfo(f"chest-xray-enhancement-tb/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, content)
    shutil.copyfile(root / "04_colab_improved_training.ipynb", notebook_path)
    checksums_path.write_text("".join(f"{digest(path.read_bytes())}  {path.name}\n"
                                    for path in (archive_path, notebook_path)), encoding="utf-8")
    with zipfile.ZipFile(archive_path) as archive:
        for name, content in members.items():
            if archive.read(f"chest-xray-enhancement-tb/{name}") != content:
                raise ValueError(f"Delivery content verification failed: {name}")
    print(f"Packaged and verified {len(members)} source/documentation files: {archive_path}")


if __name__ == "__main__":
    main()
