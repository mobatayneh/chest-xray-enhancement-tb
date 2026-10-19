"""Small integrity and serialization helpers."""

import hashlib
import json
import os
import platform
import sys
import uuid
from importlib import metadata
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: str | Path, value: object, overwrite: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not overwrite:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        return
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, destination)


def source_hash() -> str:
    root = Path(__file__).parent
    return canonical_hash({path.name: sha256_file(path) for path in sorted(root.glob("*.py"))})


def environment_info() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "opencv-python-headless", "matplotlib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    installed = {distribution.metadata["Name"]: distribution.version for distribution in metadata.distributions()
                 if distribution.metadata["Name"]}
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "installed_packages": dict(sorted(installed.items(), key=lambda item: item[0].lower()))}
