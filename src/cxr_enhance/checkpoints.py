"""Immutable epoch checkpoints with explicit integrity and resume contracts."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock


def local_lock(path: Path, purpose: str) -> FileLock:
    """Coordinate processes on one runtime; separate Colab VMs must not share a run destination."""
    identifier = hashlib.sha256(f"{purpose}:{path.resolve()}".encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / "cxr-enhance-locks"
    directory.mkdir(parents=True, exist_ok=True)
    return FileLock(directory / f"{identifier}.lock", timeout=0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    source_dir = Path(__file__).parent
    for source in sorted(source_dir.rglob("*.py")):
        digest.update(source.relative_to(source_dir).as_posix().encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def write_json(path: str | Path, value: Any, overwrite: bool = True) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            _publish_immutable(Path(temporary), destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _publish_immutable(temporary: Path, destination: Path) -> None:
    with local_lock(destination, "publication"):
        _publish_locked(temporary, destination)


def _publish_locked(temporary: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite verified artifact: {destination}")
    try:
        os.link(temporary, destination)
        temporary.unlink()
    except FileExistsError:
        raise
    except OSError:
        if destination.exists():
            raise FileExistsError(f"Destination appeared during publication: {destination}")
        temporary.rename(destination)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(run_dir: Path, epoch: int, state: dict[str, Any]) -> dict[str, Any]:
    with local_lock(Path(run_dir), "checkpoint-transaction"):
        return _save_checkpoint(run_dir, epoch, state)


def _save_checkpoint(run_dir: Path, epoch: int, state: dict[str, Any]) -> dict[str, Any]:
    if type(epoch) is not int or epoch < 1 or state.get("epoch") != epoch:
        raise ValueError("Checkpoint epoch must be a positive integer matching state['epoch']")
    directory = Path(run_dir) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epoch_{epoch:04d}.pt"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    record_path = destination.with_suffix(".json")
    if record_path.exists():
        orphan_directory = directory / "orphans"
        orphan_directory.mkdir(exist_ok=True)
        record_path.rename(orphan_directory / f"{record_path.stem}-{uuid.uuid4().hex}.json")
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".pt", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = sha256_file(temporary)
        record = {"epoch": epoch, "path": destination.name, "sha256": checksum,
                  "bytes": Path(temporary).stat().st_size}
        write_json(record_path, record, overwrite=False)
        _publish_immutable(Path(temporary), destination)
    except Exception:
        if not destination.exists() and record_path.exists():
            orphan_directory = directory / "orphans"
            orphan_directory.mkdir(exist_ok=True)
            record_path.rename(orphan_directory / f"{record_path.stem}-{uuid.uuid4().hex}.json")
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
    return record


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a trusted, locally generated checkpoint; PyTorch pickle is not a safe interchange format."""
    path = Path(path)
    record_path = path.with_suffix(".json")
    if not record_path.exists():
        raise ValueError(f"Missing checkpoint integrity record: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("path") != path.name:
        raise ValueError(f"Checkpoint integrity record does not match file: {path}")
    if record["sha256"] != sha256_file(path) or record.get("bytes") != path.stat().st_size:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("epoch") != record["epoch"]:
        raise ValueError("Checkpoint epoch disagrees with integrity record")
    return state


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((Path(run_dir) / "checkpoints").glob("epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def validate_contract(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    differences = sorted(key for key in set(actual) | set(expected)
                         if actual.get(key) != expected.get(key))
    if differences:
        raise ValueError("Resume contract mismatch: " + ", ".join(differences))
