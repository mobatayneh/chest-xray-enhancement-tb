from pathlib import Path
import json

import pytest
import torch

from cxr_enhance.checkpoints import (
    canonical_hash,
    latest_checkpoint,
    local_lock,
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
    validate_contract,
    write_json,
)


def test_immutable_checkpoint_and_checksum(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, 1, {"epoch": 1, "tensor": torch.tensor([1.0])})
    checkpoint = latest_checkpoint(tmp_path)
    assert load_checkpoint(checkpoint)["tensor"].item() == 1
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path, 1, {"epoch": 1, "tensor": torch.tensor([2.0])})
    with checkpoint.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(checkpoint)


def test_atomic_json_no_overwrite_and_contract(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_json(path, {"value": 1}, overwrite=False)
    with pytest.raises(FileExistsError):
        write_json(path, {"value": 2}, overwrite=False)
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    validate_contract({"seed": 42}, {"seed": 42})
    with pytest.raises(ValueError, match="seed"):
        validate_contract({"seed": 42}, {"seed": 43})


def test_restores_all_cpu_rng_streams() -> None:
    import random
    import numpy as np

    state = rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    restore_rng(state)
    assert random.random() == expected[0]
    assert float(np.random.random()) == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_orphan_sidecar_is_preserved_and_does_not_block_retry(tmp_path: Path) -> None:
    orphan = tmp_path / "checkpoints" / "epoch_0001.json"
    original_record = {"epoch": 1, "path": "epoch_0001.pt", "sha256": "interrupted", "bytes": 10}
    write_json(orphan, original_record, overwrite=False)
    save_checkpoint(tmp_path, 1, {"epoch": 1, "tensor": torch.tensor([2.0])})
    preserved = list((tmp_path / "checkpoints" / "orphans").glob("*.json"))
    assert len(preserved) == 1
    assert json.loads(preserved[0].read_text()) == original_record
    assert load_checkpoint(latest_checkpoint(tmp_path))["tensor"].item() == 2


def test_checkpoint_rejects_mismatched_epoch_without_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epoch"):
        save_checkpoint(tmp_path, 1, {"epoch": 2})
    assert not (tmp_path / "checkpoints").exists()


def test_checkpoint_refuses_another_writer(tmp_path: Path) -> None:
    from filelock import Timeout

    with local_lock(tmp_path, "checkpoint-transaction"):
        with pytest.raises(Timeout):
            save_checkpoint(tmp_path, 1, {"epoch": 1})
    save_checkpoint(tmp_path, 1, {"epoch": 1})
    assert load_checkpoint(latest_checkpoint(tmp_path))["epoch"] == 1


def test_immutable_publication_without_hardlinks(tmp_path: Path, monkeypatch) -> None:
    import cxr_enhance.checkpoints as checkpoints

    def unsupported_link(*args, **kwargs):
        raise OSError("Hard links unavailable on mounted storage")

    monkeypatch.setattr(checkpoints.os, "link", unsupported_link)
    save_checkpoint(tmp_path, 1, {"epoch": 1, "tensor": torch.tensor([1.0])})
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path, 1, {"epoch": 1, "tensor": torch.tensor([2.0])})
    assert load_checkpoint(latest_checkpoint(tmp_path))["tensor"].item() == 1
