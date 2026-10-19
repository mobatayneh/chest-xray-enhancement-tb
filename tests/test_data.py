import csv
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cxr_enhance import data


def make_sources(tmp_path: Path) -> tuple[Path, Path]:
    roots = (tmp_path / "shenzhen", tmp_path / "montgomery")
    generator = np.random.default_rng(2026)
    for root, prefix, count in zip(roots, ("CHNCXR", "MCUCXR"), (20, 8)):
        directory = root / "CXR_png"
        directory.mkdir(parents=True)
        for number in range(1, count + 1):
            values = generator.integers(0, 65536, (16, 20), dtype=np.uint16)
            assert cv2.imwrite(str(directory / f"{prefix}_{number:04d}_{number % 2}.png"), values)
    return roots


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_fixed_stratified_splits_and_mask_exclusion(tmp_path: Path) -> None:
    shenzhen, montgomery = make_sources(tmp_path)
    mask_folder = montgomery / "ManualMask" / "leftMask"
    mask_folder.mkdir(parents=True)
    assert cv2.imwrite(str(mask_folder / "MCUCXR_0001_1.png"), np.zeros((16, 20), dtype=np.uint8))
    first = data.prepare_dataset(shenzhen, montgomery, tmp_path / "first", allow_partial=True)
    second = data.prepare_dataset(shenzhen, montgomery, tmp_path / "second", allow_partial=True)
    assert first.read_bytes() == second.read_bytes()
    assert data.prepare_dataset(shenzhen, montgomery, tmp_path / "first", allow_partial=True) == first
    rows = read_rows(first)
    assert len(rows) == 28
    assert sum(row["split"] == "validation" for row in rows) == 4
    assert all(row["split"] == "test" for row in rows if row["dataset"] == "montgomery")
    assert {row["label"] for row in rows if row["split"] == "validation"} == {"0", "1"}
    groups = {split: {row["group_id"] for row in rows if row["split"] == split} for split in ("train", "validation", "test")}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    report = json.loads(first.with_name("data_validation.json").read_text())
    assert not report["patient_grouping_verified_by_supplied_map"]
    assert report["manifest_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert all(not Path(row["path"]).is_absolute() for row in rows)


def test_cxr_root_and_exact_counts(tmp_path: Path) -> None:
    shenzhen, montgomery = make_sources(tmp_path)
    with pytest.raises(ValueError, match="expected 662"):
        data.prepare_dataset(shenzhen, montgomery, tmp_path / "strict")
    manifest = data.prepare_dataset(shenzhen / "CXR_png", montgomery / "CXR_png", tmp_path / "partial", allow_partial=True)
    assert all("/" not in row["path"] for row in read_rows(manifest))


@pytest.mark.parametrize("problem", ["filename", "corrupt", "constant", "byte_duplicate", "pixel_duplicate"])
def test_rejects_invalid_inputs_and_cross_dataset_duplicates(tmp_path: Path, problem: str) -> None:
    shenzhen, montgomery = make_sources(tmp_path)
    target = montgomery / "CXR_png" / "MCUCXR_0001_1.png"
    source = shenzhen / "CXR_png" / "CHNCXR_0002_0.png"
    if problem == "filename":
        target.rename(target.with_name("MCUCXR_0001_2.png"))
        expected = "Malformed"
    elif problem == "corrupt":
        target.write_bytes(b"invalid PNG")
        expected = "decode"
    elif problem == "constant":
        assert cv2.imwrite(str(target), np.ones((16, 20), dtype=np.uint16))
        expected = "Constant"
    elif problem == "byte_duplicate":
        target.write_bytes(source.read_bytes())
        expected = "Duplicate byte"
    else:
        assert cv2.imwrite(str(target), data.read_image(source), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        expected = "Duplicate pixel"
    with pytest.raises(ValueError, match=expected):
        data.prepare_dataset(shenzhen, montgomery, tmp_path / "rejected", allow_partial=True)
    assert not (tmp_path / "rejected" / "manifest.csv").exists()


def test_existing_manifest_cannot_change(tmp_path: Path) -> None:
    shenzhen, montgomery = make_sources(tmp_path)
    manifest = data.prepare_dataset(shenzhen, montgomery, tmp_path / "prepared", allow_partial=True)
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="Refusing"):
        data.prepare_dataset(shenzhen, montgomery, manifest.parent, split_seed=99, allow_partial=True)
    assert manifest.read_bytes() == original


def test_replicated_grayscale_encoding_cannot_hide_cross_dataset_duplicates(tmp_path: Path) -> None:
    shenzhen, montgomery = make_sources(tmp_path)
    source = shenzhen / "CXR_png" / "CHNCXR_0002_0.png"
    target = montgomery / "CXR_png" / "MCUCXR_0001_1.png"
    original = data.read_image(source)
    replicated = np.repeat(original[:, :, None], 3, axis=2)
    assert cv2.imwrite(str(target), replicated)
    assert data.sha256_file(source) != data.sha256_file(target)
    with pytest.raises(ValueError, match="Duplicate pixel"):
        data.prepare_dataset(shenzhen, montgomery, tmp_path / "rejected", allow_partial=True)


def test_case_identifier_cannot_have_conflicting_labels(tmp_path: Path) -> None:
    shenzhen, montgomery = make_sources(tmp_path)
    values = np.random.default_rng(90).integers(0, 65536, (16, 20), dtype=np.uint16)
    assert cv2.imwrite(str(shenzhen / "CXR_png" / "CHNCXR_0001_0.png"), values)
    with pytest.raises(ValueError, match="Conflicting labels"):
        data.prepare_dataset(shenzhen, montgomery, tmp_path / "rejected", allow_partial=True)


def write_patient_map(shenzhen: Path, montgomery: Path, destination: Path, overlap: bool = False) -> None:
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "sample_id", "patient_id"))
        writer.writeheader()
        for dataset, root in (("shenzhen", shenzhen), ("montgomery", montgomery)):
            for index, path in enumerate(sorted((root / "CXR_png").glob("*.png"))):
                identity = f"{dataset}-patient-{index // 2}"
                if overlap and dataset == "montgomery" and index == 0:
                    identity = "shenzhen-patient-0"
                writer.writerow({"dataset": dataset, "sample_id": path.stem, "patient_id": identity})


def test_grouped_split_and_cross_dataset_overlap(tmp_path: Path) -> None:
    shenzhen, montgomery = make_sources(tmp_path)
    patient_map = tmp_path / "patients.csv"
    write_patient_map(shenzhen, montgomery, patient_map)
    manifest = data.prepare_dataset(shenzhen, montgomery, tmp_path / "grouped", allow_partial=True, patient_map=patient_map)
    rows = read_rows(manifest)
    training = {row["group_id"] for row in rows if row["split"] == "train"}
    validation = {row["group_id"] for row in rows if row["split"] == "validation"}
    assert not training & validation
    assert len(validation) == 2
    write_patient_map(shenzhen, montgomery, patient_map, overlap=True)
    with pytest.raises(ValueError, match="Patient overlap"):
        data.prepare_dataset(shenzhen, montgomery, tmp_path / "overlap", allow_partial=True, patient_map=patient_map)


def test_incomplete_patient_map_rejected(tmp_path: Path) -> None:
    shenzhen, montgomery = make_sources(tmp_path)
    patient_map = tmp_path / "patients.csv"
    patient_map.write_text("dataset,sample_id,patient_id\nshenzhen,CHNCXR_0001_1,p1\n")
    with pytest.raises(ValueError, match="no identity"):
        data.prepare_dataset(shenzhen, montgomery, tmp_path / "grouped", allow_partial=True, patient_map=patient_map)


def test_official_download_reuse_and_tamper_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(data.DATASET_INFO, "shenzhen", {**data.DATASET_INFO["shenzhen"], "count": 2})
    names = ["CHNCXR_0001_0.png", "CHNCXR_0002_1.png"]
    page = "".join(f'<a href="{name}">{name}</a>' for name in names).encode()
    images = {name: cv2.imencode(".png", np.arange(100, dtype=np.uint8).reshape(10, 10) + index)[1].tobytes() for index, name in enumerate(names)}
    calls = []

    def fake_open(request: object, timeout: int) -> io.BytesIO:
        url = request.full_url
        calls.append(url)
        return io.BytesIO(page if url.endswith("index.html") else images[Path(url).name])

    monkeypatch.setattr(data, "urlopen", fake_open)
    root = data.download_official("shenzhen", tmp_path / "download")
    assert len(calls) == 3
    data.download_official("shenzhen", root)
    assert len(calls) == 4
    assert (root / "acquisition_manifest.json").exists()
    (root / "CXR_png" / names[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        data.download_official("shenzhen", root)


def test_official_download_refuses_unverified_changed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(data.DATASET_INFO, "shenzhen", {**data.DATASET_INFO["shenzhen"], "count": 1})
    filename = "CHNCXR_0001_0.png"
    page = f'<a href="{filename}">{filename}</a>'.encode()
    encoded = cv2.imencode(".png", np.eye(8, dtype=np.uint8) * 255)[1].tobytes()
    monkeypatch.setattr(data, "urlopen", lambda request, timeout: io.BytesIO(page if request.full_url.endswith("index.html") else encoded))
    directory = tmp_path / "download" / "CXR_png"
    directory.mkdir(parents=True)
    existing = directory / filename
    existing.write_bytes(b"previous data")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        data.download_official("shenzhen", directory.parent)
    assert existing.read_bytes() == b"previous data"
    assert not list(directory.glob("*.part"))


def mock_official_source(monkeypatch: pytest.MonkeyPatch, image: bytes | None = None, link: str | None = None) -> tuple[str, bytes, list[str]]:
    filename = "CHNCXR_0001_0.png"
    encoded = image if image is not None else cv2.imencode(".png", np.eye(8, dtype=np.uint8) * 255)[1].tobytes()
    page = f'<a href="{link or filename}">{filename}</a>'.encode()
    calls = []
    monkeypatch.setitem(data.DATASET_INFO, "shenzhen", {**data.DATASET_INFO["shenzhen"], "count": 1})

    def fake_open(request: object, timeout: int) -> io.BytesIO:
        calls.append(request.full_url)
        return io.BytesIO(page if request.full_url.endswith("index.html") else encoded)

    monkeypatch.setattr(data, "urlopen", fake_open)
    return filename, encoded, calls


def test_download_accepts_official_replicated_grayscale_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grayscale = np.arange(100, dtype=np.uint8).reshape(10, 10)
    encoded = cv2.imencode(".png", np.repeat(grayscale[:, :, None], 3, axis=2))[1].tobytes()
    filename, _, calls = mock_official_source(monkeypatch, image=encoded)
    root = data.download_official("shenzhen", tmp_path / "download")
    saved = root / "CXR_png" / filename
    assert saved.read_bytes() == encoded
    np.testing.assert_array_equal(data.read_image(saved), grayscale)
    data.download_official("shenzhen", root)
    assert len(calls) == 3


@pytest.mark.parametrize("link", [
    "https://unexpected.example/CHNCXR_0001_0.png", "../CHNCXR_0001_0.png",
    "CHNCXR_0001_0.png?download=true", "CHNCXR_0001_0.png#fragment",
    "%2E%2E/CHNCXR_0001_0.png", "//unexpected.example/CHNCXR_0001_0.png",
])
def test_download_rejects_noncanonical_index_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link: str) -> None:
    _, _, calls = mock_official_source(monkeypatch, link=link)
    with pytest.raises(ValueError, match="noncanonical"):
        data.download_official("shenzhen", tmp_path / "download")
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [b"not a png", b"<html>error</html>"])
def test_download_rejects_corrupt_payload_without_publishing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    filename, _, _ = mock_official_source(monkeypatch, image=payload)
    with pytest.raises(ValueError, match="decode"):
        data.download_official("shenzhen", tmp_path / "download")
    assert not (tmp_path / "download" / "CXR_png" / filename).exists()
    assert not list((tmp_path / "download").rglob("*.part"))
    assert not list((tmp_path / "download" / "acquisition_receipts").glob("*.json"))


def test_download_rejects_redirects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_official_source(monkeypatch)

    class Redirected(io.BytesIO):
        def geturl(self) -> str:
            return "https://unexpected.example/index.html"

    monkeypatch.setattr(data, "urlopen", lambda request, timeout: Redirected(b"index"))
    with pytest.raises(ValueError, match="redirect"):
        data.download_official("shenzhen", tmp_path / "download")


def test_download_requires_expected_index_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_official_source(monkeypatch)
    monkeypatch.setitem(data.DATASET_INFO, "shenzhen", {**data.DATASET_INFO["shenzhen"], "count": 2})
    with pytest.raises(ValueError, match="expected 2"):
        data.download_official("shenzhen", tmp_path / "download")


def test_download_recovers_own_partial_without_hardlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filename, encoded, _ = mock_official_source(monkeypatch)
    root = tmp_path / "download"
    staging = root / ".cxr_acquisition" / "shenzhen"
    staging.mkdir(parents=True)
    (staging / (filename + ".download.part")).write_bytes(b"interrupted")
    outside = root / "user.part"
    outside.write_bytes(b"preserve")

    def forbidden_link(*args: object) -> None:
        raise AssertionError("Hard links must not be required on Drive FUSE")

    monkeypatch.setattr(data.os, "link", forbidden_link)
    data.download_official("shenzhen", root)
    assert (root / "CXR_png" / filename).read_bytes() == encoded
    assert outside.read_bytes() == b"preserve"
    assert not list(staging.glob("*.part"))


def test_download_recovers_publication_before_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filename, encoded, calls = mock_official_source(monkeypatch)
    root = tmp_path / "download"
    publish = data._publish_receipt

    def interrupt_receipt(*args: object) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(data, "_publish_receipt", interrupt_receipt)
    with pytest.raises(OSError, match="interruption"):
        data.download_official("shenzhen", root)
    assert (root / "CXR_png" / filename).read_bytes() == encoded
    monkeypatch.setattr(data, "_publish_receipt", publish)
    data.download_official("shenzhen", root)
    assert len(calls) == 4
    assert (root / "acquisition_receipts" / (filename + ".json")).is_file()
    data.download_official("shenzhen", root)
    assert len(calls) == 5


def test_download_process_lock_releases_after_failure(tmp_path: Path) -> None:
    directory = tmp_path / "CXR_png"
    with data._acquisition_lock("shenzhen", directory):
        with pytest.raises(RuntimeError, match="Another acquisition"):
            with data._acquisition_lock("shenzhen", directory):
                pytest.fail("Concurrent acquisition entered the protected section")
    with data._acquisition_lock("shenzhen", directory):
        assert True


def test_download_keeps_unknown_staging_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_official_source(monkeypatch)
    staging = tmp_path / "download" / ".cxr_acquisition" / "shenzhen"
    staging.mkdir(parents=True)
    unknown = staging / "user.part"
    unknown.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="Unexpected entry"):
        data.download_official("shenzhen", tmp_path / "download")
    assert unknown.read_bytes() == b"preserve"


def test_download_recovers_network_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filename, encoded, _ = mock_official_source(monkeypatch)
    original_open = data.urlopen

    class InterruptedImage(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            if self.tell():
                raise OSError("network interrupted")
            return super().read(16)

    def interrupted_open(request: object, timeout: int) -> io.BytesIO:
        if request.full_url.endswith("index.html"):
            return original_open(request, timeout)
        return InterruptedImage(encoded)

    monkeypatch.setattr(data, "urlopen", interrupted_open)
    root = tmp_path / "download"
    with pytest.raises(OSError, match="network interrupted"):
        data.download_official("shenzhen", root)
    assert not (root / "CXR_png" / filename).exists()
    assert not list(root.rglob("*.part"))
    monkeypatch.setattr(data, "urlopen", original_open)
    data.download_official("shenzhen", root)
    assert (root / "CXR_png" / filename).read_bytes() == encoded
