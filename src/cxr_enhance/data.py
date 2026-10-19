"""Restricted dataset acquisition, validation, provenance and fixed splits."""

import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Iterator
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .filters import normalize_image, read_image

LOGGER = logging.getLogger(__name__)
DATASET_INFO = {
    "shenzhen": {
        "prefix": "CHNCXR", "count": 662, "class_counts": {0: 326, 1: 336},
        "url": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/CXR_png/",
    },
    "montgomery": {
        "prefix": "MCUCXR", "count": 138, "class_counts": {0: 80, 1: 58},
        "url": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/",
    },
}
MANIFEST_FIELDS = ("dataset", "sample_id", "path", "label", "group_id", "split", "sha256", "pixel_sha256", "height", "width", "dtype")


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _acquisition_lock("prepared-metadata", path):
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to replace existing different artifact: {path}")
            return
        descriptor, temporary = tempfile.mkstemp(prefix=".prepare-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


@contextmanager
def _acquisition_lock(dataset: str, directory: Path) -> Iterator[None]:
    """Serialize writers on this runtime using an auto-released OS lock.

    The lock lives on local storage because Drive FUSE does not support every
    advisory locking primitive. Use only one Colab runtime per destination.
    """
    identity = hashlib.sha256(f"{dataset}:{os.path.normcase(str(directory.resolve()))}".encode()).hexdigest()
    lock_root = Path(tempfile.gettempdir()) / "cxr-enhance-acquisition-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f"{identity}.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another acquisition process is using {directory}; retry after it finishes") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _publish_download(temporary: Path, destination: Path) -> None:
    """Publish a complete file by same-filesystem rename under acquisition lock."""
    if destination.exists():
        if sha256_file(destination) != sha256_file(temporary):
            raise ValueError(f"Refusing to overwrite changed existing image or artifact: {destination.name}")
        return
    os.rename(temporary, destination)


def _publish_receipt(destination: Path, content: bytes, staging: Path) -> None:
    temporary = staging / (destination.name + ".receipt.part")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
    _publish_download(temporary, destination)
    temporary.unlink(missing_ok=True)


def _validate_response_url(response: object, expected: str) -> None:
    actual = response.geturl() if hasattr(response, "geturl") else expected
    if actual != expected:
        raise ValueError(f"Unexpected redirect from official source: {actual}")


def _image_directory(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory is missing: {root}")
    candidates = [root] if root.name == "CXR_png" else sorted(path for path in root.rglob("CXR_png") if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one CXR_png directory under {root}; found {len(candidates)}")
    directory = candidates[0].resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Dataset image directory escapes the supplied root")
    return directory


def _patient_groups(patient_map: str | Path | None) -> dict[tuple[str, str], str]:
    if patient_map is None:
        return {}
    with Path(patient_map).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"dataset", "sample_id", "patient_id"}.issubset(reader.fieldnames or []):
            raise ValueError("patient_map needs dataset,sample_id,patient_id columns")
        groups = {}
        for row in reader:
            dataset, sample_id, patient_id = (str(row.get(key) or "").strip() for key in ("dataset", "sample_id", "patient_id"))
            if dataset not in DATASET_INFO or not sample_id or not patient_id:
                raise ValueError("Invalid patient-map row")
            key = (dataset, sample_id)
            if key in groups:
                raise ValueError(f"Repeated sample in patient map: {key}")
            groups[key] = "patient:" + hashlib.sha256(patient_id.encode()).hexdigest()
    return groups


def _split_training(rows: list[dict], split_seed: int, val_fraction: float, grouped: bool) -> set[int]:
    labels = np.array([row["label"] for row in rows], dtype=int)
    indices = np.arange(len(rows))
    if set(labels.tolist()) != {0, 1} or np.bincount(labels).min() < 2:
        raise ValueError("Shenzhen needs at least two examples of each class for splitting")
    if not grouped:
        _, validation = train_test_split(indices, test_size=val_fraction, random_state=split_seed, stratify=labels)
        return set(validation.tolist())
    groups = np.array([row["group_id"] for row in rows])
    if len(set(groups)) < 4:
        raise ValueError("Patient-level splitting needs at least four independent groups")
    splitter = GroupShuffleSplit(n_splits=512, test_size=val_fraction, random_state=split_seed)
    candidates = []
    for training, validation in splitter.split(indices, labels, groups):
        if len(set(labels[training])) != 2 or len(set(labels[validation])) != 2:
            continue
        score = abs(len(validation) / len(rows) - val_fraction)
        score += sum(abs(np.sum(labels[validation] == label) / np.sum(labels == label) - val_fraction) for label in (0, 1))
        candidates.append((float(score), tuple(validation.tolist())))
    if not candidates:
        raise ValueError("Cannot form a group-disjoint train/validation split containing both classes")
    return set(min(candidates)[1])


def prepare_dataset(
    shenzhen_root: str | Path,
    montgomery_root: str | Path,
    output_dir: str | Path,
    split_seed: int = 2026,
    val_fraction: float = 0.2,
    allow_partial: bool = False,
    patient_map: str | Path | None = None,
) -> Path:
    """Validate images and save an immutable split manifest plus audit metadata.

    patient_map is an optional complete CSV keyed by dataset and filename stem;
    patient_id must use one shared identity namespace across the two datasets.
    Without it, group IDs are explicitly case IDs, not verified patient IDs.
    allow_partial is exclusively for synthetic smoke fixtures.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between zero and one")
    roots = {"shenzhen": Path(shenzhen_root).resolve(), "montgomery": Path(montgomery_root).resolve()}
    mapped = _patient_groups(patient_map)
    rows, hashes, pixels, observed = [], {}, {}, set()
    for dataset, root in roots.items():
        info = DATASET_INFO[dataset]
        directory = _image_directory(root)
        files = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".png")
        if not files or (not allow_partial and len(files) != info["count"]):
            raise ValueError(f"{dataset}: expected {info['count']} PNGs, found {len(files)}")
        pattern = re.compile(rf"{info['prefix']}_\d{{4}}_([01])\.png")
        case_ids = set()
        for path in files:
            match = pattern.fullmatch(path.name)
            if match is None or not path.resolve().is_relative_to(root):
                raise ValueError(f"Malformed image filename or escaped path: {path}")
            case_id = path.stem.rsplit("_", 1)[0]
            if case_id in case_ids:
                raise ValueError(f"Conflicting labels for case identifier: {case_id}")
            case_ids.add(case_id)
            image = read_image(path)
            normalize_image(image)
            file_hash = sha256_file(path)
            canonical = np.ascontiguousarray(image, dtype="<u4")
            pixel_hash = hashlib.sha256(str(image.shape).encode() + canonical.tobytes()).hexdigest()
            identity = f"{dataset}/{path.stem}"
            for digest, registry, kind in ((file_hash, hashes, "byte"), (pixel_hash, pixels, "pixel")):
                if digest in registry:
                    raise ValueError(f"Duplicate {kind} content: {registry[digest]} and {identity}; resolve before splitting")
                registry[digest] = identity
            key = (dataset, path.stem)
            observed.add(key)
            if patient_map is not None and key not in mapped:
                raise ValueError(f"Patient map has no identity for {identity}")
            rows.append({
                "dataset": dataset, "sample_id": path.stem, "path": path.relative_to(root).as_posix(),
                "label": int(match.group(1)), "group_id": mapped.get(key, f"{dataset}:case:{path.stem}"),
                "split": "test" if dataset == "montgomery" else "train", "sha256": file_hash,
                "pixel_sha256": pixel_hash, "height": image.shape[0], "width": image.shape[1], "dtype": str(image.dtype),
            })
    if patient_map is not None and set(mapped) != observed:
        raise ValueError("Patient map contains samples outside the validated image inventory")
    shenzhen = [row for row in rows if row["dataset"] == "shenzhen"]
    montgomery = [row for row in rows if row["dataset"] == "montgomery"]
    if not allow_partial:
        for dataset in DATASET_INFO:
            observed_counts = {label: sum(row["dataset"] == dataset and row["label"] == label for row in rows) for label in (0, 1)}
            if observed_counts != DATASET_INFO[dataset]["class_counts"]:
                raise ValueError(f"Unexpected class counts in {dataset}: {observed_counts}")
    if {row["label"] for row in montgomery} != {0, 1}:
        raise ValueError("Montgomery test data must include both labeled classes")
    if {row["group_id"] for row in shenzhen} & {row["group_id"] for row in montgomery}:
        raise ValueError("Patient overlap across Shenzhen and Montgomery")
    validation = _split_training(shenzhen, split_seed, val_fraction, patient_map is not None)
    for index, row in enumerate(shenzhen):
        row["split"] = "validation" if index in validation else "train"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    manifest = Path(output_dir) / "manifest.csv"
    report = {
        "schema_version": 1, "roots": {name: str(root) for name, root in roots.items()},
        "split_seed": split_seed, "validation_fraction_requested": val_fraction,
        "validation_fraction_actual": len(validation) / len(shenzhen), "synthetic_or_partial": allow_partial,
        "patient_grouping_verified_by_supplied_map": patient_map is not None,
        "grouping": "user-supplied patient map; independent provenance not verified" if patient_map is not None else "case IDs only; patient independence unverified",
        "split_algorithm": "best of 512 deterministic group splits by size/class-fraction distance" if patient_map is not None else "stratified train_test_split",
        "patient_map_sha256": sha256_file(patient_map) if patient_map is not None else None,
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "counts": {split: {str(label): sum(row["split"] == split and row["label"] == label for row in rows) for label in (0, 1)} for split in ("train", "validation", "test")},
        "duplicate_check": "No exact byte or decoded-pixel duplicates; transformed/near duplicates not ruled out",
        "test_policy": "Montgomery is excluded from training, model selection and threshold tuning",
    }
    artifacts = {manifest: content, Path(output_dir) / "data_validation.json": _json_bytes(report)}
    for path, payload in artifacts.items():
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace existing different artifact: {path}")
    for path, payload in artifacts.items():
        _immutable_bytes(path, payload)
    LOGGER.info("Prepared %s images; manifest %s", len(rows), manifest)
    return manifest


def validate_prepared_manifest(manifest_path: str | Path, config: dict) -> dict:
    """Verify that prepared evidence matches the experiment before any model use."""
    manifest = Path(manifest_path)
    report = json.loads((manifest.parent / "data_validation.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("Prepared manifest checksum or schema does not match its validation report")
    synthetic = config.get("synthetic", False)
    if type(synthetic) is not bool or type(report.get("synthetic_or_partial")) is not bool:
        raise ValueError("Synthetic preparation and experiment flags must be boolean")
    if report["synthetic_or_partial"] != synthetic:
        raise ValueError("Synthetic/partial preparation differs from the experiment synthetic flag")
    if type(report.get("split_seed")) is not int or report["split_seed"] != config.get("split_seed", 2026):
        raise ValueError("Prepared split seed differs from the experiment configuration")
    fraction = report.get("validation_fraction_requested")
    if not isinstance(fraction, (int, float)) or not math.isclose(fraction, 0.2, rel_tol=0, abs_tol=1e-12):
        raise ValueError("The approved development split requires validation_fraction_requested=0.2")
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(MANIFEST_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Prepared manifest is missing required schema columns")
        rows = list(reader)
    if not rows or {row["dataset"] for row in rows} != set(DATASET_INFO):
        raise ValueError("Prepared manifest requires both Shenzhen and Montgomery cohorts")
    identifiers, case_ids, file_hashes, pixel_hashes = set(), set(), set(), set()
    group_splits: dict[str, str] = {}
    counts = {split: {str(label): 0 for label in (0, 1)} for split in ("train", "validation", "test")}
    for row in rows:
        dataset, sample_id, label, split = (row[key] for key in ("dataset", "sample_id", "label", "split"))
        prefix = DATASET_INFO[dataset]["prefix"]
        if label not in {"0", "1"} or re.fullmatch(rf"{prefix}_\d{{4}}_{label}", sample_id) is None:
            raise ValueError("Prepared manifest has a malformed sample identifier or label")
        expected_splits = {"train", "validation"} if dataset == "shenzhen" else {"test"}
        if split not in expected_splits:
            raise ValueError("Prepared manifest violates the reserved external cohort split")
        identifier, case_id = (dataset, sample_id), (dataset, sample_id.rsplit("_", 1)[0])
        if identifier in identifiers or case_id in case_ids:
            raise ValueError("Prepared manifest repeats a sample or case identifier")
        identifiers.add(identifier)
        case_ids.add(case_id)
        relative = PurePosixPath(row["path"])
        if (relative.is_absolute() or ".." in relative.parts or "\\" in row["path"] or ":" in row["path"]
                or relative.name != sample_id + ".png"):
            raise ValueError("Prepared manifest has an unsafe or mismatched image path")
        for key, registry in (("sha256", file_hashes), ("pixel_sha256", pixel_hashes)):
            value = row[key]
            if re.fullmatch(r"[0-9a-f]{64}", value) is None or value in registry:
                raise ValueError(f"Prepared manifest has an invalid or duplicate {key}")
            registry.add(value)
        group = row["group_id"]
        if not group or group.strip() != group:
            raise ValueError("Prepared manifest has an empty or malformed group identifier")
        if group in group_splits and group_splits[group] != split:
            raise ValueError("Prepared manifest has a group crossing data splits")
        group_splits[group] = split
        try:
            dimensions_valid = int(row["height"]) >= 2 and int(row["width"]) >= 2
        except (TypeError, ValueError):
            dimensions_valid = False
        if not dimensions_valid or row["dtype"] not in {"uint8", "uint16"}:
            raise ValueError("Prepared manifest has invalid image dimensions or integer depth")
        counts[split][label] += 1
    if any(count == 0 for classes in counts.values() for count in classes.values()):
        raise ValueError("Every prepared split must contain both labeled classes")
    if counts != report.get("counts"):
        raise ValueError("Prepared manifest class counts differ from its validation report")
    actual_fraction = sum(counts["validation"].values()) / sum(row["dataset"] == "shenzhen" for row in rows)
    recorded_fraction = report.get("validation_fraction_actual")
    if (not isinstance(recorded_fraction, (int, float))
            or not math.isclose(actual_fraction, recorded_fraction, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Prepared validation fraction differs from its validation report")
    if not synthetic:
        for dataset, information in DATASET_INFO.items():
            observed = {label: sum(row["dataset"] == dataset and row["label"] == str(label) for row in rows)
                        for label in (0, 1)}
            if observed != information["class_counts"]:
                raise ValueError(f"Research manifest requires the complete {dataset} cohort")
    return report


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def download_official(dataset: str, destination: str | Path) -> Path:
    """Acquire only official CXR PNGs, with four workers and immutable receipts.

    Existing files with receipts are hash-verified locally; receipt-less files
    are compared to a fresh official download. Files are never replaced on a
    mismatch. Local SHA-256 receipts are not publisher-provided checksums.
    Dataset terms must be reviewed by the caller before this function is used.
    Local storage and a mounted Google Drive are supported. Complete files are
    published by same-filesystem rename under an OS process lock; use only one
    Colab runtime per destination. Completed files resume from their receipts.
    Interrupted individual images are re-downloaded, not resumed by byte range.
    Only this downloader's private staging files may be removed on recovery.
    """
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset}")
    info = DATASET_INFO[dataset]
    base_url = str(info["url"])
    request = Request(base_url + "index.html", headers={"User-Agent": "cxr-enhance-research/0.1"})
    with urlopen(request, timeout=60) as response:
        _validate_response_url(response, request.full_url)
        index = response.read(8 * 1024 * 1024 + 1)
    if len(index) > 8 * 1024 * 1024:
        raise ValueError("Official index is unexpectedly large")
    parser = _IndexParser()
    parser.feed(index.decode("utf-8"))
    pattern = re.compile(rf"{info['prefix']}_\d{{4}}_[01]\.png")
    filenames = set()
    for link in parser.links:
        resolved = urlparse(urljoin(base_url, link))
        filename = unquote(Path(resolved.path).name)
        if pattern.fullmatch(filename):
            expected = urlparse(base_url + filename)
            if resolved != expected:
                raise ValueError("Unexpected noncanonical image link in official index")
            filenames.add(filename)
    if len(filenames) != info["count"]:
        raise ValueError(f"Official {dataset} index contains {len(filenames)} images; expected {info['count']}")
    target = Path(destination).resolve()
    directory = target if target.name == "CXR_png" else target / "CXR_png"
    receipt_dir = directory.parent / "acquisition_receipts"
    staging = directory.parent / ".cxr_acquisition" / dataset

    def acquire(filename: str) -> dict:
        path = directory / filename
        receipt_path = receipt_dir / (filename + ".json")
        source_url = base_url + filename
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("url") != source_url or receipt.get("filename") != filename
                    or not path.is_file() or sha256_file(path) != receipt.get("sha256")
                    or path.stat().st_size != receipt.get("bytes")):
                raise ValueError(f"Existing download differs from its acquisition receipt: {filename}")
            normalize_image(read_image(path))
            return receipt
        temporary_path = staging / (filename + ".download.part")
        with temporary_path.open("wb") as temporary:
            try:
                with urlopen(Request(source_url, headers={"User-Agent": "cxr-enhance-research/0.1"}), timeout=120) as response:
                    _validate_response_url(response, source_url)
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 128 * 1024 * 1024:
                            raise ValueError(f"Official image is unexpectedly large: {filename}")
                        temporary.write(chunk)
                    temporary.flush()
            except Exception:
                temporary.close()
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            normalize_image(read_image(temporary_path))
            digest = sha256_file(temporary_path)
            receipt = {"filename": filename, "url": source_url, "sha256": digest, "bytes": temporary_path.stat().st_size}
            _publish_download(temporary_path, path)
            _publish_receipt(receipt_path, _json_bytes(receipt), staging)
            LOGGER.info("Verified download %s", filename)
            return receipt
        finally:
            temporary_path.unlink(missing_ok=True)

    with _acquisition_lock(dataset, directory):
        for owned in (directory, receipt_dir, staging):
            if not owned.resolve().is_relative_to(directory.parent.resolve()):
                raise ValueError("Acquisition directory escapes the supplied destination")
        directory.mkdir(parents=True, exist_ok=True)
        receipt_dir.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=True, exist_ok=True)
        pending_names = {filename + suffix for filename in filenames for suffix in (".download.part", ".json.receipt.part")}
        pending_names.add("acquisition_manifest.json.receipt.part")
        for temporary in staging.glob("*.part"):
            if temporary.name not in pending_names or not temporary.is_file() or temporary.is_symlink():
                raise ValueError("Unexpected entry in private acquisition staging directory")
            temporary.unlink()
        with ThreadPoolExecutor(max_workers=4) as executor:
            receipts = list(executor.map(acquire, sorted(filenames)))
        manifest = {"dataset": dataset, "index_url": base_url + "index.html", "checksum_source": "locally measured; not publisher supplied", "files": receipts}
        _publish_receipt(directory.parent / "acquisition_manifest.json", _json_bytes(manifest), staging)
    return directory.parent
