import csv
import io
import json

import pytest

from cxr_enhance.data import prepare_dataset, sha256_file, validate_prepared_manifest
from cxr_enhance.smoke import create_synthetic_data, smoke_config


def test_interrupted_metadata_publication_leaves_no_partial_artifact(tmp_path, monkeypatch):
    import cxr_enhance.data as data

    path = tmp_path / "manifest.csv"

    def interruption(*args):
        raise OSError("Simulated interruption before metadata publication")

    with monkeypatch.context() as patch:
        patch.setattr(data.os, "rename", interruption)
        with pytest.raises(OSError, match="interruption"):
            data._immutable_bytes(path, b"complete metadata")
    assert not path.exists()
    data._immutable_bytes(path, b"complete metadata")
    assert path.read_bytes() == b"complete metadata"


@pytest.fixture
def prepared_manifest(tmp_path):
    roots = create_synthetic_data(tmp_path / "images")
    manifest = prepare_dataset(roots["shenzhen"], roots["montgomery"], tmp_path / "prepared", allow_partial=True)
    return manifest, smoke_config()


def replace_rows(manifest, transform):
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    transform(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest.write_text(buffer.getvalue(), encoding="utf-8")
    report_path = manifest.parent / "data_validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["manifest_sha256"] = sha256_file(manifest)
    report_path.write_text(json.dumps(report), encoding="utf-8")


def test_prepared_contract_accepts_matching_synthetic_evidence(prepared_manifest):
    manifest, config = prepared_manifest
    report = validate_prepared_manifest(manifest, config)
    assert report["synthetic_or_partial"] is True


def test_research_cannot_use_synthetic_or_partial_preparation(prepared_manifest):
    manifest, config = prepared_manifest
    with pytest.raises(ValueError, match="Synthetic/partial"):
        validate_prepared_manifest(manifest, {**config, "synthetic": False})


def test_modified_manifest_fails_its_recorded_checksum(prepared_manifest):
    manifest, config = prepared_manifest
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        validate_prepared_manifest(manifest, config)


def test_prepared_split_seed_must_match_experiment(prepared_manifest):
    manifest, config = prepared_manifest
    with pytest.raises(ValueError, match="split seed"):
        validate_prepared_manifest(manifest, {**config, "split_seed": 1})


def test_external_cases_cannot_be_reassigned_to_training(prepared_manifest):
    manifest, config = prepared_manifest
    replace_rows(manifest, lambda rows: rows[-1].update(split="train"))
    with pytest.raises(ValueError, match="reserved external"):
        validate_prepared_manifest(manifest, config)


def test_cross_split_patient_groups_are_rejected(prepared_manifest):
    manifest, config = prepared_manifest

    def mix_groups(rows):
        training = next(row for row in rows if row["split"] == "train")
        validation = next(row for row in rows if row["split"] == "validation")
        validation["group_id"] = training["group_id"]

    replace_rows(manifest, mix_groups)
    with pytest.raises(ValueError, match="group crossing"):
        validate_prepared_manifest(manifest, config)


def test_duplicate_decoded_pixels_are_rejected(prepared_manifest):
    manifest, config = prepared_manifest
    replace_rows(manifest, lambda rows: rows[-1].update(pixel_sha256=rows[0]["pixel_sha256"]))
    with pytest.raises(ValueError, match="duplicate pixel_sha256"):
        validate_prepared_manifest(manifest, config)


def test_research_requires_complete_cohorts_even_if_report_flag_is_changed(prepared_manifest):
    manifest, config = prepared_manifest
    report_path = manifest.parent / "data_validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["synthetic_or_partial"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="complete shenzhen"):
        validate_prepared_manifest(manifest, {**config, "synthetic": False})
