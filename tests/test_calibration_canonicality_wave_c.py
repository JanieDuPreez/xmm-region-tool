from __future__ import annotations

import copy

import pytest

from xmm_region_tool.artifacts import ArtifactMaterializationError
from xmm_region_tool.calibration import (
    CalibrationConstituent,
    CalibrationIdentity,
    CalibrationIdentityError,
    CalibrationReplacement,
    SasProducerIdentity,
)
from xmm_region_tool.execution import SasProjectionContext
from xmm_region_tool.managed import _canonical_sha256, _validate_calibration_record


def _record() -> dict[str, object]:
    canonical = {
        "schema": "xmm-region-tool.calibration-identity/v2",
        "calindex_sha256": "1" * 64,
        "constituents": [
            {"name": "A.CCF", "sha256": "2" * 64},
            {"name": "B.CCF", "sha256": "3" * 64},
        ],
        "replacements": [
            {"name": "C.CCF", "sha256": "4" * 64},
        ],
    }
    return {
        **canonical,
        "identity_sha256": _canonical_sha256(canonical),
        "cif_file_sha256": "5" * 64,
        "constituent_evidence": [
            {"name": "A.CCF", "sha256": "2" * 64, "cif_md5": "6" * 32, "size": 10},
            {"name": "B.CCF", "sha256": "3" * 64, "cif_md5": None, "size": 20},
        ],
    }


def _validate(record: dict[str, object]) -> None:
    _validate_calibration_record(
        {"calibration": record},
        expected_identity=str(record["identity_sha256"]),
    )


def _rehash(record: dict[str, object]) -> None:
    canonical = {
        "schema": record["schema"],
        "calindex_sha256": record["calindex_sha256"],
        "constituents": record["constituents"],
        "replacements": record["replacements"],
    }
    record["identity_sha256"] = _canonical_sha256(canonical)


def test_managed_calibration_rejects_non_sha_calindex_even_when_self_consistent():
    record = _record()
    record["calindex_sha256"] = "not-a-sha"
    _rehash(record)

    with pytest.raises(ArtifactMaterializationError, match="CALINDEX"):
        _validate(record)


def test_managed_calibration_rejects_malformed_replacement_even_when_rehashed():
    record = _record()
    record["replacements"] = [{"name": "C.CCF", "sha256": "not-a-sha"}]
    _rehash(record)

    with pytest.raises(ArtifactMaterializationError, match="replacement"):
        _validate(record)


@pytest.mark.parametrize(
    "replacement",
    [
        {"name": "", "sha256": "4" * 64},
        {"name": "subdir/C.CCF", "sha256": "4" * 64},
        {"name": "C.CCF", "sha256": "4" * 64, "extra": "lookalike"},
    ],
)
def test_managed_calibration_rejects_noncanonical_replacement_records(replacement):
    record = _record()
    record["replacements"] = [replacement]
    _rehash(record)

    with pytest.raises(ArtifactMaterializationError, match="replacement"):
        _validate(record)


def test_managed_calibration_rejects_duplicate_or_unsorted_constituents():
    duplicate = _record()
    duplicate["constituents"] = [
        {"name": "A.CCF", "sha256": "2" * 64},
        {"name": "A.CCF", "sha256": "2" * 64},
    ]
    duplicate["constituent_evidence"] = [
        {"name": "A.CCF", "sha256": "2" * 64, "cif_md5": None, "size": 10},
        {"name": "A.CCF", "sha256": "2" * 64, "cif_md5": None, "size": 10},
    ]
    _rehash(duplicate)
    with pytest.raises(ArtifactMaterializationError, match="duplicate"):
        _validate(duplicate)

    unsorted = _record()
    unsorted["constituents"] = list(reversed(unsorted["constituents"]))
    unsorted["constituent_evidence"] = list(reversed(unsorted["constituent_evidence"]))
    _rehash(unsorted)
    with pytest.raises(ArtifactMaterializationError, match="canonical basename order"):
        _validate(unsorted)


def test_managed_calibration_rejects_unexpected_top_level_and_evidence_fields():
    top_level = _record()
    top_level["extra"] = "lookalike"
    with pytest.raises(ArtifactMaterializationError, match="not canonical"):
        _validate(top_level)

    evidence = _record()
    evidence["constituent_evidence"][0]["extra"] = "lookalike"
    with pytest.raises(ArtifactMaterializationError, match="not canonical"):
        _validate(evidence)


def test_calibration_identity_snapshots_caller_owned_lists(tmp_path):
    replacements = [CalibrationReplacement("R.CCF", "1" * 64)]
    constituents = [CalibrationConstituent("B.CCF", "2" * 64)]
    search_path = [tmp_path / "ccf"]

    calibration = CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="3" * 64,
        calindex_sha256="4" * 64,
        replacements=replacements,
        ccf_search_path=search_path,
        constituents=constituents,
    )
    before = calibration.identity_sha256

    replacements.append(CalibrationReplacement("R2.CCF", "5" * 64))
    constituents.append(CalibrationConstituent("A.CCF", "6" * 64))
    search_path.append(tmp_path / "other")

    assert calibration.identity_sha256 == before
    assert tuple(item.name for item in calibration.replacements) == ("R.CCF",)
    assert tuple(item.name for item in calibration.constituents) == ("B.CCF",)
    assert calibration.ccf_search_path == ((tmp_path / "ccf").resolve(),)


def test_projection_context_identity_cannot_drift_from_source_lists(tmp_path):
    replacements = [CalibrationReplacement("R.CCF", "1" * 64)]
    calibration = CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="2" * 64,
        calindex_sha256="3" * 64,
        replacements=replacements,
        ccf_search_path=[],
    )
    producer = SasProducerIdentity(
        esky2det_version="esky2det-1.20 [test]",
        sas_version="test",
        esky2det_sha256="4" * 64,
    )
    context = SasProjectionContext(
        environment={},
        calibration=calibration,
        producer=producer,
        esky2det_path=tmp_path / "esky2det",
    )
    before = context.identity_sha256

    replacements.append(CalibrationReplacement("R2.CCF", "5" * 64))

    assert context.identity_sha256 == before


def test_calibration_value_objects_reject_noncanonical_manual_records(tmp_path):
    with pytest.raises(CalibrationIdentityError, match="basename"):
        CalibrationReplacement("subdir/R.CCF", "1" * 64)
    with pytest.raises(CalibrationIdentityError, match="SHA256"):
        CalibrationConstituent("A.CCF", "not-a-sha")
    with pytest.raises(CalibrationIdentityError, match="unique"):
        CalibrationIdentity(
            cif_path=tmp_path / "ccf.cif",
            cif_file_sha256="2" * 64,
            calindex_sha256="3" * 64,
            replacements=[
                CalibrationReplacement("R.CCF", "4" * 64),
                CalibrationReplacement("R.CCF", "4" * 64),
            ],
            ccf_search_path=[],
        )


def test_constituents_are_canonicalized_to_basename_order(tmp_path):
    calibration = CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="1" * 64,
        calindex_sha256="2" * 64,
        replacements=(),
        ccf_search_path=(),
        constituents=[
            CalibrationConstituent("B.CCF", "3" * 64),
            CalibrationConstituent("A.CCF", "4" * 64),
        ],
    )
    assert [item.name for item in calibration.constituents] == ["A.CCF", "B.CCF"]


def test_nominal_managed_calibration_record_still_validates():
    record = copy.deepcopy(_record())
    _validate(record)
