from __future__ import annotations

from pathlib import Path

import pytest
from astropy.io import fits

from xmm_region_tool.calibration import CalibrationIdentity
from xmm_region_tool.context_suitability import (
    ContextSuitabilityError,
    ObservationRecord,
    capture_context_suitability,
)
from xmm_region_tool.provenance import EventIdentity


def _write_cif(path: Path, *, obsvdate: str = "2013-06-08T14:52:32") -> Path:
    primary = fits.PrimaryHDU()
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="FNAME", format="32A", array=["A.CCF"])],
        name="CALINDEX",
    )
    table.header["OBSVDATE"] = obsvdate
    table.header["ANALDATE"] = "2026-07-13T13:45:48"
    fits.HDUList([primary, table]).writeto(path)
    return path


def _calibration(cif: Path) -> CalibrationIdentity:
    return CalibrationIdentity(
        cif_path=cif,
        cif_file_sha256="a" * 64,
        calindex_sha256="b" * 64,
        replacements=(),
        ccf_search_path=(),
        constituents=(),
    )


def _event(*, obs_id: str = "0723802001", date_obs: str = "2013-06-08T15:10:41") -> EventIdentity:
    return EventIdentity(
        path=Path("event.fits"),
        instrument="mos1",
        instrument_header="EMOS1",
        obs_id=obs_id,
        exposure_id="S001",
        telescope="XMM",
        date_obs=date_obs,
        ra_pnt=15.645,
        dec_pnt=-21.8721388888889,
        pa_pnt=55.8575019836426,
    )


def _declared(*, obs_id: str = "0723802001") -> ObservationRecord:
    return ObservationRecord(
        obs_id=obs_id,
        scheduled_start="2013-06-08T14:52:32",
        scheduled_end="2013-06-09T16:34:12",
    )


def _observation_block(*, obs_id: str, start: str, end: str) -> str:
    return "\n".join(
        ["OBSERVATION", f"{obs_id:<10} 2472", "unused", start, end]
    ) + "\n"


def _write_original_odf(directory: Path, *, obs_id: str = "0723802001") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / f"2472_{obs_id}_SCX00000SUM.ASC"
    summary.write_text(
        _observation_block(
            obs_id=obs_id,
            start="2013-06-08T14:52:32",
            end="2013-06-09T16:34:12",
        )
    )
    return summary


def _write_sosf(path: Path, odf_dir: Path, *, obs_id: str = "0723802001") -> Path:
    path.write_text(
        _observation_block(
            obs_id=obs_id,
            start="2013-06-08T15:09:56",
            end="2013-06-09T15:44:18",
        )
        + f"PATH {odf_dir}\n"
    )
    return path


def test_real_fixture_relationship_accepts_sosf_plus_original_odf(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")
    odf_dir = tmp_path / "odf"
    _write_original_odf(odf_dir)
    sosf = _write_sosf(tmp_path / "2472_0723802001_SCX00000SUM.SAS", odf_dir)

    evidence = capture_context_suitability(
        environment={"SAS_ODF": str(sosf)},
        calibration=_calibration(cif),
        event_identity=_event(),
    )

    assert evidence.event_obs_id == "0723802001"
    assert evidence.association_obs_id == "0723802001"
    assert evidence.association_source == "sas-summary+original-odf"
    assert evidence.cif_obsvdate == evidence.observation_start
    assert evidence.active_summary_sha256 is not None
    assert evidence.original_summary_sha256 is not None


def test_sosf_obsid_mismatch_fails_before_material_date_check(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")
    odf_dir = tmp_path / "odf"
    _write_original_odf(odf_dir)
    sosf = _write_sosf(tmp_path / "other.SAS", odf_dir, obs_id="0144310101")

    with pytest.raises(ContextSuitabilityError, match="active SAS ODF summary belongs"):
        capture_context_suitability(
            environment={"SAS_ODF": str(sosf)},
            calibration=_calibration(cif),
            event_identity=_event(),
        )


def test_original_odf_obsid_mismatch_fails(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")
    odf_dir = tmp_path / "odf"
    _write_original_odf(odf_dir, obs_id="0144310101")
    sosf = _write_sosf(tmp_path / "summary.SAS", odf_dir)

    with pytest.raises(ContextSuitabilityError, match="original ODF belongs"):
        capture_context_suitability(
            environment={"SAS_ODF": str(sosf)},
            calibration=_calibration(cif),
            event_identity=_event(),
        )


def test_stale_cif_observation_date_fails_even_when_sosf_obsid_matches(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif", obsvdate="2003-04-01T00:00:00")
    odf_dir = tmp_path / "odf"
    _write_original_odf(odf_dir)
    sosf = _write_sosf(tmp_path / "summary.SAS", odf_dir)

    with pytest.raises(ContextSuitabilityError, match="CIF OBSVDATE does not match"):
        capture_context_suitability(
            environment={"SAS_ODF": str(sosf)},
            calibration=_calibration(cif),
            event_identity=_event(),
        )


def test_original_odf_directory_is_valid_without_sosf(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")
    odf_dir = tmp_path / "odf"
    _write_original_odf(odf_dir)

    evidence = capture_context_suitability(
        environment={"SAS_ODF": str(odf_dir)},
        calibration=_calibration(cif),
        event_identity=_event(),
    )

    assert evidence.association_source == "original-odf"
    assert evidence.active_summary_sha256 is None
    assert evidence.original_summary_sha256 is not None


def test_missing_sas_odf_fails_closed_without_explicit_evidence(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")

    with pytest.raises(ContextSuitabilityError, match="no explicit observation evidence"):
        capture_context_suitability(
            environment={},
            calibration=_calibration(cif),
            event_identity=_event(),
        )


def test_declared_observation_supports_odf_independent_cifbuild(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")

    evidence = capture_context_suitability(
        environment={},
        calibration=_calibration(cif),
        event_identity=_event(),
        declared_observation=_declared(),
    )

    assert evidence.association_source == "declared-observation"
    assert evidence.association_obs_id == "0723802001"
    assert evidence.observation_start == "2013-06-08T14:52:32.000"


def test_declared_observation_rejects_wrong_obsid(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")

    with pytest.raises(ContextSuitabilityError, match="declared observation belongs"):
        capture_context_suitability(
            environment={},
            calibration=_calibration(cif),
            event_identity=_event(),
            declared_observation=_declared(obs_id="0144310101"),
        )


def test_declared_observation_rejects_event_outside_interval(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")

    with pytest.raises(ContextSuitabilityError, match="outside the declared observation interval"):
        capture_context_suitability(
            environment={},
            calibration=_calibration(cif),
            event_identity=_event(date_obs="2014-01-01T00:00:00"),
            declared_observation=_declared(),
        )


def test_sosf_without_original_odf_path_fails_material_suitability(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")
    sosf = tmp_path / "summary.SAS"
    sosf.write_text(
        _observation_block(
            obs_id="0723802001",
            start="2013-06-08T15:09:56",
            end="2013-06-09T15:44:18",
        )
    )

    with pytest.raises(ContextSuitabilityError, match="no usable PATH"):
        capture_context_suitability(
            environment={"SAS_ODF": str(sosf)},
            calibration=_calibration(cif),
            event_identity=_event(),
        )


def test_evidence_changes_when_summary_bytes_change(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")
    odf_dir = tmp_path / "odf"
    _write_original_odf(odf_dir)
    sosf = _write_sosf(tmp_path / "summary.SAS", odf_dir)

    before = capture_context_suitability(
        environment={"SAS_ODF": str(sosf)},
        calibration=_calibration(cif),
        event_identity=_event(),
    )
    sosf.write_text(sosf.read_text() + "COMMENT changed\n")
    after = capture_context_suitability(
        environment={"SAS_ODF": str(sosf)},
        calibration=_calibration(cif),
        event_identity=_event(),
    )

    assert before != after
