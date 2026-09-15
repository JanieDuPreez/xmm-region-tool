from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import xmm_region_tool
from xmm_region_tool import (
    ArtifactMaterializationError,
    ProjectionProvenance,
    ProjectionResult,
    ProjectionRule,
    materialize_bound_sas_regionfile,
    write_bound_detector_geometry,
)
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.context_identity import canonical_context_identity_sha256
from xmm_region_tool.event_roles import read_science_calinfoset_identity
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.artifacts import write_detector_geometry
from xmm_region_tool.provenance import (
    EventIdentity,
    EventIdentityError,
    file_sha256,
    read_event_identity,
)


def _write_event(path: Path, *, marker: str | None = None) -> Path:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0
    if marker is not None:
        primary.header["TESTTAG"] = marker
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([3.0, 4.0])),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path, overwrite=True)
    return path


def _calibration(tmp_path: Path) -> CalibrationIdentity:
    return CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="1" * 64,
        calindex_sha256="3" * 64,
        replacements=(),
        ccf_search_path=(),
    )


def _producer() -> SasProducerIdentity:
    return SasProducerIdentity(
        esky2det_version="esky2det-test",
        sas_version="SAS-22",
        esky2det_sha256="5" * 64,
    )


def _projection_result(
    event: Path,
    calibration: CalibrationIdentity,
    producer: SasProducerIdentity,
) -> ProjectionResult:
    identity = read_science_calinfoset_identity(event)
    detector = DetectorSelection(
        regions=(
            DetectorRegion(
                True,
                (
                    DetectorBoundary(
                        np.asarray(
                            [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
                            dtype=float,
                        )
                    ),
                ),
            ),
        ),
        source_geometry_sha256="a" * 64,
    )
    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event),
        event_identity_sha256=identity.identity_sha256,
        celestial_geometry_sha256="a" * 64,
        detector_geometry_sha256=detector.geometry_sha256,
        context_identity_sha256=canonical_context_identity_sha256(
            calibration.identity_sha256,
            producer.identity_sha256,
        ),
        calibration_identity_sha256=calibration.identity_sha256,
        producer_identity_sha256=producer.identity_sha256,
        calibration_execution_evidence=calibration.evidence_record(),
        rule=ProjectionRule(samples=128),
        command=(
            "/sas/esky2det",
            "datastyle=set",
            "intab=/tmp/xmm-region-tool-test:INPUT",
            "witherrorcol=no",
            "withouttab=no",
            "outunit=det",
            "calinfostyle=set",
            f"calinfoset={event}",
            "checkfov=no",
        ),
        relevant_environment={"SAS_CCF": "/cal/ccf.cif"},
    )
    return ProjectionResult(selection=detector, provenance=provenance)


def test_top_level_managed_api_exposes_self_contained_event_file_path():
    write_parameters = inspect.signature(write_bound_detector_geometry).parameters
    reuse_parameters = inspect.signature(materialize_bound_sas_regionfile).parameters

    assert "event_file" in write_parameters
    assert "event_file" in reuse_parameters
    assert hasattr(xmm_region_tool, "selection_from_sky_regions")


def test_top_level_managed_workflow_is_self_contained(tmp_path):
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _projection_result(event, calibration, producer)

    artifact = write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
    )
    wrapper = materialize_bound_sas_regionfile(
        tmp_path / "run" / "region.txt",
        artifact,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
        expected_celestial_geometry_sha256=result.provenance.celestial_geometry_sha256,
        expected_projection_identity_sha256=result.provenance.projection_identity_sha256,
    )

    assert artifact.path.is_file()
    assert artifact.projection_evidence.is_file()
    assert wrapper.path.is_file()
    assert wrapper.geometry_path.is_file()
    assert wrapper.geometry_path != artifact.path
    assert file_sha256(wrapper.geometry_path) == artifact.file_sha256
    assert wrapper.path.read_text() == f"&&region({wrapper.geometry_path},DETX,DETY)\n"


def test_managed_execution_copy_is_independent_of_later_cache_replacement(tmp_path):
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _projection_result(event, calibration, producer)
    artifact = write_bound_detector_geometry(
        tmp_path / "cache" / "geometry.fits",
        result,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
    )

    wrapper = materialize_bound_sas_regionfile(
        tmp_path / "run" / "region.txt",
        artifact,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
        expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
        expected_projection_identity_sha256=artifact.projection_identity_sha256,
    )
    staged_bytes = wrapper.geometry_path.read_bytes()

    artifact.path.write_bytes(b"replaced cache bytes")

    assert wrapper.geometry_path.read_bytes() == staged_bytes
    assert file_sha256(wrapper.geometry_path) == artifact.file_sha256
    assert str(artifact.path) not in wrapper.path.read_text()
    assert str(wrapper.geometry_path) in wrapper.path.read_text()


def test_legacy_event_identity_argument_is_only_a_path_carrier(tmp_path):
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _projection_result(event, calibration, producer)
    real_identity = read_science_calinfoset_identity(event)
    forged = EventIdentity(
        path=event,
        instrument="pn",
        instrument_header="EPN",
        obs_id="forged",
        exposure_id="forged",
        telescope="XMM",
        date_obs=real_identity.date_obs,
        ra_pnt=real_identity.ra_pnt,
        dec_pnt=real_identity.dec_pnt,
        pa_pnt=real_identity.pa_pnt,
    )

    artifact = write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_identity=forged,
        calibration_identity=calibration,
        sas_producer=producer,
    )

    with fits.open(artifact.path) as hdus:
        header = hdus["REGION"].header
        assert header["INSTRUME"] == "EMOS1"
        assert header["OBS_ID"] == "0144310101"
        assert header["XMMRGID"] == real_identity.identity_sha256


def test_top_level_managed_write_rejects_changed_exact_event_generation(tmp_path):
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _projection_result(event, calibration, producer)

    _write_event(event, marker="generation-b")

    with pytest.raises(ArtifactMaterializationError, match="exact calinfoset"):
        write_bound_detector_geometry(
            tmp_path / "geometry.fits",
            result,
            event_file=event,
            calibration_identity=calibration,
            sas_producer=producer,
        )


def test_top_level_managed_reuse_rejects_changed_exact_event_generation(tmp_path):
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _projection_result(event, calibration, producer)
    artifact = write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
    )

    _write_event(event, marker="generation-b")

    with pytest.raises(ArtifactMaterializationError, match="exact event used"):
        materialize_bound_sas_regionfile(
            tmp_path / "region.txt",
            artifact,
            event_file=event,
            calibration_identity=calibration,
            sas_producer=producer,
            expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
            expected_projection_identity_sha256=artifact.projection_identity_sha256,
        )



def test_managed_write_keeps_science_calinfoset_event_role_during_durable_reread(tmp_path):
    event = tmp_path / "pnS003-allevc.fits"
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EPN"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "0144310101003"
    primary.header["EXPIDSTR"] = "S003"
    primary.header["DATE-OBS"] = "2002-12-22T21:48:12"
    primary.header["DATE_OBS"] = "2002-12-22T20:56:03.000"
    primary.header["RA_PNT"] = 15.701125
    primary.header["DEC_PNT"] = -21.8844166666667
    primary.header["PA_PNT"] = 227.224395751953

    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([3.0, 4.0])),
        ],
        name="EVENTS",
    )
    events.header["TELESCOP"] = "XMM"
    events.header["INSTRUME"] = "EPN"
    events.header["OBS_ID"] = "0144310101"
    events.header["EXP_ID"] = "0144310101003"
    events.header["EXPIDSTR"] = "S003"
    events.header["DATE-OBS"] = "2002-12-22T21:48:12"
    events.header["DATE_OBS"] = "2002-12-22T20:56:03.000"
    events.header["RA_PNT"] = 15.701125
    events.header["DEC_PNT"] = -21.8844166666667
    events.header["PA_PNT"] = 227.224395751953
    fits.HDUList([primary, events]).writeto(event)

    with pytest.raises(EventIdentityError, match="conflicting event header DATE-OBS"):
        read_event_identity(event)

    science_identity = read_science_calinfoset_identity(event)
    assert science_identity.date_obs == "2002-12-22T21:48:12"

    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _projection_result(event, calibration, producer)

    artifact = write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
    )
    reloaded = xmm_region_tool.load_bound_detector_geometry(
        artifact.path,
        projection_evidence=artifact.projection_evidence,
    )
    wrapper = materialize_bound_sas_regionfile(
        tmp_path / "run" / "region.txt",
        reloaded,
        event_file=event,
        calibration_identity=calibration,
        sas_producer=producer,
        expected_celestial_geometry_sha256=artifact.celestial_geometry_sha256,
        expected_projection_identity_sha256=artifact.projection_identity_sha256,
    )

    assert artifact.path.is_file()
    assert artifact.projection_evidence.is_file()
    assert reloaded.event_identity_sha256 == result.provenance.event_identity_sha256
    assert wrapper.path.is_file()
    assert wrapper.geometry_path.is_file()


def test_generic_detector_writer_still_uses_strict_generic_event_identity(tmp_path):
    event = tmp_path / "event.fits"
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
    primary.header["DATE_OBS"] = "2003-06-21T23:59:00.000"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0

    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0, 2.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([3.0, 4.0])),
        ],
        name="EVENTS",
    )
    events.header["TELESCOP"] = "XMM"
    events.header["INSTRUME"] = "EMOS1"
    events.header["OBS_ID"] = "0144310101"
    events.header["EXP_ID"] = "S001"
    events.header["DATE-OBS"] = "2003-06-22T00:00:00"
    events.header["DATE_OBS"] = "2003-06-21T23:59:00.000"
    events.header["RA_PNT"] = 15.673
    events.header["DEC_PNT"] = -21.88
    events.header["PA_PNT"] = 72.0
    fits.HDUList([primary, events]).writeto(event)

    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _projection_result(event, calibration, producer)
    science_identity = read_science_calinfoset_identity(event)

    with pytest.raises(ArtifactMaterializationError, match="cannot re-read"):
        write_detector_geometry(
            tmp_path / "generic-geometry.fits",
            result,
            event_identity=science_identity,
            calibration_identity=calibration,
            sas_producer=producer,
        )
