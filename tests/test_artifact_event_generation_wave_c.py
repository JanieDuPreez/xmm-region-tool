from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.artifacts import ArtifactMaterializationError, write_detector_geometry
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.execution import ProjectionProvenance, ProjectionResult, ProjectionRule
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.provenance import EventIdentity, file_sha256, read_event_identity


def _write_event(path: Path) -> Path:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "001"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2000-01-01T00:00:00"
    primary.header["RA_PNT"] = 10.0
    primary.header["DEC_PNT"] = -9.0
    primary.header["PA_PNT"] = 0.0
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="D", array=np.asarray([1.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([2.0])),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path)
    return path


def test_fabricated_identity_and_matching_provenance_cannot_vouch_for_each_other(tmp_path):
    event_path = _write_event(tmp_path / "events.fits")
    actual = read_event_identity(event_path)
    fabricated = EventIdentity(
        path=event_path.resolve(),
        instrument=actual.instrument,
        instrument_header=actual.instrument_header,
        obs_id="999",
        exposure_id=actual.exposure_id,
        telescope=actual.telescope,
        date_obs=actual.date_obs,
        ra_pnt=actual.ra_pnt,
        dec_pnt=actual.dec_pnt,
        pa_pnt=actual.pa_pnt,
    )
    assert fabricated.identity_sha256 != actual.identity_sha256

    calibration = CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="1" * 64,
        calindex_sha256="2" * 64,
        replacements=(),
        ccf_search_path=(),
    )
    producer = SasProducerIdentity(
        esky2det_version="22",
        sas_version="22",
        esky2det_sha256="3" * 64,
    )
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    selection = DetectorSelection(
        regions=(DetectorRegion(True, (DetectorBoundary(vertices),)),),
        source_geometry_sha256="a" * 64,
    )
    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event_path),
        event_identity_sha256=fabricated.identity_sha256,
        celestial_geometry_sha256="a" * 64,
        detector_geometry_sha256=selection.geometry_sha256,
        context_identity_sha256="4" * 64,
        calibration_identity_sha256=calibration.identity_sha256,
        producer_identity_sha256=producer.identity_sha256,
        calibration_execution_evidence=calibration.evidence_record(),
        rule=ProjectionRule(samples=128),
        command=("fake-esky2det",),
        relevant_environment={},
    )
    result = ProjectionResult(selection=selection, provenance=provenance)

    with pytest.raises(
        ArtifactMaterializationError,
        match="caller event_identity does not describe the current event artifact",
    ):
        write_detector_geometry(
            tmp_path / "geometry.fits",
            result,
            event_identity=fabricated,
            calibration_identity=calibration,
            sas_producer=producer,
        )
