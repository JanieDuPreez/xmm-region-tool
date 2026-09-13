from __future__ import annotations

import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool.batch import convert_selection_batch
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.execution import (
    ProjectionProvenance,
    ProjectionResult,
    ProjectionRule,
    SasProjectionContext,
)
from xmm_region_tool.model import (
    CelestialSelection,
    DetectorBoundary,
    DetectorRegion,
    DetectorSelection,
)
from xmm_region_tool.provenance import file_sha256, read_event_identity, read_region_binding


def _event(path: Path) -> Path:
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
            fits.Column(name="DETX", format="J", array=np.asarray([1], dtype=np.int32)),
            fits.Column(name="DETY", format="J", array=np.asarray([1], dtype=np.int32)),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path)
    return path


def _context(tmp_path: Path) -> SasProjectionContext:
    executable = tmp_path / "esky2det"
    executable.write_text("fake")
    cif = tmp_path / "ccf.cif"
    cif.write_text("fake")
    calibration = CalibrationIdentity(
        cif_path=cif,
        cif_file_sha256="1" * 64,
        calindex_sha256="2" * 64,
        replacements=(),
        ccf_search_path=(),
    )
    producer = SasProducerIdentity(
        esky2det_version="fake",
        sas_version="22",
        esky2det_sha256="3" * 64,
    )
    return SasProjectionContext(
        environment={"PATH": str(tmp_path), "SAS_CCF": str(cif)},
        calibration=calibration,
        producer=producer,
        esky2det_path=executable,
    )


def _fake_project(selection, *, calinfoset, context, rule):
    event_path = Path(calinfoset).resolve()
    identity = read_event_identity(event_path)
    detector = DetectorSelection(
        regions=(
            DetectorRegion(
                True,
                (
                    DetectorBoundary(
                        np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
                    ),
                ),
            ),
        ),
        source_geometry_sha256=selection.geometry_sha256,
    )
    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event_path),
        event_identity_sha256=identity.identity_sha256,
        celestial_geometry_sha256=selection.geometry_sha256,
        detector_geometry_sha256=detector.geometry_sha256,
        context_identity_sha256=context.identity_sha256,
        calibration_identity_sha256=context.calibration.identity_sha256,
        producer_identity_sha256=context.producer.identity_sha256,
        calibration_execution_evidence=context.calibration.evidence_record(),
        rule=rule,
        command=(
            "esky2det",
            "datastyle=set",
            "intab=/tmp/xmm-region-tool-test:INPUT",
            "witherrorcol=no",
            "withouttab=no",
            "outunit=det",
            "calinfostyle=set",
            f"calinfoset={event_path}",
            "checkfov=no",
        ),
        relevant_environment=context.relevant_environment_record(),
    )
    return ProjectionResult(selection=detector, provenance=provenance)


def test_batch_fits_binds_authoritative_celestial_and_projection_identities(
    monkeypatch,
    tmp_path,
):
    selection = CelestialSelection.polygon(
        SkyCoord(
            [10.0, 10.001, 10.0] * u.deg,
            [-9.0, -9.0, -8.999] * u.deg,
            frame="icrs",
        )
    )
    event = _event(tmp_path / "events.fits")
    context = _context(tmp_path)
    monkeypatch.setattr("xmm_region_tool.batch.project_selection", _fake_project)

    result = convert_selection_batch(
        selection,
        [event],
        tmp_path / "out",
        rule=ProjectionRule(samples=128),
        context=context,
        representation="fits",
    )

    assert result.successful
    item = result.items[0]
    assert item.written is not None
    assert item.written.fits_region is not None

    binding = read_region_binding(item.written.fits_region)
    manifest = json.loads(result.manifest.read_text())
    row = manifest["items"][0]

    assert binding.celestial_geometry_sha256 == item.celestial_geometry_sha256
    assert binding.projection_identity_sha256 == item.projection_identity_sha256
    assert row["celestial_geometry_sha256"] == item.celestial_geometry_sha256
    assert row["projection_identity_sha256"] == item.projection_identity_sha256

    with fits.open(item.written.fits_region) as hdus:
        header = hdus["REGION"].header
        assert header["XMRGCEL"] == item.celestial_geometry_sha256
        assert header["XMRGPRO"] == item.projection_identity_sha256
