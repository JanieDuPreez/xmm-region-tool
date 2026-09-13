from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.artifacts import ArtifactMaterializationError
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.execution import ProjectionProvenance, ProjectionResult, ProjectionRule
from xmm_region_tool.limits import MAX_REGION_VERTICES
from xmm_region_tool.managed import write_bound_detector_geometry
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.provenance import EventIdentity, file_sha256


def _event(tmp_path: Path) -> EventIdentity:
    path = tmp_path / "events.fits"
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
    return EventIdentity(
        path=path,
        instrument="mos1",
        instrument_header="EMOS1",
        obs_id="001",
        exposure_id="S001",
        telescope="XMM",
        date_obs="2000-01-01T00:00:00",
        ra_pnt=10.0,
        dec_pnt=-9.0,
        pa_pnt=0.0,
    )


def _calibration(tmp_path: Path) -> CalibrationIdentity:
    return CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="1" * 64,
        calindex_sha256="2" * 64,
        replacements=(),
        ccf_search_path=(),
    )


def _producer() -> SasProducerIdentity:
    return SasProducerIdentity(
        esky2det_version="esky2det-test",
        sas_version="SAS-22",
        esky2det_sha256="3" * 64,
    )


def _result(
    tmp_path: Path,
    regions: tuple[DetectorRegion, ...],
) -> tuple[ProjectionResult, EventIdentity, CalibrationIdentity, SasProducerIdentity]:
    event = _event(tmp_path)
    calibration = _calibration(tmp_path)
    producer = _producer()
    selection = DetectorSelection(
        regions=regions,
        source_geometry_sha256="a" * 64,
    )
    provenance = ProjectionProvenance(
        event_file_sha256=file_sha256(event.path),
        event_identity_sha256=event.identity_sha256,
        celestial_geometry_sha256="a" * 64,
        detector_geometry_sha256=selection.geometry_sha256,
        context_identity_sha256="d" * 64,
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
            f"calinfoset={event.path}",
            "checkfov=no",
        ),
        relevant_environment={"SAS_CCF": "/cal/ccf.cif"},
    )
    return ProjectionResult(selection=selection, provenance=provenance), event, calibration, producer


def _write(tmp_path: Path, result, event, calibration, producer):
    return write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        result,
        event_identity=event,
        calibration_identity=calibration,
        sas_producer=producer,
    )


def test_hash_consistent_adjacent_duplicate_polygon_is_rejected_before_output(tmp_path):
    malformed = DetectorBoundary(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    result, event, calibration, producer = _result(
        tmp_path,
        (DetectorRegion(True, (malformed,)),),
    )

    with pytest.raises(ArtifactMaterializationError, match="degenerate edge"):
        _write(tmp_path, result, event, calibration, producer)

    assert not (tmp_path / "geometry.fits").exists()


def test_hash_consistent_bow_tie_polygon_is_rejected_before_output(tmp_path):
    malformed = DetectorBoundary(
        np.asarray([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]])
    )
    result, event, calibration, producer = _result(
        tmp_path,
        (DetectorRegion(True, (malformed,)),),
    )

    with pytest.raises(ArtifactMaterializationError, match="self-intersects"):
        _write(tmp_path, result, event, calibration, producer)

    assert not (tmp_path / "geometry.fits").exists()


def test_hash_consistent_collinear_polygon_is_rejected_before_output(tmp_path):
    malformed = DetectorBoundary(np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    result, event, calibration, producer = _result(
        tmp_path,
        (DetectorRegion(True, (malformed,)),),
    )

    with pytest.raises(ArtifactMaterializationError, match="zero/near-zero enclosed area"):
        _write(tmp_path, result, event, calibration, producer)

    assert not (tmp_path / "geometry.fits").exists()


def test_hash_consistent_invalid_hole_is_rejected_before_output(tmp_path):
    outer = DetectorBoundary(np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))
    outside = DetectorBoundary(
        np.asarray([[20.0, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 30.0]]),
        subtract=True,
    )
    result, event, calibration, producer = _result(
        tmp_path,
        (DetectorRegion(True, (outer, outside)),),
    )

    with pytest.raises(ArtifactMaterializationError, match="not strictly contained"):
        _write(tmp_path, result, event, calibration, producer)

    assert not (tmp_path / "geometry.fits").exists()


def _regular_polygon(vertex_count: int) -> np.ndarray:
    phase = np.arange(vertex_count, dtype=float) * (2.0 * np.pi / vertex_count)
    return np.column_stack((np.cos(phase), np.sin(phase)))


def test_managed_writer_rejects_table_overlimit_before_pairwise_topology(monkeypatch, tmp_path):
    from xmm_region_tool import topology

    boundary = DetectorBoundary(_regular_polygon(MAX_REGION_VERTICES + 1))
    result, event, calibration, producer = _result(
        tmp_path,
        (DetectorRegion(True, (boundary,)),),
    )
    monkeypatch.setattr(
        topology,
        "segments_intersect",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pairwise work started")),
    )

    with pytest.raises(ArtifactMaterializationError, match="REGION vertex resource limit"):
        _write(tmp_path, result, event, calibration, producer)

    assert not (tmp_path / "geometry.fits").exists()


def test_managed_writer_rejects_quadratic_work_budget_before_pairwise_topology(
    monkeypatch,
    tmp_path,
):
    from xmm_region_tool import topology

    boundary = DetectorBoundary(_regular_polygon(4097))
    result, event, calibration, producer = _result(
        tmp_path,
        (DetectorRegion(True, (boundary,)),),
    )
    monkeypatch.setattr(
        topology,
        "segments_intersect",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pairwise work started")),
    )

    with pytest.raises(ArtifactMaterializationError, match="topology pair-check resource limit"):
        _write(tmp_path, result, event, calibration, producer)

    assert not (tmp_path / "geometry.fits").exists()


def test_valid_hash_consistent_result_still_materialises(tmp_path):
    outer = DetectorBoundary(np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))
    hole = DetectorBoundary(
        np.asarray([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]),
        subtract=True,
    )
    result, event, calibration, producer = _result(
        tmp_path,
        (DetectorRegion(True, (outer, hole)),),
    )

    artifact = _write(tmp_path, result, event, calibration, producer)

    assert artifact.path.is_file()
    assert artifact.projection_evidence.is_file()
