from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.execution import ADAPTIVE_REFINEMENT, ProjectionRule, SasProjectionContext
from xmm_region_tool.model import CelestialSelection
from xmm_region_tool.sas import SasConversionError, _validate_detector_boundary, project_selection


def write_event(path: Path) -> Path:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0151900101"
    primary.header["EXP_ID"] = "S007"
    primary.header["DATE-OBS"] = "2001-01-01T00:00:00"
    primary.header["RA_PNT"] = 10.0
    primary.header["DEC_PNT"] = -9.0
    primary.header["PA_PNT"] = 0.0
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="J", array=np.asarray([0], dtype=np.int32)),
            fits.Column(name="DETY", format="J", array=np.asarray([0], dtype=np.int32)),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path)
    return path


def context(tmp_path: Path) -> SasProjectionContext:
    executable = tmp_path / "esky2det"
    executable.write_text("fake")
    cif = tmp_path / "ccf.cif"
    cif.write_text("fake")
    return SasProjectionContext(
        environment={"PATH": str(tmp_path), "SAS_CCF": str(cif)},
        calibration=CalibrationIdentity(
            cif_path=cif,
            cif_file_sha256="1" * 64,
            calindex_sha256="2" * 64,
            replacements=(),
            ccf_search_path=(),
        ),
        producer=SasProducerIdentity(
            esky2det_version="22",
            sas_version="22",
            esky2det_sha256="3" * 64,
        ),
        esky2det_path=executable,
    )


def _install_projection(monkeypatch, mapping) -> None:
    def fake_run(command, **kwargs):
        intab = next(value for value in command if value.startswith("intab="))
        path = Path(intab.removeprefix("intab=").split(":", maxsplit=1)[0])
        with fits.open(path) as hdus:
            source = hdus["INPUT"].data
            ra = np.asarray(source["RA"], dtype=float)
            dec = np.asarray(source["DEC"], dtype=float)
            detx, dety = mapping(ra, dec)
            output = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="RA", format="D", array=ra),
                    fits.Column(name="DEC", format="D", array=dec),
                    fits.Column(name="ROW_ID", format="J", array=source["ROW_ID"]),
                    fits.Column(name="DETX", format="D", array=detx),
                    fits.Column(name="DETY", format="D", array=dety),
                ],
                name="INPUT",
            )
        fits.HDUList([fits.PrimaryHDU(), output]).writeto(path, overwrite=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", fake_run)

    # This suite deliberately exercises adaptive detector refinement against a
    # synthetic nonlinear transform with fake CIF/executable fixtures. Context
    # coherence is tested independently in tests/test_execution.py.
    monkeypatch.setattr(SasProjectionContext, "validate", lambda self: None)


def install_nonlinear_projection(monkeypatch) -> None:
    def mapping(ra, dec):
        x = ra - 10.0
        detx = x * 1000.0
        dety = (dec + 9.0) * 1000.0 + 50000.0 * x * (0.02 - x)
        return detx, dety

    _install_projection(monkeypatch, mapping)


def install_midpoint_zero_s_projection(monkeypatch) -> None:
    def mapping(ra, dec):
        x = ra - 10.0
        detx = x * 1000.0
        # On the horizontal [10, 10.02] edge the endpoints and midpoint all
        # have zero added displacement, while the quarter probes do not.
        dety = (
            (dec + 9.0) * 1000.0
            + 8.0e7 * x * (0.02 - x) * (x - 0.01)
        )
        return detx, dety

    _install_projection(monkeypatch, mapping)


def rectangle_selection() -> CelestialSelection:
    vertices = SkyCoord(
        [10.0, 10.02, 10.02, 10.0] * u.deg,
        [-9.0, -9.0, -8.98, -8.98] * u.deg,
        frame="icrs",
    )
    return CelestialSelection.polygon(vertices)


def test_adaptive_projection_refines_nonlinear_polygon_edges(monkeypatch, tmp_path):
    install_nonlinear_projection(monkeypatch)
    event = write_event(tmp_path / "events.fits")
    rule = ProjectionRule(detector_tolerance=0.1, max_depth=10)

    result = project_selection(
        rectangle_selection(),
        calinfoset=event,
        context=context(tmp_path),
        rule=rule,
    )

    vertices = result.selection.regions[0].boundaries[0].vertices
    diagnostics = result.provenance.refinement_diagnostics
    assert rule.refinement == ADAPTIVE_REFINEMENT
    assert len(vertices) > 4
    assert diagnostics["mode"] == ADAPTIVE_REFINEMENT
    assert float(diagnostics["max_accepted_probe_error"]) <= 0.1 + 1e-12
    assert diagnostics["interior_probe_fractions"] == (0.25, 0.5, 0.75)
    assert diagnostics["total_vertices"] == len(vertices)
    assert diagnostics["sas_calls"] >= 1
    assert len(result.provenance.commands) == diagnostics["sas_calls"]


def test_three_probe_estimator_rejects_midpoint_zero_s_curve(monkeypatch, tmp_path):
    install_midpoint_zero_s_projection(monkeypatch)
    event = write_event(tmp_path / "s-events.fits")
    rule = ProjectionRule(detector_tolerance=0.1, max_depth=10)

    result = project_selection(
        rectangle_selection(),
        calinfoset=event,
        context=context(tmp_path),
        rule=rule,
    )

    diagnostics = result.provenance.refinement_diagnostics
    vertices = result.selection.regions[0].boundaries[0].vertices
    assert len(vertices) > 4
    assert float(diagnostics["max_observed_probe_error"]) > rule.detector_tolerance
    assert float(diagnostics["max_accepted_probe_error"]) <= rule.detector_tolerance + 1e-12
    assert diagnostics["interior_probe_fractions"] == (0.25, 0.5, 0.75)


def test_adaptive_v2_refinement_changes_execution_identity_vocabulary():
    rule = ProjectionRule(detector_tolerance=1.0)
    record = rule.canonical_record()
    assert record["refinement"] == "adaptive-detector-chord/v2"

    old_record = dict(record)
    old_record["refinement"] = "adaptive-detector-chord/v1"

    def canonical(payload):
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    assert canonical(record) != canonical(old_record)


def test_adaptive_projection_refines_sector_arcs_and_edges(monkeypatch, tmp_path):
    install_nonlinear_projection(monkeypatch)
    event = write_event(tmp_path / "sector-events.fits")
    rule = ProjectionRule(detector_tolerance=0.02, max_depth=12)
    selection = CelestialSelection.sector_annulus(
        SkyCoord(10.01 * u.deg, -8.99 * u.deg, frame="icrs"),
        0.12 * u.deg,
        0.30 * u.deg,
        20.0 * u.deg,
        230.0 * u.deg,
    )

    result = project_selection(
        selection,
        calinfoset=event,
        context=context(tmp_path),
        rule=rule,
    )

    vertices = result.selection.regions[0].boundaries[0].vertices
    diagnostics = result.provenance.refinement_diagnostics
    path = selection.regions[0].boundaries[0].path
    assert len(path.initial_intervals()) == 4
    assert len(vertices) > 4
    assert float(diagnostics["max_accepted_probe_error"]) <= rule.detector_tolerance + 1e-12
    assert diagnostics["total_projected_point_evaluations"] > 8
    assert diagnostics["total_vertices"] == len(vertices)


def test_adaptive_projection_fails_if_tolerance_cannot_be_met(monkeypatch, tmp_path):
    install_nonlinear_projection(monkeypatch)
    event = write_event(tmp_path / "events.fits")

    with pytest.raises(SasConversionError, match="could not satisfy sampled probe tolerance"):
        project_selection(
            rectangle_selection(),
            calinfoset=event,
            context=context(tmp_path),
            rule=ProjectionRule(detector_tolerance=1e-12, max_depth=1),
        )


def test_detector_topology_rejects_degenerate_edges():
    with pytest.raises(SasConversionError, match="degenerate edge"):
        _validate_detector_boundary(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )


def test_detector_topology_rejects_self_intersection():
    with pytest.raises(SasConversionError, match="self-intersects"):
        _validate_detector_boundary(
            np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        )
