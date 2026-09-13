from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

import xmm_region_tool.batch as batch
import xmm_region_tool.cli as cli
import xmm_region_tool.public_managed as public_managed
from xmm_region_tool.artifacts import ArtifactMaterializationError
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.constructors import circle_selection
from xmm_region_tool.execution import ProjectionRule, SasProjectionContext
from xmm_region_tool.workflow import ExtractionCell, WorkflowError


def _cli_identity():
    return SimpleNamespace(
        instrument="mos1",
        exposure_id="S001",
        obs_id="0144310101",
    )


def _cli_context(tmp_path: Path):
    cif = tmp_path / "ccf.cif"
    cif.write_text("cif")
    return SimpleNamespace(calibration=SimpleNamespace(cif_path=cif))


def _cli_args(source: Path, event: Path, output: Path, *, representation: str = "fits"):
    return cli.build_parser().parse_args(
        [
            str(source),
            "--event-file",
            str(event),
            "--output",
            str(output),
            "--representation",
            representation,
            "--quiet",
        ]
    )


def _prepare_cli_preflight(monkeypatch, source: Path, event: Path):
    identity = _cli_identity()
    selection = object()
    cell = ExtractionCell(index=1, label="r001", selection=object())
    monkeypatch.setattr(cli, "read_event_identity", lambda path: identity)
    monkeypatch.setattr(cli, "load_ds9_selection_snapshot", lambda *args, **kwargs: (selection, "a" * 64))
    monkeypatch.setattr(cli, "_cells", lambda args, value: (cell,))
    monkeypatch.setattr(
        cli,
        "project_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("projection started")),
    )
    return _cli_context(source.parent)


def test_cli_rejects_event_vs_derived_fits_before_projection(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    event = tmp_path / "event.fits"
    output = tmp_path / "event.txt"
    source.write_text("fk5\n")
    event.write_bytes(b"science-event")
    original = event.read_bytes()
    context = _prepare_cli_preflight(monkeypatch, source, event)

    with pytest.raises(WorkflowError, match="aliases protected event input"):
        cli._run(_cli_args(source, event, output), context=context)

    assert event.read_bytes() == original
    assert not output.exists()


def test_cli_rejects_wrapper_equal_source_before_projection(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    event = tmp_path / "event.fits"
    source.write_text("original-source")
    event.write_bytes(b"science-event")
    original = source.read_text()
    context = _prepare_cli_preflight(monkeypatch, source, event)

    with pytest.raises(WorkflowError, match="aliases protected source DS9 region"):
        cli._run(
            _cli_args(source, event, source, representation="expression"),
            context=context,
        )

    assert source.read_text() == original


def test_cli_rejects_wrapper_manifest_collision_before_projection(monkeypatch, tmp_path):
    source = tmp_path / "region.reg"
    event = tmp_path / "event.fits"
    output = tmp_path / "region-xmm-region-manifest.json"
    source.write_text("fk5\n")
    event.write_bytes(b"science-event")
    context = _prepare_cli_preflight(monkeypatch, source, event)

    with pytest.raises(WorkflowError, match="aliases generated workflow manifest"):
        cli._run(
            _cli_args(source, event, output, representation="expression"),
            context=context,
        )

    assert not output.exists()


def test_cli_rejects_repeated_event_before_identity_read(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    event = tmp_path / "event.fits"
    source.write_text("fk5\n")
    event.write_bytes(b"science-event")
    args = cli.build_parser().parse_args(
        [
            str(source),
            "--event-file",
            str(event),
            "--event-file",
            str(event),
            "--output-dir",
            str(tmp_path / "out"),
            "--quiet",
        ]
    )
    monkeypatch.setattr(
        cli,
        "read_event_identity",
        lambda path: (_ for _ in ()).throw(AssertionError("identity read started")),
    )

    with pytest.raises(WorkflowError, match="duplicate/aliased event input"):
        cli._run(args, context=_cli_context(tmp_path))

    assert not (tmp_path / "out").exists()


def test_public_managed_write_rejects_event_output_alias(monkeypatch, tmp_path):
    event = tmp_path / "event.fits"
    cif = tmp_path / "ccf.cif"
    event.write_bytes(b"science-event")
    cif.write_bytes(b"calibration")
    original = event.read_bytes()
    monkeypatch.setattr(public_managed, "_validate_producer_for_managed_use", lambda value: None)
    monkeypatch.setattr(public_managed, "_validate_result_context_identity", lambda value: None)
    monkeypatch.setattr(
        public_managed,
        "_projection_event_binding",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("binding started")),
    )

    with pytest.raises(ArtifactMaterializationError, match="aliases protected science event"):
        public_managed.write_bound_detector_geometry(
            event,
            object(),
            event_file=event,
            calibration_identity=SimpleNamespace(cif_path=cif),
            sas_producer=object(),
        )

    assert event.read_bytes() == original


def test_public_managed_write_rejects_cif_output_alias(monkeypatch, tmp_path):
    event = tmp_path / "event.fits"
    cif = tmp_path / "ccf.cif"
    event.write_bytes(b"science-event")
    cif.write_bytes(b"calibration")
    original = cif.read_bytes()
    monkeypatch.setattr(public_managed, "_validate_producer_for_managed_use", lambda value: None)
    monkeypatch.setattr(public_managed, "_validate_result_context_identity", lambda value: None)

    with pytest.raises(ArtifactMaterializationError, match="aliases protected active CIF"):
        public_managed.write_bound_detector_geometry(
            cif,
            object(),
            event_file=event,
            calibration_identity=SimpleNamespace(cif_path=cif),
            sas_producer=object(),
        )

    assert cif.read_bytes() == original


def test_public_managed_staging_rejects_hardlink_to_geometry(tmp_path):
    geometry = tmp_path / "geometry.fits"
    sidecar = tmp_path / "geometry.provenance.json"
    wrapper = tmp_path / "wrapper.txt"
    event = tmp_path / "event.fits"
    cif = tmp_path / "ccf.cif"
    geometry.write_bytes(b"geometry")
    sidecar.write_text("{}")
    event.write_bytes(b"event")
    cif.write_bytes(b"cif")
    os.link(geometry, wrapper)
    artifact = SimpleNamespace(path=geometry, projection_evidence=sidecar)

    with pytest.raises(ArtifactMaterializationError, match="aliases protected bound detector geometry"):
        public_managed.materialize_bound_sas_regionfile(
            wrapper,
            artifact,
            event_file=event,
            calibration_identity=SimpleNamespace(cif_path=cif),
            sas_producer=object(),
            expected_celestial_geometry_sha256="a" * 64,
            expected_projection_identity_sha256="b" * 64,
        )

    assert geometry.read_bytes() == b"geometry"


def _batch_context(tmp_path: Path) -> SasProjectionContext:
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
            esky2det_version="fake",
            sas_version="22",
            esky2det_sha256="3" * 64,
        ),
        esky2det_path=executable,
    )


def test_batch_rejects_event_vs_derived_fits_before_projection(monkeypatch, tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    event = root / "collision.fits"
    event.write_bytes(b"science-event")
    identity = SimpleNamespace(obs_id="001", instrument="mos1", exposure_id="S001")
    monkeypatch.setattr(batch, "read_event_identity", lambda path: identity)
    monkeypatch.setattr(batch, "file_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(batch, "event_output_label", lambda *args: "collision")
    monkeypatch.setattr(
        batch,
        "project_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("projection started")),
    )
    selection = circle_selection(SkyCoord(10 * u.deg, -9 * u.deg), 1 * u.arcmin)

    with pytest.raises(batch.BatchConversionError, match="aliases protected event input"):
        batch.convert_selection_batch(
            selection,
            [event],
            root,
            rule=ProjectionRule(detector_tolerance=1.0),
            context=_batch_context(tmp_path),
        )

    assert event.read_bytes() == b"science-event"


def test_ds9_batch_rejects_source_manifest_collision_before_projection(monkeypatch, tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    source = root / "xmm-region-manifest.json"
    source.write_text("source-region")
    event = tmp_path / "event.fits"
    event.write_bytes(b"science-event")
    identity = SimpleNamespace(obs_id="001", instrument="mos1", exposure_id="S001")
    selection = circle_selection(SkyCoord(10 * u.deg, -9 * u.deg), 1 * u.arcmin)
    monkeypatch.setattr(batch, "load_ds9_selection_snapshot", lambda *args, **kwargs: (selection, "9" * 64))
    monkeypatch.setattr(batch, "read_event_identity", lambda path: identity)
    monkeypatch.setattr(batch, "file_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(
        batch,
        "project_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("projection started")),
    )

    with pytest.raises(batch.BatchConversionError, match="aliases protected source DS9 region"):
        batch.convert_batch(
            source,
            [event],
            root,
            context=_batch_context(tmp_path),
        )

    assert source.read_text() == "source-region"
