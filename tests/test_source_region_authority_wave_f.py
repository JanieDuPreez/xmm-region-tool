from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from xmm_region_tool.artifacts import DetectorGeometryArtifact
from xmm_region_tool.execution import ProjectionRule
import xmm_region_tool.batch as batch
import xmm_region_tool.cli as cli
import xmm_region_tool.managed as managed
import xmm_region_tool.public_batch as public_batch
import xmm_region_tool.public_managed as public_managed


_SOURCE_SHA = "9" * 64


def _projection_result_stub():
    provenance = SimpleNamespace(
        event_file_sha256="1" * 64,
        event_identity_sha256="2" * 64,
        calibration_identity_sha256="3" * 64,
        producer_identity_sha256="4" * 64,
        celestial_geometry_sha256="5" * 64,
        projection_identity_sha256="6" * 64,
    )
    return SimpleNamespace(provenance=provenance)


def test_internal_managed_writer_preserves_explicit_source_origin(tmp_path, monkeypatch):
    captured = {}

    def fake_write_detector_geometry(path, result, **kwargs):
        captured.update(kwargs)
        geometry = Path(path)
        geometry.write_bytes(b"staged geometry")
        evidence = geometry.with_suffix(".provenance.json")
        evidence.write_text(
            json.dumps({"geometry_artifact": {"path": str(geometry)}}) + "\n"
        )
        return DetectorGeometryArtifact(
            path=geometry,
            geometry_sha256="7" * 64,
            file_sha256="8" * 64,
            projection_evidence=evidence,
        )

    monkeypatch.setattr(managed, "write_detector_geometry", fake_write_detector_geometry)
    monkeypatch.setattr(managed, "_validate_projection_evidence", lambda artifact: {})

    managed.write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        _projection_result_stub(),
        event_identity=object(),
        calibration_identity=object(),
        sas_producer=object(),
        source_region_sha256=_SOURCE_SHA,
        source_region_origin="tool-hashed-source-file",
    )

    assert captured["source_region_sha256"] == _SOURCE_SHA
    assert captured["source_region_origin"] == "tool-hashed-source-file"


def test_public_managed_writer_does_not_downgrade_source_origin(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()

    monkeypatch.setattr(public_managed, "_validate_producer_for_managed_use", lambda producer: None)
    monkeypatch.setattr(public_managed, "_validate_result_context_identity", lambda result: None)
    monkeypatch.setattr(
        public_managed,
        "_projection_event_binding",
        lambda event_path, result: object(),
    )
    monkeypatch.setattr(
        public_managed,
        "_validate_persisted_managed_evidence",
        lambda artifact: {},
    )

    def fake_write(path, result, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(public_managed, "_write_bound_detector_geometry", fake_write)

    result = public_managed.write_bound_detector_geometry(
        tmp_path / "geometry.fits",
        object(),
        event_file=tmp_path / "event.fits",
        calibration_identity=SimpleNamespace(cif_path=tmp_path / "ccf.cif"),
        sas_producer=object(),
        source_region_sha256=_SOURCE_SHA,
        source_region_origin="tool-hashed-source-file",
    )

    assert result is sentinel
    assert captured["source_region_sha256"] == _SOURCE_SHA
    assert captured["source_region_origin"] == "tool-hashed-source-file"


def test_programmatic_batch_source_digest_is_caller_asserted_by_default():
    assert batch._source_region_origin(_SOURCE_SHA, None) == "caller-asserted"
    assert (
        batch._source_region_origin(_SOURCE_SHA, "tool-hashed-source-file")
        == "tool-hashed-source-file"
    )


def test_ds9_batch_adapter_marks_tool_hashed_source_file(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()
    selection = object()

    monkeypatch.setattr(
        batch,
        "load_ds9_selection_snapshot",
        lambda source, samples: (selection, _SOURCE_SHA),
    )

    def fake_convert_selection_batch(celestial_selection, event_files, output_dir, **kwargs):
        captured.update(kwargs)
        assert celestial_selection is selection
        return sentinel

    monkeypatch.setattr(batch, "convert_selection_batch", fake_convert_selection_batch)

    result = batch.convert_batch(
        tmp_path / "source.reg",
        [tmp_path / "event.fits"],
        tmp_path / "out",
        rule=ProjectionRule(samples=16),
        context=object(),
    )

    assert result is sentinel
    assert captured["source_region_sha256"] == _SOURCE_SHA
    assert captured["source_region_origin"] == "tool-hashed-source-file"


def test_public_programmatic_batch_forwards_explicit_source_origin(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()

    def fake_convert(celestial_selection, event_files, output_dir, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(public_batch, "_convert_selection_batch", fake_convert)

    result = public_batch.convert_selection_batch(
        object(),
        [tmp_path / "event.fits"],
        tmp_path / "out",
        rule=ProjectionRule(samples=16),
        source_region_sha256=_SOURCE_SHA,
        source_region_origin="tool-hashed-source-file",
    )

    assert result is sentinel
    assert captured["source_region_sha256"] == _SOURCE_SHA
    assert captured["source_region_origin"] == "tool-hashed-source-file"


def test_cli_records_tool_hashed_source_authority(tmp_path, monkeypatch):
    event = tmp_path / "event.fits"
    source = tmp_path / "source.reg"
    output = tmp_path / "region.txt"
    evidence = output.with_suffix(".provenance.json")
    captured = {}

    event_identity = SimpleNamespace(
        instrument="mos1",
        obs_id="0144310101",
        exposure_id="S001",
    )
    celestial_selection = SimpleNamespace(
        geometry_sha256="a" * 64,
        canonical_geometry_record=lambda: {"schema": "test"},
    )
    cell_selection = SimpleNamespace(
        geometry_sha256="b" * 64,
        canonical_geometry_record=lambda: {"schema": "test-cell"},
    )
    cell = SimpleNamespace(index=1, label="region-1", selection=cell_selection)
    projection_provenance = SimpleNamespace(
        event_identity_sha256="c" * 64,
        event_file_sha256="d" * 64,
        projection_identity_sha256="e" * 64,
        refinement_diagnostics={},
        evidence_record=lambda: {"refinement_diagnostics": {}},
    )
    projected = SimpleNamespace(
        selection=SimpleNamespace(regions=(), geometry_sha256="f" * 64),
        provenance=projection_provenance,
    )
    context = SimpleNamespace(
        identity_sha256="1" * 64,
        calibration=SimpleNamespace(
            identity_sha256="2" * 64,
            cif_path=tmp_path / "ccf.cif",
        ),
        producer=SimpleNamespace(identity_sha256="3" * 64),
    )
    args = SimpleNamespace(
        region=source,
        event_files=[event],
        instrument=None,
        search_dir=None,
        output=output,
        output_dir=tmp_path,
        combine=False,
        quiet=True,
        verbose=False,
        detector_tolerance=1.0,
        max_depth=12,
        max_vertices=4096,
        samples=16,
        representation="fits",
        inline_limit=4096,
        max_components=4096,
    )

    monkeypatch.setattr(cli, "_resolve_events", lambda args: (event,))
    monkeypatch.setattr(cli, "_validate_event_set", lambda event_files: ((event, event_identity),))
    monkeypatch.setattr(cli, "file_sha256", lambda path: "4" * 64)
    monkeypatch.setattr(
        cli,
        "load_ds9_selection_snapshot",
        lambda path, samples: (celestial_selection, _SOURCE_SHA),
    )
    monkeypatch.setattr(cli, "_cells", lambda args, selection: (cell,))
    monkeypatch.setattr(cli, "project_selection", lambda *args, **kwargs: projected)
    monkeypatch.setattr(
        cli,
        "validate_projection_event_generation",
        lambda *args, **kwargs: event_identity,
    )

    def fake_atomic(output_path, result, *, precommit=None, **kwargs):
        captured.update(kwargs)
        written = SimpleNamespace(
            regionfile=output_path,
            fits_region=tmp_path / "region.fits",
        )
        command = precommit(written) if precommit is not None else None
        return written, evidence, command

    monkeypatch.setattr(cli, "write_projected_product_atomic", fake_atomic)
    monkeypatch.setattr(cli, "_esas_command", lambda *args, **kwargs: "mosspectra")

    assert cli._run(args, context=context) == 0
    assert captured["source_region_origin"] == "tool-hashed-source-file"

    manifest = json.loads((tmp_path / "source-xmm-region-manifest.json").read_text())
    assert manifest["source_region_sha256"] == _SOURCE_SHA
    assert manifest["source_region_origin"] == "tool-hashed-source-file"
