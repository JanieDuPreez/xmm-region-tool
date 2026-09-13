from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from xmm_region_tool import cli
from xmm_region_tool.execution import ProjectionProvenance, ProjectionResult, ProjectionRule
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.provenance import file_sha256, read_event_identity


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
    fits.HDUList([primary, events]).writeto(path)
    return path


def _projected(cell, event: Path, context, rule: ProjectionRule) -> ProjectionResult:
    identity = read_event_identity(event)
    event_sha = file_sha256(event)
    shift = int(event_sha[:4], 16) / 1000.0
    detector = DetectorSelection(
        regions=(
            DetectorRegion(
                True,
                (
                    DetectorBoundary(
                        np.asarray(
                            [
                                [shift, 0.0],
                                [shift + 1.0, 0.0],
                                [shift, 1.0],
                            ],
                            dtype=float,
                        )
                    ),
                ),
            ),
        ),
        source_geometry_sha256=cell.geometry_sha256,
    )
    provenance = ProjectionProvenance(
        event_file_sha256=event_sha,
        event_identity_sha256=identity.identity_sha256,
        celestial_geometry_sha256=cell.geometry_sha256,
        detector_geometry_sha256=detector.geometry_sha256,
        context_identity_sha256=context.identity_sha256,
        calibration_identity_sha256=context.calibration.identity_sha256,
        producer_identity_sha256=context.producer.identity_sha256,
        rule=rule,
        command=("fake-esky2det",),
        relevant_environment={},
        refinement_diagnostics={"mode": "test"},
    )
    return ProjectionResult(selection=detector, provenance=provenance)


def test_cli_freezes_one_exact_event_generation_across_cells(monkeypatch, tmp_path):
    source = tmp_path / "profile.reg"
    source.write_text("icrs\ncircle(10,-9,30\")\n")
    event = _write_event(tmp_path / "mos1S001-allevc.fits")
    replacement = _write_event(tmp_path / "replacement.fits", marker="generation-b")
    identity_a = read_event_identity(event)
    identity_b = read_event_identity(replacement)
    sha_a = file_sha256(event)
    sha_b = file_sha256(replacement)
    assert identity_a.identity_sha256 == identity_b.identity_sha256
    assert sha_a != sha_b

    selection = SimpleNamespace(
        geometry_sha256="a" * 64,
        canonical_geometry_record=lambda: {
            "schema": "synthetic-test-selection/v1",
            "regions": [],
        },
    )
    cells = (
        cli.ExtractionCell(index=1, label="r001", selection=selection),
        cli.ExtractionCell(index=2, label="r002", selection=selection),
    )
    context = SimpleNamespace(
        identity_sha256="b" * 64,
        calibration=SimpleNamespace(
            identity_sha256="c" * 64,
            cif_path=tmp_path / "ccf.cif",
        ),
        producer=SimpleNamespace(identity_sha256="d" * 64),
    )

    monkeypatch.setattr(cli.SasProjectionContext, "from_environment", lambda: context)
    monkeypatch.setattr(
        cli,
        "load_ds9_selection_snapshot",
        lambda path, samples: (selection, file_sha256(path)),
    )
    monkeypatch.setattr(cli, "extraction_cells", lambda value: cells)

    calls = 0

    def project(cell, *, calinfoset, context, rule):
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement.replace(event)
        return _projected(cell, event, context, rule)

    monkeypatch.setattr(cli, "project_selection", project)

    def write_product(output, projected, *, precommit=None, **kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("synthetic region wrapper\n")
        written = SimpleNamespace(regionfile=output, fits_region=None)
        command = precommit(written) if precommit is not None else None
        return written, output.with_suffix(".provenance.json"), command

    monkeypatch.setattr(cli, "write_projected_product_atomic", write_product)
    monkeypatch.setattr(cli, "_esas_command", lambda *args, **kwargs: "mosspectra ...")

    output_dir = tmp_path / "out"
    status = cli.main(
        [
            str(source),
            "--event-file",
            str(event),
            "--output-dir",
            str(output_dir),
            "--quiet",
        ]
    )

    assert status == 1
    assert calls == 2
    wrappers = sorted(output_dir.glob("*.txt"))
    assert len(wrappers) == 1
    assert "r001" in wrappers[0].name

    manifest = json.loads((output_dir / "profile-xmm-region-manifest.json").read_text())
    assert manifest["successful"] is False
    assert [item["status"] for item in manifest["products"]] == ["success", "failed"]
    assert manifest["products"][0]["event_file_sha256"] == sha_a
    assert manifest["products"][0]["event_identity_sha256"] == identity_a.identity_sha256
    assert "event_file_sha256" not in manifest["products"][1]
    assert "generation changed" in manifest["products"][1]["error_message"]
    assert file_sha256(event) == sha_b
