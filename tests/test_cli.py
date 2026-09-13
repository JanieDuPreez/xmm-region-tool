from __future__ import annotations

from types import SimpleNamespace

import pytest

from xmm_region_tool import cli
from xmm_region_tool.sas import SasConversionError


def test_expected_sas_failure_is_rendered_without_traceback(monkeypatch, tmp_path, capsys):
    source = tmp_path / "out-of-domain.reg"
    source.write_text("icrs\ncircle(180.54,-0.006,1\")\n")
    event = tmp_path / "events.fits"
    event.write_bytes(b"synthetic event bytes")
    output = tmp_path / "region.txt"

    monkeypatch.setattr(
        cli,
        "read_event_identity",
        lambda path: SimpleNamespace(
            obs_id="001",
            instrument="mos1",
            exposure_id="S001",
        ),
    )
    monkeypatch.setattr(
        cli.SasProjectionContext,
        "from_environment",
        lambda: SimpleNamespace(
            identity_sha256="b" * 64,
            calibration=SimpleNamespace(
                identity_sha256="c" * 64,
                cif_path=tmp_path / "ccf.cif",
            ),
            producer=SimpleNamespace(identity_sha256="d" * 64),
        ),
    )
    selection = SimpleNamespace(
        geometry_sha256="a" * 64,
        canonical_geometry_record=lambda: {
            "schema": "synthetic-test-selection/v1",
            "regions": [],
        },
    )
    monkeypatch.setattr(
        cli,
        "load_ds9_selection_snapshot",
        lambda path, samples: (selection, cli.file_sha256(path)),
    )
    monkeypatch.setattr(
        cli,
        "extraction_cells",
        lambda value: (cli.ExtractionCell(index=1, label="r001", selection=selection),),
    )

    def fail_projection(*args, **kwargs):
        raise SasConversionError(
            "esky2det failed with exit code 1: ** esky2det: error "
            "(thetaGreaterThan90), sky position is outside the projection domain",
            command=("/sas/bin/esky2det", "checkfov=no"),
            returncode=1,
            stderr="** esky2det: error (thetaGreaterThan90)",
        )

    monkeypatch.setattr(cli, "project_selection", fail_projection)

    status = cli.main(
        [
            str(source),
            "--event-file",
            str(event),
            "--output",
            str(output),
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert captured.err.startswith("xmm-region: error: r001/mos1: esky2det failed with exit code 1")
    assert "thetaGreaterThan90" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()
    assert not output.with_suffix(".fits").exists()
    assert not output.with_suffix(".provenance.json").exists()


def test_unexpected_programming_error_is_not_hidden(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    source.write_text("icrs\ncircle(10,-9,1\")\n")
    event = tmp_path / "events.fits"
    event.write_bytes(b"synthetic event bytes")

    def fail_unexpected(args):
        raise RuntimeError("unexpected internal defect")

    monkeypatch.setattr(cli, "_run", fail_unexpected)

    with pytest.raises(RuntimeError, match="unexpected internal defect"):
        cli.main(
            [
                str(source),
                "--event-file",
                str(event),
                "--output",
                str(tmp_path / "region.txt"),
            ]
        )


def test_quiet_and_verbose_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["source.reg", "--quiet", "--verbose"])


def test_compact_summary_is_short_and_hides_projection_diagnostics(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "regions"
    output_dir.mkdir()
    event = tmp_path / "mos1S001-allevc.fits"
    manifest = output_dir / "profile-xmm-region-manifest.json"
    identity = SimpleNamespace(instrument="mos1")
    products = [
        {
            "status": "success",
            "cell_label": "r001",
            "instrument": "mos1",
            "regionfile": str(output_dir / "profile-r001-mos1.txt"),
            "refinement_diagnostics": {
                "total_vertices": 220,
                "max_accepted_error": 0.94,
            },
        },
        {
            "status": "success",
            "cell_label": "r002",
            "instrument": "mos1",
            "regionfile": str(output_dir / "profile-r002-mos1.txt"),
            "refinement_diagnostics": {
                "total_vertices": 276,
                "max_accepted_error": 0.98,
            },
        },
    ]

    cli._print_compact_summary(
        products=products,
        manifest=manifest,
        event_set=((event, identity),),
        cell_count=2,
    )

    output = capsys.readouterr().out
    assert "Created 2 SAS region files." in output
    assert "Output: regions" in output
    assert "r001" in output
    assert "profile-r001-mos1.txt" in output
    assert "run mosspectra once per generated .txt file" in output
    assert "Event: mos1S001-allevc.fits" in output
    assert "Region: choose one of the .txt files above" in output
    assert "--verbose" in output
    assert "vertices" not in output
    assert "max_error" not in output
    assert str(tmp_path) not in output


def test_check_cli_manifest_rejects_wrong_science_cell(tmp_path, monkeypatch):
    import json

    import numpy as np
    from astropy.io import fits

    import xmm_region_tool.check_cli as check_cli
    from xmm_region_tool.output import write_fits_region
    from xmm_region_tool.provenance import read_event_identity
    from xmm_region_tool.sas import DetectorBoundary, DetectorRegion

    event = tmp_path / "events.fits"
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="J", array=np.array([1, 2], dtype=np.int32)),
            fits.Column(name="DETY", format="J", array=np.array([3, 4], dtype=np.int32)),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(event)

    detector = DetectorRegion(
        True,
        (
            DetectorBoundary(
                np.asarray(
                    [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
                    dtype=float,
                )
            ),
        ),
    )

    r001 = tmp_path / "profile-r001-mos1.fits"
    r002 = tmp_path / "profile-r002-mos1.fits"

    identity = read_event_identity(event)

    write_fits_region(
        r001,
        [detector],
        event_identity=identity,
        celestial_geometry_sha256="1" * 64,
        projection_identity_sha256="a" * 64,
    )
    write_fits_region(
        r002,
        [detector],
        event_identity=identity,
        celestial_geometry_sha256="2" * 64,
        projection_identity_sha256="b" * 64,
    )

    monkeypatch.setattr(
        check_cli,
        "load_bound_detector_geometry",
        lambda *args, **kwargs: object(),
    )

    manifest = tmp_path / "profile-xmm-region-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "xmm-region-tool.workflow/v1",
                "products": [
                    {
                        "cell_index": 1,
                        "cell_label": "r001",
                        "fits_region": str(r001),
                        "projection_evidence": str(tmp_path / "r001.provenance.json"),
                        "celestial_geometry_sha256": "1" * 64,
                        "projection_identity_sha256": "a" * 64,
                    },
                    {
                        "cell_index": 2,
                        "cell_label": "r002",
                        "fits_region": str(r002),
                        "projection_evidence": str(tmp_path / "r002.provenance.json"),
                        "celestial_geometry_sha256": "2" * 64,
                        "projection_identity_sha256": "b" * 64,
                    },
                ],
            }
        )
    )

    assert (
        check_cli.main(
            [
                str(r001),
                "--event-file",
                str(event),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )

    r001.write_bytes(r002.read_bytes())

    assert (
        check_cli.main(
            [
                str(r001),
                "--event-file",
                str(event),
                "--manifest",
                str(manifest),
            ]
        )
        == 1
    )


def test_success_manifest_serializes_nested_immutable_refinement_diagnostics(
    monkeypatch,
    tmp_path,
):
    import json

    from xmm_region_tool.execution import ProjectionProvenance, ProjectionRule

    source = tmp_path / "profile.reg"
    source.write_text('icrs\ncircle(10,-9,30")\n')

    event = tmp_path / "mos1S001-allevc.fits"
    event.write_bytes(b"synthetic event bytes")

    output_dir = tmp_path / "output"

    event_identity = SimpleNamespace(
        obs_id="0144310101",
        instrument="mos1",
        exposure_id="S001",
    )
    context = SimpleNamespace(
        identity_sha256="b" * 64,
        calibration=SimpleNamespace(
            identity_sha256="c" * 64,
            cif_path=tmp_path / "ccf.cif",
        ),
        producer=SimpleNamespace(identity_sha256="d" * 64),
    )
    selection = SimpleNamespace(
        geometry_sha256="a" * 64,
        canonical_geometry_record=lambda: {
            "schema": "synthetic-test-selection/v1",
            "regions": [],
        },
    )

    diagnostics = {
        "total_vertices": 12,
        "max_accepted_error": 0.75,
        "boundaries": [
            {
                "index": 0,
                "max_error": 0.75,
                "depths": [1, 2, 3],
            }
        ],
    }

    provenance = ProjectionProvenance(
        event_file_sha256="e" * 64,
        event_identity_sha256="f" * 64,
        celestial_geometry_sha256=selection.geometry_sha256,
        detector_geometry_sha256="9" * 64,
        context_identity_sha256=context.identity_sha256,
        calibration_identity_sha256=context.calibration.identity_sha256,
        producer_identity_sha256=context.producer.identity_sha256,
        rule=ProjectionRule(),
        command=("esky2det",),
        relevant_environment={},
        refinement_diagnostics=diagnostics,
    )
    projected = SimpleNamespace(
        selection=SimpleNamespace(
            regions=(),
            geometry_sha256=provenance.detector_geometry_sha256,
        ),
        provenance=provenance,
    )

    monkeypatch.setattr(cli, "_resolve_events", lambda args: (event,))
    monkeypatch.setattr(
        cli,
        "_validate_event_set",
        lambda paths: ((event, event_identity),),
    )
    monkeypatch.setattr(
        cli.SasProjectionContext,
        "from_environment",
        lambda: context,
    )
    monkeypatch.setattr(
        cli,
        "load_ds9_selection_snapshot",
        lambda path, samples: (selection, cli.file_sha256(path)),
    )
    monkeypatch.setattr(
        cli,
        "extraction_cells",
        lambda value: (
            cli.ExtractionCell(index=1, label="r001", selection=selection),
        ),
    )
    monkeypatch.setattr(cli, "project_selection", lambda *args, **kwargs: projected)
    monkeypatch.setattr(
        cli,
        "validate_projection_event_generation",
        lambda path, caller_identity, **kwargs: caller_identity,
    )

    def fake_atomic(output, projected_result, *, precommit=None, **kwargs):
        written = SimpleNamespace(regionfile=output, fits_region=None)
        command = precommit(written) if precommit is not None else None
        return written, output.with_suffix(".provenance.json"), command

    monkeypatch.setattr(cli, "write_projected_product_atomic", fake_atomic)
    monkeypatch.setattr(cli, "_esas_command", lambda *args, **kwargs: "mosspectra ...")

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

    assert status == 0

    manifest = output_dir / "profile-xmm-region-manifest.json"
    payload = json.loads(manifest.read_text())

    assert payload["successful"] is True
    assert payload["products"][0]["status"] == "success"
    assert payload["products"][0]["event_file_sha256"] == provenance.event_file_sha256
    assert payload["products"][0]["event_identity_sha256"] == provenance.event_identity_sha256
    assert payload["products"][0]["refinement_diagnostics"] == diagnostics


@pytest.mark.parametrize("tolerance", ["nan", "inf"])
def test_nonfinite_tolerance_fails_cleanly_before_sas(tolerance, tmp_path, capsys):
    status = cli.main([
        str(tmp_path / "source.reg"),
        "--event-file",
        str(tmp_path / "event.fits"),
        "--detector-tolerance",
        tolerance,
    ])
    assert status == 1
    captured = capsys.readouterr()
    assert "detector_tolerance" in captured.err
    assert "Traceback" not in captured.err
