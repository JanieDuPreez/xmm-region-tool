from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool import check_cli
from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.context_identity import canonical_context_identity_sha256
from xmm_region_tool.execution import ProjectionProvenance, ProjectionResult, ProjectionRule
from xmm_region_tool.model import DetectorBoundary, DetectorRegion, DetectorSelection
from xmm_region_tool.provenance import RegionBindingError, file_sha256, read_event_identity, read_region_binding
from xmm_region_tool.transactional_products import write_projected_product_atomic

_SOURCE_SHA = "9" * 64


def _write_event(path):
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
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
    fits.HDUList([primary, events]).writeto(path)
    return path


def _calibration(tmp_path):
    return CalibrationIdentity(
        cif_path=tmp_path / "ccf.cif",
        cif_file_sha256="1" * 64,
        calindex_sha256="3" * 64,
        replacements=(),
        ccf_search_path=(),
    )


def _producer():
    return SasProducerIdentity(
        esky2det_version="esky2det-test",
        sas_version="SAS-22",
        esky2det_sha256="5" * 64,
    )


def _projection_result(event, calibration, producer, *, offset=0.0):
    identity = read_event_identity(event)
    detector = DetectorSelection(
        regions=(
            DetectorRegion(
                True,
                (
                    DetectorBoundary(
                        np.asarray(
                            [
                                [offset + 0.0, 0.0],
                                [offset + 10.0, 0.0],
                                [offset + 0.0, 10.0],
                            ],
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


def _product(tmp_path, *, name="region", offset=0.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    event = tmp_path / "mos1S001-allevc.fits"
    if not event.exists():
        _write_event(event)
    calibration = _calibration(tmp_path)
    producer = _producer()
    result = _projection_result(event, calibration, producer, offset=offset)
    identity = read_event_identity(event)
    written, evidence, _ = write_projected_product_atomic(
        tmp_path / f"{name}.txt",
        result,
        representation="fits",
        event_identity=identity,
        event_file_sha256=result.provenance.event_file_sha256,
        source_region_sha256=_SOURCE_SHA,
        celestial_geometry_sha256=result.provenance.celestial_geometry_sha256,
        projection_identity_sha256=result.provenance.projection_identity_sha256,
        calibration_identity=calibration,
        sas_producer=producer,
        source_region_origin="caller-asserted",
    )
    assert written.fits_region is not None
    return event, written.fits_region, evidence, result


def _workflow_manifest(path, region, evidence, result):
    path.write_text(
        json.dumps(
            {
                "schema": "xmm-region-tool.workflow/v1",
                "products": [
                    {
                        "status": "success",
                        "fits_region": str(region.resolve()),
                        "projection_evidence": str(evidence.resolve()),
                        "celestial_geometry_sha256": result.provenance.celestial_geometry_sha256,
                        "projection_identity_sha256": result.provenance.projection_identity_sha256,
                    }
                ],
            }
        )
    )
    return path


def test_manifest_backed_checker_verifies_current_geometry_bytes(tmp_path, capsys):
    event, region, evidence, result = _product(tmp_path)
    manifest = _workflow_manifest(tmp_path / "manifest.json", region, evidence, result)

    status = check_cli.main(
        [str(region), "--event-file", str(event), "--manifest", str(manifest)]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "current detector geometry bytes verified" in output
    assert "compatible: yes" in output


def test_manifest_backed_checker_rejects_changed_region_coordinates_with_same_binding_headers(
    tmp_path,
    capsys,
):
    event, region, evidence, result = _product(tmp_path)
    manifest = _workflow_manifest(tmp_path / "manifest.json", region, evidence, result)

    with fits.open(region, mode="update", memmap=False) as hdus:
        hdus["REGION"].data["X"][0][0] += 1.0
        hdus.flush()

    status = check_cli.main(
        [str(region), "--event-file", str(event), "--manifest", str(manifest)]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "projection-evidence integrity check failed" in captured.err


def test_manifest_backed_checker_rejects_different_valid_geometry_under_old_manifest(
    tmp_path,
    capsys,
):
    event, region, evidence, result = _product(tmp_path, name="first", offset=0.0)
    _, other_region, _, _ = _product(tmp_path, name="second", offset=100.0)
    manifest = _workflow_manifest(tmp_path / "manifest.json", region, evidence, result)

    shutil.copyfile(other_region, region)

    status = check_cli.main(
        [str(region), "--event-file", str(event), "--manifest", str(manifest)]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "projection-evidence integrity check failed" in captured.err


def test_manifest_backed_checker_rejects_missing_or_tampered_sidecar(tmp_path, capsys):
    event, region, evidence, result = _product(tmp_path)
    manifest = _workflow_manifest(tmp_path / "manifest.json", region, evidence, result)

    original = evidence.read_text()
    evidence.unlink()
    assert check_cli.main(
        [str(region), "--event-file", str(event), "--manifest", str(manifest)]
    ) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "projection-evidence integrity check failed" in captured.err

    evidence.write_text(original)
    payload = json.loads(evidence.read_text())
    payload["geometry_artifact"]["file_sha256"] = "f" * 64
    evidence.write_text(json.dumps(payload))
    assert check_cli.main(
        [str(region), "--event-file", str(event), "--manifest", str(manifest)]
    ) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "projection-evidence integrity check failed" in captured.err


def test_relocated_batch_pair_uses_manifest_relative_sidecar_not_historical_path(tmp_path):
    event, region, evidence, result = _product(tmp_path / "original")
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    relocated_region = relocated / region.name
    relocated_evidence = relocated / evidence.name
    shutil.copyfile(region, relocated_region)
    shutil.copyfile(evidence, relocated_evidence)
    manifest = relocated / "xmm-region-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "xmm-region-tool.batch/v4",
                "items": [
                    {
                        "status": "success",
                        "fits_region": relocated_region.name,
                        "projection_evidence": relocated_evidence.name,
                        "celestial_geometry_sha256": result.provenance.celestial_geometry_sha256,
                        "projection_identity_sha256": result.provenance.projection_identity_sha256,
                    }
                ],
            }
        )
    )

    assert check_cli.main(
        [str(relocated_region), "--event-file", str(event), "--manifest", str(manifest)]
    ) == 0


@pytest.mark.parametrize("key", ["XMRGSHA", "XMRGCCF", "XMRGCIF", "XMRGCAL", "XMRGPRD"])
def test_region_binding_rejects_malformed_optional_digest_headers(tmp_path, key):
    _, region, _, _ = _product(tmp_path)
    with fits.open(region, mode="update", memmap=False) as hdus:
        hdus["REGION"].header[key] = "garbage"
        hdus.flush()

    with pytest.raises(RegionBindingError, match=key):
        read_region_binding(region)


def test_header_only_checker_labels_geometry_integrity_as_unverified(tmp_path, capsys):
    event, region, _, _ = _product(tmp_path)

    assert check_cli.main([str(region), "--event-file", str(event)]) == 0
    output = capsys.readouterr().out
    assert "header/event binding only" in output
    assert "detector geometry bytes NOT verified" in output


def test_header_only_checker_rejects_malformed_source_digest_cleanly(tmp_path, capsys):
    event, region, _, _ = _product(tmp_path)
    with fits.open(region, mode="update", memmap=False) as hdus:
        hdus["REGION"].header["XMRGSHA"] = "garbage"
        hdus.flush()

    assert check_cli.main([str(region), "--event-file", str(event)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "validation could not be completed" in captured.err
    assert "XMRGSHA" in captured.err
    assert "Traceback" not in captured.err
