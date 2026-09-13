from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool.batch import (
    BatchConversionError,
    BatchItemResult,
    convert_batch,
    convert_selection_batch,
)
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
from xmm_region_tool.provenance import file_sha256, read_event_identity
from xmm_region_tool.sas import SasConversionError


def write_event(
    path: Path,
    *,
    instrument: str,
    obs_id: str,
    exposure: str,
) -> Path:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = instrument
    primary.header["OBS_ID"] = obs_id
    primary.header["EXP_ID"] = exposure
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


def context(tmp_path: Path, marker: str = "shared") -> SasProjectionContext:
    root = tmp_path / marker
    root.mkdir(exist_ok=True)
    executable = root / "esky2det"
    executable.write_text(f"fake-{marker}")
    cif = root / "ccf.cif"
    cif.write_text(f"fake-{marker}")
    marker_digit = str((sum(ord(char) for char in marker) % 8) + 1)
    return SasProjectionContext(
        environment={"PATH": str(root), "SAS_CCF": str(cif)},
        calibration=CalibrationIdentity(
            cif_path=cif,
            cif_file_sha256=marker_digit * 64,
            calindex_sha256=str((int(marker_digit) % 8) + 1) * 64,
            replacements=(),
            ccf_search_path=(),
        ),
        producer=SasProducerIdentity(
            esky2det_version=f"22-{marker}",
            sas_version="22",
            esky2det_sha256=str((int(marker_digit) % 7) + 2) * 64,
        ),
        esky2det_path=executable,
    )


def fake_project(selection, *, calinfoset, context, rule):
    event_path = Path(calinfoset).resolve()
    event_sha = file_sha256(event_path)
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
                            ]
                        )
                    ),
                ),
            ),
        ),
        source_geometry_sha256=selection.geometry_sha256,
    )
    identity = read_event_identity(event_path)
    provenance = ProjectionProvenance(
        event_file_sha256=event_sha,
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


def test_batch_materialises_distinct_outputs_for_cameras_in_one_observation(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.reg"
    source.write_text("fk5\npolygon(10,-9,10.001,-9,10,-8.999)\n")
    events = [
        write_event(tmp_path / "m1.fits", instrument="EMOS1", obs_id="001", exposure="S001"),
        write_event(tmp_path / "m2.fits", instrument="EMOS2", obs_id="001", exposure="S002"),
        write_event(tmp_path / "pn.fits", instrument="EPN", obs_id="001", exposure="S003"),
    ]
    shared = context(tmp_path)
    monkeypatch.setattr("xmm_region_tool.batch.project_selection", fake_project)

    result = convert_batch(source, events, tmp_path / "out", context=shared)

    assert result.successful
    assert len(result.items) == 3
    assert len({item.label for item in result.items}) == 3
    manifest = json.loads(result.manifest.read_text())
    assert manifest["context_mode"] == "shared-single-obsid"
    assert manifest["shared_context"]["identity_sha256"] == shared.identity_sha256
    assert all(row["event_file_sha256"] for row in manifest["items"])
    assert all(row["context_identity_sha256"] == shared.identity_sha256 for row in manifest["items"])
    for event, item, row in zip(events, result.items, manifest["items"], strict=True):
        assert item.event_file_sha256 == file_sha256(event)
        assert row["event_file_sha256"] == item.event_file_sha256
        assert row["event_identity_sha256"] == item.event_identity.identity_sha256


def test_shared_context_rejects_multiple_observations(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    source.write_text("fk5\npolygon(10,-9,10.001,-9,10,-8.999)\n")
    events = [
        write_event(tmp_path / "m1.fits", instrument="EMOS1", obs_id="001", exposure="S001"),
        write_event(tmp_path / "m1b.fits", instrument="EMOS1", obs_id="002", exposure="S001"),
    ]
    monkeypatch.setattr("xmm_region_tool.batch.project_selection", fake_project)

    with pytest.raises(BatchConversionError, match="cannot be reused across multiple ObsIDs"):
        convert_batch(source, events, tmp_path / "out", context=context(tmp_path))


def test_per_event_context_resolver_supports_multiple_observations(monkeypatch, tmp_path):
    vertices = SkyCoord(
        [10.0, 10.001, 10.0] * u.deg,
        [-9.0, -9.0, -8.999] * u.deg,
        frame="icrs",
    )
    selection = CelestialSelection.polygon(vertices)
    events = [
        write_event(tmp_path / "m1.fits", instrument="EMOS1", obs_id="001", exposure="S001"),
        write_event(tmp_path / "m1b.fits", instrument="EMOS1", obs_id="002", exposure="S001"),
    ]
    contexts = {"001": context(tmp_path, "obs001"), "002": context(tmp_path, "obs002")}
    monkeypatch.setattr("xmm_region_tool.batch.project_selection", fake_project)

    def resolver(event_path, identity):
        assert event_path in [path.resolve() for path in events]
        return contexts[identity.obs_id]

    result = convert_selection_batch(
        selection,
        events,
        tmp_path / "out",
        rule=ProjectionRule(samples=128),
        context_resolver=resolver,
    )

    assert result.successful
    manifest = json.loads(result.manifest.read_text())
    assert manifest["context_mode"] == "per-event-resolver"
    assert "shared_context" not in manifest
    per_item = {row["event_identity"]["obs_id"]: row for row in manifest["items"]}
    assert per_item["001"]["context_identity_sha256"] == contexts["001"].identity_sha256
    assert per_item["002"]["context_identity_sha256"] == contexts["002"].identity_sha256
    assert per_item["001"]["context_identity_sha256"] != per_item["002"]["context_identity_sha256"]


def test_batch_records_failure_without_omitting_other_contributors(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    source.write_text("fk5\npolygon(10,-9,10.001,-9,10,-8.999)\n")
    valid = write_event(
        tmp_path / "m1.fits",
        instrument="EMOS1",
        obs_id="001",
        exposure="S001",
    )
    missing = tmp_path / "missing.fits"
    monkeypatch.setattr("xmm_region_tool.batch.project_selection", fake_project)

    result = convert_batch(
        source,
        [valid, missing],
        tmp_path / "out",
        context=context(tmp_path),
    )

    assert not result.successful
    assert [item.status for item in result.items] == ["success", "failed"]
    assert result.items[1].error_type is not None
    assert result.items[1].event_file_sha256 is None
    manifest = json.loads(result.manifest.read_text())
    assert manifest["successful"] is False
    assert [row["status"] for row in manifest["items"]] == ["success", "failed"]
    assert manifest["items"][1]["event_file_sha256"] is None


def test_batch_rejects_generation_drift_before_materialisation(monkeypatch, tmp_path):
    vertices = SkyCoord(
        [10.0, 10.001, 10.0] * u.deg,
        [-9.0, -9.0, -8.999] * u.deg,
        frame="icrs",
    )
    selection = CelestialSelection.polygon(vertices)
    event = write_event(
        tmp_path / "m1.fits",
        instrument="EMOS1",
        obs_id="001",
        exposure="S001",
    )
    replacement = write_event(
        tmp_path / "replacement.fits",
        instrument="EMOS1",
        obs_id="001",
        exposure="S001",
    )
    with fits.open(replacement, mode="update") as hdus:
        hdus[0].header["TESTTAG"] = "new-generation"
        hdus.flush()
    assert read_event_identity(event).identity_sha256 == read_event_identity(replacement).identity_sha256
    assert file_sha256(event) != file_sha256(replacement)

    def replace_then_project(selection, *, calinfoset, context, rule):
        replacement.replace(event)
        return fake_project(
            selection,
            calinfoset=calinfoset,
            context=context,
            rule=rule,
        )

    monkeypatch.setattr("xmm_region_tool.batch.project_selection", replace_then_project)
    output = tmp_path / "out"
    result = convert_selection_batch(
        selection,
        [event],
        output,
        rule=ProjectionRule(samples=128),
        context=context(tmp_path),
    )

    assert not result.successful
    item = result.items[0]
    assert item.status == "failed"
    assert item.event_file_sha256 is None
    assert "generation changed" in item.error_message
    assert not list(output.glob("*.txt"))
    manifest = json.loads(result.manifest.read_text())
    assert manifest["items"][0]["event_file_sha256"] is None


def test_batch_manifest_serializes_nested_immutable_refinement_diagnostics(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.reg"
    source.write_text(
        "fk5\n"
        "polygon(10,-9,10.001,-9,10,-8.999)\n"
    )
    event = write_event(
        tmp_path / "m1.fits",
        instrument="EMOS1",
        obs_id="001",
        exposure="S001",
    )
    shared = context(tmp_path)

    diagnostics = {
        "total_vertices": 17,
        "max_accepted_error": 0.75,
        "boundaries": [
            {
                "region_index": 0,
                "boundary_index": 0,
                "vertices": 17,
                "max_accepted_error": 0.75,
                "segments": [
                    {
                        "depth": 2,
                        "accepted_error": 0.5,
                    }
                ],
            }
        ],
    }

    captured = []

    def project_with_nested_diagnostics(
        selection,
        *,
        calinfoset,
        context,
        rule,
    ):
        projected = fake_project(
            selection,
            calinfoset=calinfoset,
            context=context,
            rule=rule,
        )
        provenance = replace(
            projected.provenance,
            refinement_diagnostics=diagnostics,
        )
        captured.append(provenance)
        return ProjectionResult(
            selection=projected.selection,
            provenance=provenance,
        )

    monkeypatch.setattr(
        "xmm_region_tool.batch.project_selection",
        project_with_nested_diagnostics,
    )

    result = convert_batch(
        source,
        [event],
        tmp_path / "out",
        context=shared,
    )

    assert result.successful
    assert len(captured) == 1

    provenance = captured[0]

    with pytest.raises(TypeError):
        provenance.refinement_diagnostics["new"] = "value"

    boundary = provenance.refinement_diagnostics["boundaries"][0]
    with pytest.raises(TypeError):
        boundary["new"] = "value"

    returned = result.items[0].refinement_diagnostics
    assert returned["max_accepted_error"] == 0.75
    with pytest.raises(TypeError):
        returned["max_accepted_error"] = 0.0
    with pytest.raises(TypeError):
        returned["boundaries"][0]["segments"][0]["accepted_error"] = 0.0
    with pytest.raises(TypeError):
        returned["boundaries"][0] = {}

    payload = json.loads(result.manifest.read_text())
    assert payload["successful"] is True
    assert payload["items"][0]["refinement_diagnostics"] == diagnostics


def test_batch_manifest_retains_structured_sas_failure_evidence(monkeypatch, tmp_path):
    source = tmp_path / "source.reg"
    source.write_text("fk5\npolygon(10,-9,10.001,-9,10,-8.999)\n")
    event = write_event(
        tmp_path / "m1.fits",
        instrument="EMOS1",
        obs_id="001",
        exposure="S001",
    )

    def failing_project(selection, *, calinfoset, context, rule):
        raise SasConversionError(
            "synthetic projection failure",
            command=("/sas/bin/esky2det", "checkfov=no"),
            returncode=23,
            stdout="synthetic stdout",
            stderr="synthetic stderr",
        )

    monkeypatch.setattr("xmm_region_tool.batch.project_selection", failing_project)

    result = convert_batch(source, [event], tmp_path / "out", context=context(tmp_path))

    assert not result.successful
    assert result.items[0].failure_evidence is not None
    assert result.items[0].event_file_sha256 is None
    manifest = json.loads(result.manifest.read_text())
    failure = manifest["items"][0]["failure_evidence"]
    assert failure["returncode"] == 23
    assert failure["command"] == ["/sas/bin/esky2det", "checkfov=no"]
    assert failure["stdout"] == "synthetic stdout"
    assert failure["stderr"] == "synthetic stderr"
    assert manifest["items"][0]["event_file_sha256"] is None

    returned = result.items[0].failure_evidence
    with pytest.raises(TypeError):
        returned["returncode"] = 0
    with pytest.raises(TypeError):
        returned["command"][0] = "changed"
    assert list(returned["command"]) == failure["command"]
    assert returned["returncode"] == failure["returncode"]


@pytest.mark.parametrize("field", ["refinement_diagnostics", "failure_evidence"])
def test_batch_item_snapshots_caller_owned_evidence(tmp_path, field):
    source = {"nested": {"command": ["esky2det", {"option": "original"}]}}
    item = BatchItemResult(tmp_path / "event.fits", "failed", **{field: source})

    source["nested"]["command"][1]["option"] = "changed"
    source["nested"]["command"].append("extra")
    source["new"] = True

    stored = getattr(item, field)
    assert "new" not in stored
    assert len(stored["nested"]["command"]) == 2
    assert stored["nested"]["command"][1]["option"] == "original"
    with pytest.raises(TypeError):
        stored["nested"]["command"][1]["option"] = "changed"
