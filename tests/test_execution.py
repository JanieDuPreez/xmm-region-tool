from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.execution import (
    ProjectionProvenance,
    ProjectionRule,
    SasProjectionContext,
    SasProjectionContextError,
)
from xmm_region_tool.model import CelestialSelection
from xmm_region_tool.sas import SasConversionError, project_selection


def write_event(path: Path, *, marker: str = "same") -> Path:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    primary.header["INSTRUME"] = "EMOS1"
    primary.header["OBS_ID"] = "0144310101"
    primary.header["EXP_ID"] = "S001"
    primary.header["DATE-OBS"] = "2003-06-22T00:00:00"
    primary.header["RA_PNT"] = 15.673
    primary.header["DEC_PNT"] = -21.88
    primary.header["PA_PNT"] = 72.0
    primary.header["MARKER"] = marker
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="DETX", format="J", array=np.asarray([1, 2], dtype=np.int32)),
            fits.Column(name="DETY", format="J", array=np.asarray([3, 4], dtype=np.int32)),
        ],
        name="EVENTS",
    )
    fits.HDUList([primary, events]).writeto(path)
    return path


def selection() -> CelestialSelection:
    vertices = SkyCoord(
        [10.0, 10.001, 10.0] * u.deg,
        [-9.0, -9.0, -8.999] * u.deg,
        frame="icrs",
    )
    return CelestialSelection.polygon(vertices)


def context(tmp_path: Path) -> SasProjectionContext:
    executable = tmp_path / "esky2det"
    executable.write_text("fake executable bytes")
    cif = tmp_path / "ccf.cif"
    cif.write_text("fake cif bytes")
    calibration = CalibrationIdentity(
        cif_path=cif,
        cif_file_sha256="1" * 64,
        calindex_sha256="2" * 64,
        replacements=(),
        ccf_search_path=(),
    )
    producer = SasProducerIdentity(
        esky2det_version="esky2det-22.0",
        sas_version="SAS-22.0",
        esky2det_sha256="3" * 64,
        esky2det_name="esky2det",
    )
    return SasProjectionContext(
        environment={"PATH": str(tmp_path), "SAS_CCF": str(cif)},
        calibration=calibration,
        producer=producer,
        esky2det_path=executable,
    )


def _write_fake_projection_output(command) -> None:
    intab = next(value for value in command if value.startswith("intab="))
    path = Path(intab.removeprefix("intab=").split(":", maxsplit=1)[0])
    with fits.open(path) as hdus:
        source = hdus["INPUT"].data
        output = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="RA", format="D", array=source["RA"]),
                fits.Column(name="DEC", format="D", array=source["DEC"]),
                fits.Column(name="ROW_ID", format="J", array=source["ROW_ID"]),
                fits.Column(name="DETX", format="D", array=source["RA"] * 1000.0),
                fits.Column(name="DETY", format="D", array=source["DEC"] * 1000.0),
            ],
            name="INPUT",
        )
    fits.HDUList([fits.PrimaryHDU(), output]).writeto(path, overwrite=True)


def install_fake_esky2det(monkeypatch):
    def fake_run(command, **kwargs):
        _write_fake_projection_output(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", fake_run)

    # These unit fixtures deliberately contain fake CIF/executable identities.
    # Bypass real context re-resolution here; dedicated tests below exercise
    # SasProjectionContext.validate() itself.
    monkeypatch.setattr(SasProjectionContext, "validate", lambda self: None)


def test_projection_identity_is_stable_for_path_only_event_relocation(monkeypatch, tmp_path):
    install_fake_esky2det(monkeypatch)
    first = write_event(tmp_path / "first.fits")
    relocated = tmp_path / "relocated.fits"
    shutil.copyfile(first, relocated)
    ctx = context(tmp_path)
    rule = ProjectionRule(samples=128)

    a = project_selection(selection(), calinfoset=first, context=ctx, rule=rule)
    b = project_selection(selection(), calinfoset=relocated, context=ctx, rule=rule)

    assert a.provenance.event_file_sha256 == b.provenance.event_file_sha256
    assert a.provenance.event_identity_sha256 == b.provenance.event_identity_sha256
    assert a.provenance.projection_identity_sha256 == b.provenance.projection_identity_sha256
    assert a.provenance.command != b.provenance.command


def test_projection_identity_changes_when_event_bytes_change(monkeypatch, tmp_path):
    install_fake_esky2det(monkeypatch)
    first = write_event(tmp_path / "first.fits", marker="one")
    second = write_event(tmp_path / "second.fits", marker="two")
    ctx = context(tmp_path)
    rule = ProjectionRule(samples=128)

    a = project_selection(selection(), calinfoset=first, context=ctx, rule=rule)
    b = project_selection(selection(), calinfoset=second, context=ctx, rule=rule)

    assert a.provenance.event_identity_sha256 == b.provenance.event_identity_sha256
    assert a.provenance.event_file_sha256 != b.provenance.event_file_sha256
    assert a.provenance.projection_identity_sha256 != b.provenance.projection_identity_sha256


def test_projection_rejects_event_mutation_during_sas_execution(monkeypatch, tmp_path):
    event = write_event(tmp_path / "events.fits", marker="before")
    ctx = context(tmp_path)

    def fake_run(command, **kwargs):
        _write_fake_projection_output(command)
        with fits.open(event, mode="update") as hdus:
            hdus[0].header["MARKER"] = "after"
            hdus.flush()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", fake_run)
    monkeypatch.setattr(SasProjectionContext, "validate", lambda self: None)

    with pytest.raises(SasConversionError, match="changed while esky2det projection was running"):
        project_selection(
            selection(),
            calinfoset=event,
            context=ctx,
            rule=ProjectionRule(samples=128),
        )


def test_projection_identity_changes_with_producer_identity(tmp_path):
    ctx = context(tmp_path)
    changed = SasProjectionContext(
        environment=ctx.environment,
        calibration=ctx.calibration,
        producer=SasProducerIdentity(
            esky2det_version=ctx.producer.esky2det_version,
            sas_version=ctx.producer.sas_version,
            esky2det_sha256="4" * 64,
            esky2det_name="esky2det",
        ),
        esky2det_path=ctx.esky2det_path,
    )

    assert ctx.identity_sha256 != changed.identity_sha256


def test_sas_odf_does_not_change_esky2det_projection_context_identity(tmp_path):
    ctx = context(tmp_path)
    first_odf = tmp_path / "first.SAS"
    second_odf = tmp_path / "second.SAS"
    first_odf.write_text("first odf summary")
    second_odf.write_text("second odf summary")

    first = SasProjectionContext(
        environment={**ctx.environment, "SAS_ODF": str(first_odf)},
        calibration=ctx.calibration,
        producer=ctx.producer,
        esky2det_path=ctx.esky2det_path,
    )
    second = SasProjectionContext(
        environment={**ctx.environment, "SAS_ODF": str(second_odf)},
        calibration=ctx.calibration,
        producer=ctx.producer,
        esky2det_path=ctx.esky2det_path,
    )

    assert first.identity_sha256 == second.identity_sha256
    assert "SAS_ODF" not in first.relevant_environment_record()
    assert "SAS_ODF" not in second.relevant_environment_record()


def test_projection_context_environment_is_immutable(tmp_path):
    original = {"PATH": str(tmp_path), "SAS_CCF": str(tmp_path / "ccf.cif")}
    ctx = context(tmp_path)

    source = dict(original)
    immutable = SasProjectionContext(
        environment=source,
        calibration=ctx.calibration,
        producer=ctx.producer,
        esky2det_path=ctx.esky2det_path,
    )
    source["SAS_CCF"] = "/changed/elsewhere.cif"
    assert immutable.environment["SAS_CCF"] == original["SAS_CCF"]

    with pytest.raises(TypeError):
        immutable.environment["SAS_CCF"] = "/changed/directly.cif"


def test_projection_context_validate_accepts_matching_snapshot(monkeypatch, tmp_path):
    ctx = context(tmp_path)

    monkeypatch.setattr(
        "xmm_region_tool.execution.read_calibration_identity",
        lambda environment: ctx.calibration,
    )
    monkeypatch.setattr(
        "xmm_region_tool.execution.read_sas_producer_identity",
        lambda environment, esky2det="esky2det": ctx.producer,
    )

    ctx.validate()


def test_projection_context_validate_rejects_changed_calibration(monkeypatch, tmp_path):
    ctx = context(tmp_path)
    changed = CalibrationIdentity(
        cif_path=ctx.calibration.cif_path,
        cif_file_sha256=ctx.calibration.cif_file_sha256,
        calindex_sha256="9" * 64,
        replacements=ctx.calibration.replacements,
        ccf_search_path=ctx.calibration.ccf_search_path,
        constituents=ctx.calibration.constituents,
    )

    monkeypatch.setattr(
        "xmm_region_tool.execution.read_calibration_identity",
        lambda environment: changed,
    )
    monkeypatch.setattr(
        "xmm_region_tool.execution.read_sas_producer_identity",
        lambda environment, esky2det="esky2det": ctx.producer,
    )

    with pytest.raises(SasProjectionContextError, match="calibration identity no longer matches"):
        ctx.validate()


def test_projection_context_validate_rejects_changed_cif_bytes(monkeypatch, tmp_path):
    ctx = context(tmp_path)
    changed = CalibrationIdentity(
        cif_path=ctx.calibration.cif_path,
        cif_file_sha256="8" * 64,
        calindex_sha256=ctx.calibration.calindex_sha256,
        replacements=ctx.calibration.replacements,
        ccf_search_path=ctx.calibration.ccf_search_path,
        constituents=ctx.calibration.constituents,
    )

    monkeypatch.setattr(
        "xmm_region_tool.execution.read_calibration_identity",
        lambda environment: changed,
    )
    monkeypatch.setattr(
        "xmm_region_tool.execution.read_sas_producer_identity",
        lambda environment, esky2det="esky2det": ctx.producer,
    )

    with pytest.raises(SasProjectionContextError, match="SAS_CCF bytes no longer match"):
        ctx.validate()


def test_projection_context_validate_rejects_changed_producer(monkeypatch, tmp_path):
    ctx = context(tmp_path)
    changed = SasProducerIdentity(
        esky2det_version=ctx.producer.esky2det_version,
        sas_version=ctx.producer.sas_version,
        esky2det_sha256="7" * 64,
        esky2det_name=ctx.producer.esky2det_name,
    )

    monkeypatch.setattr(
        "xmm_region_tool.execution.read_calibration_identity",
        lambda environment: ctx.calibration,
    )
    monkeypatch.setattr(
        "xmm_region_tool.execution.read_sas_producer_identity",
        lambda environment, esky2det="esky2det": changed,
    )

    with pytest.raises(SasProjectionContextError, match="producer identity no longer matches"):
        ctx.validate()


def test_project_selection_revalidates_context_after_projection(monkeypatch, tmp_path):
    event = write_event(tmp_path / "events.fits")
    ctx = context(tmp_path)

    def fake_run(command, **kwargs):
        _write_fake_projection_output(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xmm_region_tool.sas.subprocess.run", fake_run)

    calls = {"count": 0}

    def fake_validate(self):
        calls["count"] += 1
        if calls["count"] == 2:
            raise SasProjectionContextError(
                "producer identity no longer matches the frozen projection context"
            )

    monkeypatch.setattr(SasProjectionContext, "validate", fake_validate)

    with pytest.raises(SasProjectionContextError, match="producer identity no longer matches"):
        project_selection(
            selection(),
            calinfoset=event,
            context=ctx,
            rule=ProjectionRule(samples=128),
        )

    assert calls["count"] == 2


def test_projection_rule_rejects_unsupported_method():
    with pytest.raises(ValueError, match="unsupported projection method"):
        ProjectionRule(method="something-else")


def test_legacy_projection_rule_identity_ignores_inactive_adaptive_controls():
    baseline = ProjectionRule(samples=128)
    changed_inactive = ProjectionRule(
        samples=128,
        detector_tolerance=999.0,
        max_depth=1,
        max_vertices=4,
    )

    assert baseline.canonical_record() == changed_inactive.canonical_record()
    assert baseline.canonical_record() == {
        "schema": "xmm-region-tool.projection-rule/v2",
        "method": "sas-esky2det-boundary/v2",
        "refinement": "legacy-fixed-source-sampling/v1",
        "samples": 128,
    }


def test_legacy_projection_rule_identity_changes_with_samples():
    first = ProjectionRule(samples=64)
    second = ProjectionRule(samples=128)

    assert first.canonical_record() != second.canonical_record()


def test_adaptive_projection_rule_identity_contains_only_active_controls():
    rule = ProjectionRule(
        detector_tolerance=0.5,
        max_depth=8,
        max_vertices=2048,
    )

    assert rule.canonical_record() == {
        "schema": "xmm-region-tool.projection-rule/v2",
        "method": "sas-esky2det-boundary/v2",
        "refinement": "adaptive-detector-chord/v2",
        "detector_tolerance": 0.5,
        "max_depth": 8,
        "max_vertices": 2048,
    }
    assert "samples" not in rule.canonical_record()


@pytest.mark.parametrize(
    ("field", "first_value", "second_value"),
    [
        ("detector_tolerance", 0.5, 0.25),
        ("max_depth", 8, 9),
        ("max_vertices", 2048, 4096),
    ],
)
def test_adaptive_projection_rule_identity_changes_with_active_control(
    field,
    first_value,
    second_value,
):
    kwargs = {
        "detector_tolerance": 0.5,
        "max_depth": 8,
        "max_vertices": 2048,
    }
    first_kwargs = {**kwargs, field: first_value}
    second_kwargs = {**kwargs, field: second_value}

    first = ProjectionRule(**first_kwargs)
    second = ProjectionRule(**second_kwargs)

    assert first.canonical_record() != second.canonical_record()


def test_projection_identity_schema_explicitly_bumped_for_normalized_rule(tmp_path):
    ctx = context(tmp_path)

    provenance = ProjectionProvenance(
        event_file_sha256="1" * 64,
        event_identity_sha256="2" * 64,
        celestial_geometry_sha256="3" * 64,
        detector_geometry_sha256="4" * 64,
        context_identity_sha256=ctx.identity_sha256,
        calibration_identity_sha256=ctx.calibration.identity_sha256,
        producer_identity_sha256=ctx.producer.identity_sha256,
        rule=ProjectionRule(samples=128),
        command=(),
        relevant_environment={},
    )

    assert provenance.canonical_projection_record()["schema"] == "xmm-region-tool.projection/v3"


def test_projection_provenance_evidence_mappings_are_immutable_snapshots(tmp_path):
    ctx = context(tmp_path)
    environment = {"SAS_CCF": "/original/ccf.cif"}
    diagnostics = {"adaptive": {"max_error": 0.25}, "counts": [1, 2]}

    provenance = ProjectionProvenance(
        event_file_sha256="1" * 64,
        event_identity_sha256="2" * 64,
        celestial_geometry_sha256="3" * 64,
        detector_geometry_sha256="4" * 64,
        context_identity_sha256=ctx.identity_sha256,
        calibration_identity_sha256=ctx.calibration.identity_sha256,
        producer_identity_sha256=ctx.producer.identity_sha256,
        rule=ProjectionRule(samples=128),
        command=(),
        relevant_environment=environment,
        refinement_diagnostics=diagnostics,
    )

    environment["SAS_CCF"] = "/mutated/ccf.cif"
    diagnostics["adaptive"]["max_error"] = 99.0
    diagnostics["counts"].append(3)

    assert provenance.relevant_environment["SAS_CCF"] == "/original/ccf.cif"
    assert provenance.refinement_diagnostics["adaptive"]["max_error"] == 0.25
    assert provenance.refinement_diagnostics["counts"] == (1, 2)

    with pytest.raises(TypeError):
        provenance.relevant_environment["SAS_CCF"] = "/direct/mutation.cif"

    with pytest.raises(TypeError):
        provenance.refinement_diagnostics["adaptive"]["max_error"] = 10.0

    evidence = provenance.evidence_record()
    assert evidence["refinement_diagnostics"] == {
        "adaptive": {"max_error": 0.25},
        "counts": [1, 2],
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1", [1]])
def test_projection_rule_rejects_invalid_tolerance(value):
    with pytest.raises(ValueError, match="detector_tolerance"):
        ProjectionRule(detector_tolerance=value)


@pytest.mark.parametrize("field", ["max_depth", "max_vertices", "samples"])
@pytest.mark.parametrize("value", [True, False, 128.0, float("nan"), float("inf"), "128"])
def test_projection_rule_rejects_noninteger_limits(field, value):
    with pytest.raises(ValueError, match=field):
        ProjectionRule(**{field: value})


def test_projection_rejects_duck_typed_rule_before_execution(tmp_path, monkeypatch):
    def unexpected_validate(self):
        pytest.fail("invalid rule reached execution")

    monkeypatch.setattr(SasProjectionContext, "validate", unexpected_validate)
    with pytest.raises(TypeError, match="rule must be ProjectionRule"):
        project_selection(
            selection(), calinfoset=tmp_path / "missing", context=context(tmp_path),
            rule=SimpleNamespace(detector_tolerance=float("inf")),
        )


@pytest.mark.parametrize(
    "field,value",
    [("max_depth", 12.0), ("max_vertices", True), ("samples", 128.0)],
)
def test_managed_rule_rejects_noncanonical_integer_types(field, value):
    from xmm_region_tool.managed import ArtifactMaterializationError, _canonical_rule_record

    record = ProjectionRule(samples=128 if field == "samples" else None).canonical_record()
    record[field] = value
    with pytest.raises(ArtifactMaterializationError, match="projection-rule"):
        _canonical_rule_record(record)


def test_adaptive_initial_intervals_respect_vertex_cap_before_sas(monkeypatch):
    from xmm_region_tool.sas import _adaptive_boundary

    def unexpected_project(*args, **kwargs):
        pytest.fail("oversized initial boundary reached SAS")

    monkeypatch.setattr("xmm_region_tool.sas._project_skycoords", unexpected_project)
    vertices = SkyCoord([10, 11, 12, 11, 10] * u.deg, [0, 0, 1, 2, 1] * u.deg)
    boundary = CelestialSelection.polygon(vertices).regions[0].boundaries[0]
    with pytest.raises(SasConversionError, match="max_vertices"):
        _adaptive_boundary(
            boundary,
            rule=ProjectionRule(max_vertices=4),
            calinfoset="unused",
            esky2det="unused",
            environment={},
        )


def test_context_does_not_print_inherited_environment(tmp_path):
    from dataclasses import replace
    import json

    original = context(tmp_path)
    environment = dict(original.environment, XMM_REGION_TOOL_TEST_SECRET="do-not-print")
    captured = replace(original, environment=environment)
    environment["XMM_REGION_TOOL_TEST_SECRET"] = "changed"
    assert captured.environment["XMM_REGION_TOOL_TEST_SECRET"] == "do-not-print"
    assert "do-not-print" not in repr(captured)
    assert "XMM_REGION_TOOL_TEST_SECRET" not in repr(captured)
    assert "do-not-print" not in json.dumps(captured.canonical_record())
    assert "do-not-print" not in json.dumps(captured.relevant_environment_record())
    assert captured.identity_sha256 == original.identity_sha256
    assert captured.environment["SAS_CCF"] == original.environment["SAS_CCF"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("detector_tolerance", float("nan")),
        ("detector_tolerance", float("inf")),
        ("detector_tolerance", None),
        ("detector_tolerance", True),
        ("max_depth", 0),
        ("max_depth", 2.0),
        ("max_depth", True),
        ("max_vertices", 1),
        ("max_vertices", float("inf")),
    ],
)
def test_legacy_rule_validates_inactive_controls(field, value):
    with pytest.raises(ValueError, match=field):
        ProjectionRule(samples=128, **{field: value})
