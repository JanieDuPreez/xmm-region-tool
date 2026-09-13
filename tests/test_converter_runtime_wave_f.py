from __future__ import annotations

from importlib.metadata import version
from types import MappingProxyType

import pytest

import xmm_region_tool
from xmm_region_tool.execution import ProjectionProvenance, ProjectionRule
from xmm_region_tool.runtime_producer import converter_runtime_record


def _provenance(*, runtime=None):
    kwargs = {}
    if runtime is not None:
        kwargs["converter_runtime"] = runtime
    return ProjectionProvenance(
        event_file_sha256="1" * 64,
        event_identity_sha256="2" * 64,
        celestial_geometry_sha256="3" * 64,
        detector_geometry_sha256="4" * 64,
        context_identity_sha256="5" * 64,
        calibration_identity_sha256="6" * 64,
        producer_identity_sha256="7" * 64,
        rule=ProjectionRule(samples=128),
        command=("esky2det",),
        relevant_environment={},
        **kwargs,
    )


def test_top_level_version_uses_installed_distribution_metadata() -> None:
    assert xmm_region_tool.__version__ == version("xmm-region-tool")


def test_converter_runtime_record_contains_required_stable_identifiers() -> None:
    record = converter_runtime_record()

    assert record["schema"] == "xmm-region-tool.converter-runtime/v1"
    assert record["package"]["distribution"] == "xmm-region-tool"
    assert record["package"]["version"] == version("xmm-region-tool")
    assert len(record["package"]["source_sha256"]) == 64
    int(record["package"]["source_sha256"], 16)
    assert record["python"]["implementation"]
    assert record["python"]["version"]
    assert record["dependencies"]["astropy"]
    assert record["dependencies"]["numpy"]
    assert record["dependencies"]["regions"]


def test_projection_provenance_snapshots_converter_runtime_immutably() -> None:
    provenance = _provenance()

    assert isinstance(provenance.converter_runtime, MappingProxyType)
    assert isinstance(provenance.converter_runtime["package"], MappingProxyType)
    with pytest.raises(TypeError):
        provenance.converter_runtime["package"]["version"] = "tampered"

    evidence = provenance.evidence_record()
    assert evidence["converter_runtime"]["schema"] == "xmm-region-tool.converter-runtime/v1"
    assert evidence["converter_runtime"]["package"]["version"] == xmm_region_tool.__version__


def test_converter_runtime_is_nonsemantic_projection_evidence() -> None:
    base = converter_runtime_record()
    alternate = converter_runtime_record()
    alternate["python"]["version"] = "different-runtime"

    first = _provenance(runtime=base)
    second = _provenance(runtime=alternate)

    assert first.projection_identity_sha256 == second.projection_identity_sha256
    assert first.evidence_record()["converter_runtime"] != second.evidence_record()["converter_runtime"]


def test_converter_runtime_rejects_missing_schema() -> None:
    runtime = converter_runtime_record()
    runtime.pop("schema")

    with pytest.raises(ValueError, match="converter_runtime"):
        _provenance(runtime=runtime)
