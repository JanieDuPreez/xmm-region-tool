from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import xmm_region_tool.managed as managed
import xmm_region_tool.publication as publication
import xmm_region_tool.transactional_products as transactional
from xmm_region_tool.artifacts import ArtifactMaterializationError, DetectorGeometryArtifact
from xmm_region_tool.output import WrittenRegionFiles


def test_atomic_publication_replaces_destination_symlink_without_touching_victim(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("sentinel\n")
    final = tmp_path / "output.txt"
    final.symlink_to(victim)
    stage = publication.temporary_sibling(final)
    stage.write_text("new output\n")

    publication.publish_staged_files(((stage, final),))

    assert victim.read_text() == "sentinel\n"
    assert not final.is_symlink()
    assert final.read_text() == "new output\n"


def test_multi_file_publish_restores_previous_generation_on_mid_publish_failure(
    monkeypatch,
    tmp_path,
):
    final_a = tmp_path / "a.txt"
    final_b = tmp_path / "b.txt"
    final_a.write_text("old-a\n")
    final_b.write_text("old-b\n")
    stage_a = publication.temporary_sibling(final_a)
    stage_b = publication.temporary_sibling(final_b)
    stage_a.write_text("new-a\n")
    stage_b.write_text("new-b\n")

    real_replace = publication.os.replace
    calls = 0

    def fail_second_publish(src, dst):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("synthetic publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(publication.os, "replace", fail_second_publish)

    with pytest.raises(publication.PublicationError, match="synthetic publish failure"):
        publication.publish_staged_files(((stage_a, final_a), (stage_b, final_b)))

    assert final_a.read_text() == "old-a\n"
    assert final_b.read_text() == "old-b\n"
    assert not stage_a.exists()
    assert not stage_b.exists()
    assert not list(tmp_path.glob("*.xmm-region-backup-*"))


def test_multi_file_publish_preserves_recovery_backup_if_rollback_restore_fails(
    monkeypatch,
    tmp_path,
):
    final_a = tmp_path / "a.txt"
    final_b = tmp_path / "b.txt"
    final_a.write_text("old-a\n")
    final_b.write_text("old-b\n")
    stage_a = publication.temporary_sibling(final_a)
    stage_b = publication.temporary_sibling(final_b)
    stage_a.write_text("new-a\n")
    stage_b.write_text("new-b\n")

    real_replace = publication.os.replace
    calls = 0

    def fail_publish_then_first_restore(src, dst):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("synthetic publish failure")
        if calls == 5:
            raise OSError("synthetic restore failure")
        return real_replace(src, dst)

    monkeypatch.setattr(publication.os, "replace", fail_publish_then_first_restore)

    with pytest.raises(publication.PublicationError) as exc_info:
        publication.publish_staged_files(((stage_a, final_a), (stage_b, final_b)))

    assert "synthetic publish failure" in str(exc_info.value)
    assert "synthetic restore failure" in str(exc_info.value)
    assert str(final_a) in str(exc_info.value)
    assert not final_a.exists()
    assert final_b.read_text() == "old-b\n"

    recovery = list(tmp_path.glob(".a.txt.xmm-region-backup-*"))
    assert len(recovery) == 1
    assert recovery[0].read_text() == "old-a\n"
    assert str(recovery[0]) in str(exc_info.value)
    assert not stage_a.exists()
    assert not stage_b.exists()


def test_atomic_text_write_does_not_follow_existing_symlink(tmp_path):
    victim = tmp_path / "manifest-victim.json"
    victim.write_text('{"old": true}\n')
    manifest = tmp_path / "manifest.json"
    manifest.symlink_to(victim)

    publication.atomic_write_text(manifest, '{"new": true}\n')

    assert victim.read_text() == '{"old": true}\n'
    assert not manifest.is_symlink()
    assert manifest.read_text() == '{"new": true}\n'


def _fake_fits_serializer(output, regions, **kwargs):
    output = Path(output)
    output.write_text("staged wrapper\n")
    geometry = transactional.planned_fits_output(output)
    geometry.write_bytes(b"staged fits")
    return WrittenRegionFiles(output, "fits", geometry)


def _fake_evidence_writer(path, result, *, geometry_artifact=None, **kwargs):
    path = Path(path)
    payload = {
        "geometry_artifact": {
            "path": str(geometry_artifact),
            "file_sha256": "a" * 64,
            "geometry_sha256": "b" * 64,
        }
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def _projected_stub():
    return SimpleNamespace(selection=SimpleNamespace(regions=()))


def test_product_evidence_failure_preserves_previous_generation(monkeypatch, tmp_path):
    output = tmp_path / "region.txt"
    geometry = transactional.planned_fits_output(output)
    evidence = output.with_suffix(".provenance.json")
    output.write_text("old wrapper\n")
    geometry.write_bytes(b"old fits")
    evidence.write_text("old evidence\n")

    monkeypatch.setattr(transactional, "write_esas_regionfile", _fake_fits_serializer)

    def fail_evidence(*args, **kwargs):
        raise OSError("synthetic evidence failure")

    monkeypatch.setattr(transactional, "write_projection_evidence", fail_evidence)

    with pytest.raises(OSError, match="synthetic evidence failure"):
        transactional.write_projected_product_atomic(output, _projected_stub())

    assert output.read_text() == "old wrapper\n"
    assert geometry.read_bytes() == b"old fits"
    assert evidence.read_text() == "old evidence\n"
    assert not list(tmp_path.glob("*.xmm-region-stage-*"))


def test_product_precommit_failure_leaves_no_first_generation(monkeypatch, tmp_path):
    output = tmp_path / "region.txt"
    geometry = transactional.planned_fits_output(output)
    evidence = output.with_suffix(".provenance.json")
    monkeypatch.setattr(transactional, "write_esas_regionfile", _fake_fits_serializer)
    monkeypatch.setattr(transactional, "write_projection_evidence", _fake_evidence_writer)

    def fail_precommit(written):
        raise ValueError("synthetic pn OOT ambiguity")

    with pytest.raises(ValueError, match="pn OOT ambiguity"):
        transactional.write_projected_product_atomic(
            output,
            _projected_stub(),
            precommit=fail_precommit,
        )

    assert not output.exists()
    assert not geometry.exists()
    assert not evidence.exists()
    assert not list(tmp_path.glob("*.xmm-region-stage-*"))


def test_successful_product_publishes_only_final_paths(monkeypatch, tmp_path):
    output = tmp_path / "region.txt"
    final_geometry = transactional.planned_fits_output(output)
    monkeypatch.setattr(transactional, "write_esas_regionfile", _fake_fits_serializer)
    monkeypatch.setattr(transactional, "write_projection_evidence", _fake_evidence_writer)

    written, evidence, command = transactional.write_projected_product_atomic(
        output,
        _projected_stub(),
        precommit=lambda product: f"consume {product.regionfile}",
    )

    assert written.regionfile == output
    assert written.fits_region == final_geometry
    assert command == f"consume {output}"
    assert output.read_text() == f"&&region({final_geometry},DETX,DETY)\n"
    payload = json.loads(evidence.read_text())
    assert payload["geometry_artifact"]["path"] == str(final_geometry)
    assert "xmm-region-stage" not in output.read_text()
    assert "xmm-region-stage" not in evidence.read_text()
    assert not list(tmp_path.glob("*.xmm-region-stage-*"))


def test_managed_validation_failure_preserves_previous_pair(monkeypatch, tmp_path):
    final_geometry = tmp_path / "managed.fits"
    final_evidence = final_geometry.with_suffix(".provenance.json")
    final_geometry.write_bytes(b"old managed geometry")
    final_evidence.write_text("old managed evidence\n")

    def fake_write_detector_geometry(path, result, **kwargs):
        geometry = Path(path)
        geometry.write_bytes(b"new staged geometry")
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

    def reject_staged(artifact):
        raise ArtifactMaterializationError("synthetic managed validation failure")

    monkeypatch.setattr(managed, "_validate_projection_evidence", reject_staged)

    provenance = SimpleNamespace(
        event_file_sha256="1" * 64,
        event_identity_sha256="2" * 64,
        calibration_identity_sha256="3" * 64,
        producer_identity_sha256="4" * 64,
        celestial_geometry_sha256="5" * 64,
        projection_identity_sha256="6" * 64,
    )

    with pytest.raises(ArtifactMaterializationError, match="managed validation failure"):
        managed.write_bound_detector_geometry(
            final_geometry,
            SimpleNamespace(provenance=provenance),
            event_identity=object(),
            calibration_identity=object(),
            sas_producer=object(),
        )

    assert final_geometry.read_bytes() == b"old managed geometry"
    assert final_evidence.read_text() == "old managed evidence\n"
    assert not list(tmp_path.glob("*.xmm-region-stage-*"))
