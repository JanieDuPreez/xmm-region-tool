from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from xmm_region_tool import cli
from xmm_region_tool.batch import event_output_label
from xmm_region_tool.path_safety import (
    PathSafetyError,
    paths_alias,
    resolved_descendant,
    safe_filename_component,
    validate_output_namespace,
)
from xmm_region_tool.workflow import ExtractionCell, output_basename, safe_source_stem


def test_safe_filename_component_removes_directory_semantics():
    assert safe_filename_component("../../../outside/target") == "outside-target"
    assert safe_filename_component(r"..\\..\\outside\\target") == "outside-target"
    assert safe_filename_component("...strange name?!...") == "strange-name"


def test_safe_filename_component_rejects_empty_metadata_component():
    with pytest.raises(PathSafetyError):
        safe_filename_component("../..")


def test_safe_source_stem_retains_fallback_for_empty_source_name():
    assert safe_source_stem("...reg") == "region"


def test_output_basename_sanitizes_hostile_exposure_metadata():
    cell = ExtractionCell(index=1, label="r001", selection=object())
    identity = SimpleNamespace(
        instrument="mos1",
        exposure_id="../../../outside/target",
        obs_id="0144310101",
    )

    basename = output_basename(
        "source.reg",
        cell,
        identity,
        cell_count=1,
        duplicate_instrument=True,
    )

    assert basename == "source-mos1-outside-target"
    assert "/" not in basename
    assert "\\" not in basename
    assert ".." not in basename


def test_output_basename_sanitizes_hostile_obsid_fallback():
    cell = ExtractionCell(index=1, label="r001", selection=object())
    identity = SimpleNamespace(
        instrument="mos1",
        exposure_id=None,
        obs_id=r"..\\..\\outside\\obs",
    )

    basename = output_basename(
        "source.reg",
        cell,
        identity,
        cell_count=1,
        duplicate_instrument=True,
    )

    assert basename == "source-mos1-outside-obs"


def test_batch_label_uses_same_safe_component_policy():
    identity = SimpleNamespace(
        instrument="mos1",
        exposure_id="../../../outside/target",
        obs_id=r"..\\obs?!",
    )

    label = event_output_label(identity, Path("mos1.fits"), "a" * 64)

    assert label == f"obs-mos1-outside-target-{'a' * 12}"
    assert "/" not in label
    assert "\\" not in label
    assert ".." not in label


def test_resolved_descendant_rejects_automatic_escape(tmp_path):
    root = tmp_path / "output"
    escaped = root / ".." / "outside" / "region.txt"

    with pytest.raises(PathSafetyError, match="escapes output root"):
        resolved_descendant(root, escaped, role="region wrapper")


def test_resolved_descendant_accepts_safe_automatic_child(tmp_path):
    root = tmp_path / "output"
    child = root / "source-mos1-S001.txt"

    assert resolved_descendant(root, child, role="region wrapper") == child.resolve()


def test_paths_alias_detects_existing_hardlink(tmp_path):
    original = tmp_path / "event.fits"
    alias = tmp_path / "alias.fits"
    original.write_bytes(b"science")
    os.link(original, alias)

    assert original.resolve() != alias.resolve()
    assert paths_alias(original, alias)


def test_output_namespace_rejects_hardlink_to_protected_input(tmp_path):
    event = tmp_path / "event.fits"
    output = tmp_path / "output.fits"
    event.write_bytes(b"science")
    os.link(event, output)

    with pytest.raises(PathSafetyError, match="aliases protected science event"):
        validate_output_namespace(
            (("FITS geometry", output),),
            (("science event", event),),
        )


def test_output_namespace_rejects_generated_role_collision(tmp_path):
    path = tmp_path / "same.txt"
    with pytest.raises(PathSafetyError, match="aliases generated workflow manifest"):
        validate_output_namespace(
            (("region wrapper", path), ("workflow manifest", path)),
            (),
        )


def test_default_output_fails_closed_if_name_builder_attempts_escape(
    monkeypatch,
    tmp_path,
):
    output_root = tmp_path / "output"
    args = SimpleNamespace(output_dir=output_root, region=tmp_path / "source.reg")
    cell = ExtractionCell(index=1, label="r001", selection=object())
    identity = SimpleNamespace(
        instrument="mos1",
        exposure_id="S001",
        obs_id="0144310101",
    )
    monkeypatch.setattr(cli, "output_basename", lambda *args, **kwargs: "../../outside/target")

    with pytest.raises(PathSafetyError, match="escapes output root"):
        cli._default_output(
            args,
            cell,
            identity,
            cell_count=1,
            duplicate_instrument=True,
        )

    assert not output_root.exists()
    assert not (tmp_path / "outside").exists()
