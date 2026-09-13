from __future__ import annotations

import json

import pytest

from xmm_region_tool import check_cli
from xmm_region_tool.limits import MAX_SOURCE_BYTES


_CELESTIAL = "a" * 64
_PROJECTION = "b" * 64


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_workflow_v1_uses_location_bound_absolute_product_path(tmp_path):
    target = (tmp_path / "out" / "region.fits").resolve()
    manifest = tmp_path / "workflow.json"
    _write_json(
        manifest,
        {
            "schema": "xmm-region-tool.workflow/v1",
            "products": [
                {
                    "status": "success",
                    "fits_region": str(target),
                    "celestial_geometry_sha256": _CELESTIAL,
                    "projection_identity_sha256": _PROJECTION,
                }
            ],
        },
    )

    assert check_cli._manifest_expectations(manifest, target) == (
        _CELESTIAL,
        _PROJECTION,
    )

    moved_target = (tmp_path / "moved" / "region.fits").resolve()
    with pytest.raises(ValueError, match="found 0"):
        check_cli._manifest_expectations(manifest, moved_target)


def test_workflow_v1_rejects_relative_product_path(tmp_path):
    manifest = tmp_path / "workflow.json"
    _write_json(
        manifest,
        {
            "schema": "xmm-region-tool.workflow/v1",
            "products": [
                {
                    "status": "success",
                    "fits_region": "region.fits",
                    "celestial_geometry_sha256": _CELESTIAL,
                    "projection_identity_sha256": _PROJECTION,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="must be absolute"):
        check_cli._manifest_expectations(manifest, tmp_path / "region.fits")


def test_batch_v4_resolves_product_relative_to_manifest_directory(tmp_path):
    root = tmp_path / "relocated-batch"
    target = (root / "products" / "region.fits").resolve()
    manifest = root / "xmm-region-manifest.json"
    _write_json(
        manifest,
        {
            "schema": "xmm-region-tool.batch/v4",
            "items": [
                {
                    "status": "success",
                    "fits_region": "products/region.fits",
                    "celestial_geometry_sha256": _CELESTIAL,
                    "projection_identity_sha256": _PROJECTION,
                }
            ],
        },
    )

    assert check_cli._manifest_expectations(manifest, target) == (
        _CELESTIAL,
        _PROJECTION,
    )


def test_batch_v4_rejects_absolute_product_path(tmp_path):
    target = (tmp_path / "region.fits").resolve()
    manifest = tmp_path / "xmm-region-manifest.json"
    _write_json(
        manifest,
        {
            "schema": "xmm-region-tool.batch/v4",
            "items": [
                {
                    "status": "success",
                    "fits_region": str(target),
                    "celestial_geometry_sha256": _CELESTIAL,
                    "projection_identity_sha256": _PROJECTION,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="must be relative"):
        check_cli._manifest_expectations(manifest, target)


def test_batch_v4_matching_failed_item_is_rejected(tmp_path):
    target = (tmp_path / "region.fits").resolve()
    manifest = tmp_path / "xmm-region-manifest.json"
    _write_json(
        manifest,
        {
            "schema": "xmm-region-tool.batch/v4",
            "items": [
                {
                    "status": "failed",
                    "fits_region": "region.fits",
                    "celestial_geometry_sha256": _CELESTIAL,
                    "projection_identity_sha256": _PROJECTION,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="not successful"):
        check_cli._manifest_expectations(manifest, target)


def test_batch_v4_rejects_ambiguous_matching_items(tmp_path):
    target = (tmp_path / "region.fits").resolve()
    item = {
        "status": "success",
        "fits_region": "region.fits",
        "celestial_geometry_sha256": _CELESTIAL,
        "projection_identity_sha256": _PROJECTION,
    }
    manifest = tmp_path / "xmm-region-manifest.json"
    _write_json(
        manifest,
        {
            "schema": "xmm-region-tool.batch/v4",
            "items": [dict(item), dict(item)],
        },
    )

    with pytest.raises(ValueError, match="found 2"):
        check_cli._manifest_expectations(manifest, target)


@pytest.mark.parametrize(
    "missing_key",
    ["celestial_geometry_sha256", "projection_identity_sha256"],
)
def test_batch_v4_matching_item_requires_both_identities(tmp_path, missing_key):
    target = (tmp_path / "region.fits").resolve()
    item = {
        "status": "success",
        "fits_region": "region.fits",
        "celestial_geometry_sha256": _CELESTIAL,
        "projection_identity_sha256": _PROJECTION,
    }
    del item[missing_key]
    manifest = tmp_path / "xmm-region-manifest.json"
    _write_json(
        manifest,
        {"schema": "xmm-region-tool.batch/v4", "items": [item]},
    )

    with pytest.raises(ValueError, match="no valid"):
        check_cli._manifest_expectations(manifest, target)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        "manifest",
    ],
)
def test_checker_rejects_non_object_manifest_top_levels_without_traceback(
    tmp_path,
    capsys,
    payload,
):
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, payload)

    status = check_cli.main(
        [
            str(tmp_path / "region.fits"),
            "--event-file",
            str(tmp_path / "event.fits"),
            "--manifest",
            str(manifest),
        ]
    )

    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "top level must be a JSON object" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("schema", [None, "xmm-region-tool.unknown/v1"])
def test_checker_rejects_missing_or_unsupported_manifest_schema(tmp_path, schema):
    manifest = tmp_path / "manifest.json"
    payload = {"products": []}
    if schema is not None:
        payload["schema"] = schema
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="unsupported or missing manifest schema"):
        check_cli._manifest_expectations(manifest, tmp_path / "region.fits")


def test_checker_rejects_malformed_entries_instead_of_skipping_them(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "schema": "xmm-region-tool.batch/v4",
            "items": ["not-an-object"],
        },
    )

    with pytest.raises(ValueError, match="item 0 is not an object"):
        check_cli._manifest_expectations(manifest, tmp_path / "region.fits")


def test_checker_rejects_oversized_manifest_before_json_parse(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{" + b" " * MAX_SOURCE_BYTES + b"}")

    status = check_cli.main(
        [
            str(tmp_path / "region.fits"),
            "--event-file",
            str(tmp_path / "event.fits"),
            "--manifest",
            str(manifest),
        ]
    )

    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read manifest" in captured.err
    assert "Traceback" not in captured.err
