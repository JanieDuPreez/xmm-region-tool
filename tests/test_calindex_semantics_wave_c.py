from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.calibration import (
    CalibrationIdentityError,
    _read_calindex,
    read_calibration_identity,
)


ROWS = (
    {
        "TELESCOP": "XMM",
        "SCOPE": "EMOS1",
        "TYPEID": "LINCOORD",
        "ISSUE": 19,
        "VALDATE": "1998-01-01T00:00:00",
        "VALDATE-END": "",
        "FNAME": "EMOS1_LINCOORD_0019.CCF",
        "DATE": "2024-03-14T19:00:05",
        "SUBDATE": "2024-04-29T14:02:06",
        "EXTSEQU": "ANY",
        "EXTSEQID": "LINCOORD FOV CORNER",
        "CREATOR": "deceit-a",
    },
    {
        "TELESCOP": "XMM",
        "SCOPE": "XMM",
        "TYPEID": "BORESIGHT",
        "ISSUE": 35,
        "VALDATE": "2000-01-01T00:00:00",
        "VALDATE-END": "",
        "FNAME": "XMM_BORESIGHT_0035.CCF",
        "DATE": "2025-04-02T14:00:09",
        "SUBDATE": "2025-05-29T18:02:03",
        "EXTSEQU": "ANY",
        "EXTSEQID": "BORESIGHT EMOS1_ANGVAR EMOS2_ANGVAR EPN_ANGVAR",
        "CREATOR": "deceit-b",
    },
)


def _md5(data: bytes) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    digest.update(data)
    return digest.hexdigest()


def _write_fixture(
    root: Path,
    *,
    rows: tuple[dict[str, object], ...] = ROWS,
    name: str = "ccf.cif",
) -> tuple[Path, Path]:
    ccf = root / "ccf"
    ccf.mkdir(parents=True, exist_ok=True)
    normalized = []
    for row in rows:
        current = dict(row)
        data = f"bytes:{current['FNAME']}".encode()
        (ccf / str(current["FNAME"])).write_bytes(data)
        current["FSIZE"] = len(data)
        current["MD5"] = _md5(data)
        normalized.append(current)

    columns = [
        fits.Column(name="TELESCOP", format="4A", array=[r["TELESCOP"] for r in normalized]),
        fits.Column(name="SCOPE", format="6A", array=[r["SCOPE"] for r in normalized]),
        fits.Column(name="TYPEID", format="32A", array=[r["TYPEID"] for r in normalized]),
        fits.Column(name="ISSUE", format="I", array=np.asarray([r["ISSUE"] for r in normalized], dtype=np.int16)),
        fits.Column(name="VALDATE", format="19A", array=[r["VALDATE"] for r in normalized]),
        fits.Column(name="VALDATE-END", format="19A", array=[r["VALDATE-END"] for r in normalized]),
        fits.Column(name="FNAME", format="256A", array=[r["FNAME"] for r in normalized]),
        fits.Column(name="DATE", format="19A", array=[r["DATE"] for r in normalized]),
        fits.Column(name="FSIZE", format="J", array=np.asarray([r["FSIZE"] for r in normalized], dtype=np.int32)),
        fits.Column(name="SUBDATE", format="19A", array=[r["SUBDATE"] for r in normalized]),
        fits.Column(name="EXTSEQU", format="32A", array=[r["EXTSEQU"] for r in normalized]),
        fits.Column(name="EXTSEQID", format="256A", array=[r["EXTSEQID"] for r in normalized]),
        fits.Column(name="MD5", format="32A", array=[r["MD5"] for r in normalized]),
        fits.Column(name="CREATOR", format="64A", array=[r["CREATOR"] for r in normalized]),
    ]
    path = root / name
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="CALINDEX")]).writeto(path)
    return path, ccf


def _identity(cif: Path, ccf: Path):
    return read_calibration_identity({"SAS_CCF": str(cif), "SAS_CCFPATH": str(ccf)})


def test_extseqid_is_semantic_identity_material(tmp_path):
    first_cif, first_ccf = _write_fixture(tmp_path / "first")
    changed = [dict(row) for row in ROWS]
    changed[0]["EXTSEQID"] = "LINCOORD FOV CORNER EXTRA"
    second_cif, second_ccf = _write_fixture(tmp_path / "second", rows=tuple(changed))

    first = _identity(first_cif, first_ccf)
    second = _identity(second_cif, second_ccf)

    assert first.calindex_sha256 != second.calindex_sha256
    assert first.identity_sha256 != second.identity_sha256


def test_creator_and_container_dates_are_not_calindex_semantics(tmp_path):
    first_cif, _ = _write_fixture(tmp_path / "first")
    changed = [dict(row) for row in ROWS]
    changed[0]["CREATOR"] = "/different/tool/path/deceit.pl"
    changed[0]["DATE"] = "2026-01-01T00:00:00"
    changed[0]["SUBDATE"] = "2026-01-02T00:00:00"
    second_cif, _ = _write_fixture(tmp_path / "second", rows=tuple(changed))

    first_digest, _ = _read_calindex(first_cif)
    second_digest, _ = _read_calindex(second_cif)

    assert first_digest == second_digest


def test_physical_calindex_row_order_is_nonsemantic(tmp_path):
    first_cif, first_ccf = _write_fixture(tmp_path / "first")
    second_cif, second_ccf = _write_fixture(tmp_path / "second", rows=tuple(reversed(ROWS)))

    first = _identity(first_cif, first_ccf)
    second = _identity(second_cif, second_ccf)

    assert first.calindex_sha256 == second.calindex_sha256
    assert first.identity_sha256 == second.identity_sha256


def test_md5_and_fsize_remain_fail_closed_validation_evidence(tmp_path):
    cif, ccf = _write_fixture(tmp_path / "fixture")

    with fits.open(cif, mode="update") as hdus:
        hdus["CALINDEX"].data[0]["MD5"] = "0" * 32
        hdus.flush()
    with pytest.raises(CalibrationIdentityError, match="does not match CALINDEX MD5"):
        _identity(cif, ccf)

    cif, ccf = _write_fixture(tmp_path / "fixture-size", name="size.cif")
    with fits.open(cif, mode="update") as hdus:
        hdus["CALINDEX"].data[0]["FSIZE"] = 999999
        hdus.flush()
    with pytest.raises(CalibrationIdentityError, match="does not match CALINDEX FSIZE"):
        _identity(cif, ccf)


def test_calindex_v3_does_not_treat_extsequid_as_extseqid_alias(tmp_path):
    cif, _ = _write_fixture(tmp_path / "fixture")
    with fits.open(cif, memmap=False) as hdus:
        table = hdus["CALINDEX"]
        extra = fits.Column(
            name="EXTSEQUID",
            format="256A",
            array=["legacy-spelling-a", "legacy-spelling-b"],
        )
        replacement = fits.BinTableHDU.from_columns(table.columns + extra, name="CALINDEX")
        fits.HDUList([fits.PrimaryHDU(), replacement]).writeto(
            tmp_path / "with-legacy-spelling.cif"
        )

    original, _ = _read_calindex(cif)
    legacy, _ = _read_calindex(tmp_path / "with-legacy-spelling.cif")
    assert original == legacy
