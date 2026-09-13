from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.calibration import CalibrationIdentityError, read_calibration_identity


NAME = "XMM_BORESIGHT_0035.CCF"
BYTES = b"real-shaped-boresight"


def _md5(data: bytes) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    digest.update(data)
    return digest.hexdigest()


def _write_cif(path: Path, *, fname: str = NAME, data: bytes = BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TELESCOP", format="4A", array=["XMM"]),
            fits.Column(name="SCOPE", format="3A", array=["XMM"]),
            fits.Column(name="TYPEID", format="32A", array=["BORESIGHT"]),
            fits.Column(name="ISSUE", format="J", array=np.asarray([35], dtype=np.int32)),
            fits.Column(name="FNAME", format="256A", array=[fname]),
            fits.Column(name="FSIZE", format="J", array=np.asarray([len(data)], dtype=np.int32)),
            fits.Column(name="MD5", format="32A", array=[_md5(data)]),
        ],
        name="CALINDEX",
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)
    return path


def _write_ccf(path: Path, data: bytes = BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_ccfpath_order_wins_and_cwd_shadow_is_ignored(monkeypatch, tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")
    first = _write_ccf(tmp_path / "first" / NAME)
    second = _write_ccf(tmp_path / "second" / NAME)
    shadow = _write_ccf(tmp_path / "cwd" / NAME, b"shadow-bytes")
    monkeypatch.chdir(shadow.parent)

    first_identity = read_calibration_identity(
        {"SAS_CCF": str(cif), "SAS_CCFPATH": f"{first.parent}:{second.parent}"}
    )
    second_identity = read_calibration_identity(
        {"SAS_CCF": str(cif), "SAS_CCFPATH": f"{second.parent}:{first.parent}"}
    )

    expected = hashlib.sha256(BYTES).hexdigest()
    assert first_identity.constituents[0].sha256 == expected
    assert second_identity.constituents[0].sha256 == expected
    assert hashlib.sha256(shadow.read_bytes()).hexdigest() != expected


def test_directory_valued_sas_ccf_supplies_constituent_root(tmp_path):
    store = tmp_path / "store"
    _write_ccf(store / NAME)
    cif = _write_cif(store / "ccf.cif")

    identity = read_calibration_identity({"SAS_CCF": str(store)})

    assert identity.cif_path == cif.resolve()
    assert identity.ccf_search_path == (store.resolve(),)
    assert identity.constituents[0].name == NAME


def test_relative_subpath_is_preserved_but_parent_traversal_fails(tmp_path):
    root = tmp_path / "root"
    _write_ccf(root / "subdir" / NAME)
    subpath_cif = _write_cif(tmp_path / "subpath.cif", fname=f"subdir/{NAME}")

    identity = read_calibration_identity(
        {"SAS_CCF": str(subpath_cif), "SAS_CCFPATH": str(root)}
    )
    assert identity.constituents[0].name == NAME

    traversal_cif = _write_cif(tmp_path / "traversal.cif", fname=f"../outside/{NAME}")
    with pytest.raises(CalibrationIdentityError, match="parent traversal"):
        read_calibration_identity(
            {"SAS_CCF": str(traversal_cif), "SAS_CCFPATH": str(root)}
        )


def test_relative_subpath_symlink_escape_is_rejected_before_hashing(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    target = _write_ccf(outside / NAME)
    link = root / "subdir" / NAME
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    cif = _write_cif(tmp_path / "symlink-escape.cif", fname=f"subdir/{NAME}")

    with pytest.raises(CalibrationIdentityError, match="resolves outside SAS_CCFPATH root"):
        read_calibration_identity(
            {"SAS_CCF": str(cif), "SAS_CCFPATH": str(root)}
        )


def test_cif_fname_tilde_is_not_expanded_to_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    _write_ccf(home / NAME)
    cif = _write_cif(tmp_path / "tilde.cif", fname=f"~/{NAME}")
    monkeypatch.setenv("HOME", str(home))

    with pytest.raises(CalibrationIdentityError, match="cannot be resolved through SAS_CCFPATH"):
        read_calibration_identity(
            {"SAS_CCF": str(cif), "SAS_CCFPATH": str(root)}
        )


@pytest.mark.parametrize(
    "separator",
    [" ", ",", ":", ",:", ",,"],
)
def test_sas_ccffiles_observed_separators_preserve_order(monkeypatch, tmp_path, separator):
    original_a = "XMM_BORESIGHT_0035.CCF"
    original_b = "EMOS1_LINCOORD_0019.CCF"
    data_a = b"replacement-a"
    data_b = b"replacement-b"

    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="SCOPE", format="6A", array=["XMM", "EMOS1"]),
            fits.Column(name="TYPEID", format="32A", array=["BORESIGHT", "LINCOORD"]),
            fits.Column(name="ISSUE", format="J", array=np.asarray([35, 19], dtype=np.int32)),
            fits.Column(name="FNAME", format="256A", array=[original_a, original_b]),
            fits.Column(name="FSIZE", format="J", array=np.asarray([1, 1], dtype=np.int32)),
            fits.Column(name="MD5", format="32A", array=["0" * 32, "0" * 32]),
        ],
        name="CALINDEX",
    )
    cif = tmp_path / "ccf.cif"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(cif)

    cwd = tmp_path / "cwd"
    rep_a = _write_ccf(cwd / "relative" / original_a, data_a)
    rep_b = _write_ccf(cwd / "relative" / original_b, data_b)
    monkeypatch.chdir(cwd)
    raw = separator.join([f"relative/{original_a}", f"relative/{original_b}"])

    identity = read_calibration_identity(
        {
            "SAS_CCF": str(cif),
            "SAS_CCFPATH": str(tmp_path / "missing-originals"),
            "SAS_CCFFILES": raw,
        }
    )

    assert [item.name for item in identity.replacements] == [rep_a.name, rep_b.name]
    assert [item.sha256 for item in identity.replacements] == [
        hashlib.sha256(data_a).hexdigest(),
        hashlib.sha256(data_b).hexdigest(),
    ]
    assert identity.constituents == ()


def test_quoted_sas_ccffiles_path_fails_closed(tmp_path):
    cif = _write_cif(tmp_path / "ccf.cif")
    with pytest.raises(CalibrationIdentityError, match="quoting is unsupported"):
        read_calibration_identity(
            {
                "SAS_CCF": str(cif),
                "SAS_CCFPATH": str(tmp_path),
                "SAS_CCFFILES": f'"{tmp_path / "replacement path" / NAME}"',
            }
        )
