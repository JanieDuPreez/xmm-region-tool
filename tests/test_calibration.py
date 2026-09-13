from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from xmm_region_tool.calibration import (
    CalibrationIdentityError,
    read_calibration_identity,
    read_sas_producer_identity,
)


def md5_bytes(data: bytes) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    digest.update(data)
    return digest.hexdigest()


def write_cif(
    path: Path,
    *,
    fname: str = "EMOS1_BORESIGHT_0010.CCF",
    constituent_bytes: bytes = b"boresight-calibration",
    recorded_md5: str | None = None,
    recorded_size: int | None = None,
) -> Path:
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TELESCOP", format="4A", array=["XMM"]),
            fits.Column(name="SCOPE", format="6A", array=["EMOS1"]),
            fits.Column(name="TYPEID", format="32A", array=["BORESIGHT"]),
            fits.Column(name="ISSUE", format="J", array=np.asarray([10], dtype=np.int32)),
            fits.Column(name="VALDATE", format="19A", array=["2000-01-01T00:00:00"]),
            fits.Column(name="FNAME", format="256A", array=[fname]),
            fits.Column(
                name="FSIZE",
                format="J",
                array=np.asarray(
                    [len(constituent_bytes) if recorded_size is None else recorded_size],
                    dtype=np.int32,
                ),
            ),
            fits.Column(
                name="MD5",
                format="32A",
                array=[recorded_md5 or md5_bytes(constituent_bytes)],
            ),
        ],
        name="CALINDEX",
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)
    return path


def write_constituent(directory: Path, data: bytes = b"boresight-calibration") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "EMOS1_BORESIGHT_0010.CCF"
    path.write_bytes(data)
    return path


def test_calibration_identity_normalizes_constituent_mount_path(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_constituent = write_constituent(first_dir)
    second_constituent = write_constituent(second_dir)
    first = write_cif(tmp_path / "first.cif", fname=str(first_constituent))
    second = write_cif(tmp_path / "second.cif", fname=str(second_constituent))

    first_identity = read_calibration_identity({"SAS_CCF": str(first)})
    second_identity = read_calibration_identity({"SAS_CCF": str(second)})

    assert first_identity.calindex_sha256 == second_identity.calindex_sha256
    assert first_identity.identity_sha256 == second_identity.identity_sha256
    assert first_identity.cif_file_sha256 != second_identity.cif_file_sha256
    assert len(first_identity.constituents) == 1
    assert first_identity.constituents[0].name == "EMOS1_BORESIGHT_0010.CCF"


def test_constituent_byte_change_changes_calibration_identity(tmp_path):
    ccf_dir = tmp_path / "ccf"
    constituent = write_constituent(ccf_dir, b"version-one")
    cif = write_cif(tmp_path / "ccf.cif", constituent_bytes=b"version-one")
    environment = {"SAS_CCF": str(cif), "SAS_CCFPATH": str(ccf_dir)}

    first = read_calibration_identity(environment)

    constituent.write_bytes(b"version-two")
    write_cif(cif, constituent_bytes=b"version-two")
    second = read_calibration_identity(environment)

    assert first.constituents[0].sha256 != second.constituents[0].sha256
    assert first.identity_sha256 != second.identity_sha256


def test_cif_md5_mismatch_fails_closed(tmp_path):
    ccf_dir = tmp_path / "ccf"
    write_constituent(ccf_dir)
    cif = write_cif(tmp_path / "ccf.cif", recorded_md5="0" * 32)

    with pytest.raises(CalibrationIdentityError, match="does not match CALINDEX MD5"):
        read_calibration_identity({"SAS_CCF": str(cif), "SAS_CCFPATH": str(ccf_dir)})


def test_cif_size_mismatch_fails_closed(tmp_path):
    ccf_dir = tmp_path / "ccf"
    write_constituent(ccf_dir)
    cif = write_cif(tmp_path / "ccf.cif", recorded_size=9999)

    with pytest.raises(CalibrationIdentityError, match="does not match CALINDEX FSIZE"):
        read_calibration_identity({"SAS_CCF": str(cif), "SAS_CCFPATH": str(ccf_dir)})


def test_missing_cif_constituent_fails_closed(tmp_path):
    cif = write_cif(tmp_path / "ccf.cif")

    with pytest.raises(CalibrationIdentityError, match="cannot be resolved"):
        read_calibration_identity({"SAS_CCF": str(cif), "SAS_CCFPATH": str(tmp_path / "missing")})


def test_sas_ccf_directory_resolves_default_ccf_cif(tmp_path):
    directory = tmp_path / "calibration"
    directory.mkdir()
    ccf_dir = tmp_path / "ccf"
    write_constituent(ccf_dir)
    cif = write_cif(directory / "ccf.cif")

    identity = read_calibration_identity(
        {"SAS_CCF": str(directory), "SAS_CCFPATH": str(ccf_dir)}
    )

    assert identity.cif_path == cif.resolve()


def test_explicit_replacement_changes_calibration_identity(tmp_path):
    ccf_dir = tmp_path / "ccf"
    write_constituent(ccf_dir)
    cif = write_cif(tmp_path / "ccf.cif")
    replacement = ccf_dir / "EMOS1_BORESIGHT_0011.CCF"
    replacement.write_bytes(b"replacement-one")
    environment = {
        "SAS_CCF": str(cif),
        "SAS_CCFPATH": str(ccf_dir),
        "SAS_CCFFILES": str(replacement),
    }

    first = read_calibration_identity(environment)
    replacement.write_bytes(b"replacement-two")
    second = read_calibration_identity(environment)

    assert len(first.replacements) == 1
    assert first.replacements[0].name == replacement.name
    assert first.replacements[0].sha256 != second.replacements[0].sha256
    assert first.identity_sha256 != second.identity_sha256


def test_missing_sas_ccf_fails_before_conversion():
    with pytest.raises(CalibrationIdentityError, match="SAS_CCF is not set"):
        read_calibration_identity({})


def test_invalid_cif_without_calindex_is_rejected(tmp_path):
    path = tmp_path / "not-a-cif.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)

    with pytest.raises(CalibrationIdentityError, match="no CALINDEX"):
        read_calibration_identity({"SAS_CCF": str(path)})


def test_sas_producer_identity_ignores_runtime_timestamps_and_environment(
    monkeypatch, tmp_path
):
    esky2det = tmp_path / "esky2det"
    sasversion = tmp_path / "sasversion"
    esky2det.write_bytes(b"stable esky2det executable")
    sasversion.write_bytes(b"stable sasversion executable")

    monkeypatch.setattr(
        "xmm_region_tool.calibration.resolve_executable",
        lambda name, environment=None: {"esky2det": esky2det, "sasversion": sasversion}[name],
    )

    run = {"index": 0}

    def fake_run(command, **kwargs):
        run["index"] += 1
        suffix = f"2026-09-08T12:{run['index']:02d}:00.000"
        name = Path(command[0]).name
        if name == "esky2det":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "esky2det:- Executing (routine): esky2det -v\n"
                    "esky2det:- esky2det (esky2det-1.20) "
                    "[22.1.0-a8f2c2afa-20250304] started: " + suffix + "\n"
                    "esky2det:- esky2det (esky2det-1.20) "
                    "[22.1.0-a8f2c2afa-20250304] ended: " + suffix + "\n"
                ),
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "sasversion:- sasversion (sasversion-1.3) "
                "[22.1.0-a8f2c2afa-20250304] started: " + suffix + "\n"
                "sasversion:- XMM-Newton SAS release and build information:\n"
                "SAS release: 22.1.0-a8f2c2afa-20250304\n"
                "SAS-related environment variables that are set:\n"
                f"SAS_CCF = /tmp/run-{run['index']}/ccf.cif\n"
                "sasversion:- sasversion (sasversion-1.3) "
                "[22.1.0-a8f2c2afa-20250304] ended: " + suffix + "\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("xmm_region_tool.calibration.subprocess.run", fake_run)

    first = read_sas_producer_identity({"PATH": str(tmp_path)})
    second = read_sas_producer_identity({"PATH": str(tmp_path)})

    assert first.esky2det_version == "esky2det-1.20 [22.1.0-a8f2c2afa-20250304]"
    assert first.sas_version == "22.1.0-a8f2c2afa-20250304"
    assert first.identity_sha256 == second.identity_sha256


def test_sas_producer_identity_changes_with_release(monkeypatch, tmp_path):
    esky2det = tmp_path / "esky2det"
    sasversion = tmp_path / "sasversion"
    esky2det.write_bytes(b"stable esky2det executable")
    sasversion.write_bytes(b"stable sasversion executable")
    monkeypatch.setattr(
        "xmm_region_tool.calibration.resolve_executable",
        lambda name, environment=None: {"esky2det": esky2det, "sasversion": sasversion}[name],
    )

    release = {"value": "22.1.0-a8f2c2afa-20250304"}

    def fake_run(command, **kwargs):
        name = Path(command[0]).name
        if name == "esky2det":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "esky2det:- esky2det (esky2det-1.20) "
                    "[22.1.0-a8f2c2afa-20250304] started: now\n"
                ),
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=f"SAS release: {release['value']}\n",
            stderr="",
        )

    monkeypatch.setattr("xmm_region_tool.calibration.subprocess.run", fake_run)

    first = read_sas_producer_identity({"PATH": str(tmp_path)})
    release["value"] = "22.2.0-other-build"
    second = read_sas_producer_identity({"PATH": str(tmp_path)})

    assert first.identity_sha256 != second.identity_sha256


def test_nonsemantic_cif_byte_change_preserves_calibration_identity(tmp_path):
    ccf_dir = tmp_path / "ccf"
    write_constituent(ccf_dir)

    first_cif = write_cif(tmp_path / "first.cif")
    second_cif = write_cif(tmp_path / "second.cif")

    # Alter exact CIF bytes without changing CALINDEX semantics or any selected
    # CCF constituent. Exact CIF continuity is execution evidence, not semantic
    # calibration identity material.
    with fits.open(second_cif, mode="update") as hdus:
        hdus[0].header["XRGTEST"] = "different-bytes"
        hdus.flush()

    first = read_calibration_identity(
        {"SAS_CCF": str(first_cif), "SAS_CCFPATH": str(ccf_dir)}
    )
    second = read_calibration_identity(
        {"SAS_CCF": str(second_cif), "SAS_CCFPATH": str(ccf_dir)}
    )

    assert first.cif_file_sha256 != second.cif_file_sha256
    assert first.calindex_sha256 == second.calindex_sha256
    assert first.constituents == second.constituents
    assert first.replacements == second.replacements
    assert first.identity_sha256 == second.identity_sha256
