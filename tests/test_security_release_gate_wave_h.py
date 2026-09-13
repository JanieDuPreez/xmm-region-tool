from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from types import SimpleNamespace

from xmm_region_tool.execution import SasProjectionContext


def test_projection_evidence_environment_whitelist_excludes_ambient_sentinel(tmp_path):
    executable = tmp_path / "esky2det"
    executable.write_bytes(b"synthetic executable")
    environment = {
        "PATH": str(tmp_path),
        "SAS_CCF": str(tmp_path / "ccf.cif"),
        "SAS_CCFPATH": str(tmp_path / "ccf"),
        "SAS_CCFFILES": "EMOS1_BADPIX_0001.CCF",
        "XMM_REGION_SECRET_SENTINEL": "do-not-persist-this-value",
    }
    context = SasProjectionContext(
        environment=environment,
        calibration=SimpleNamespace(identity_sha256="a" * 64),
        producer=SimpleNamespace(identity_sha256="b" * 64),
        esky2det_path=executable,
    )

    record = context.relevant_environment_record()
    encoded = json.dumps(record, sort_keys=True)

    assert record == {
        "SAS_CCF": environment["SAS_CCF"],
        "SAS_CCFPATH": environment["SAS_CCFPATH"],
        "SAS_CCFFILES": environment["SAS_CCFFILES"],
    }
    assert "XMM_REGION_SECRET_SENTINEL" not in encoded
    assert environment["XMM_REGION_SECRET_SENTINEL"] not in encoded


def test_gitignore_covers_actual_atomic_publication_siblings():
    root = Path(__file__).resolve().parents[1]
    patterns = [
        line.strip()
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    stage_name = ".region.xmm-region-stage-a1b2c3.fits"
    backup_name = ".region.fits.xmm-region-backup-a1b2c3"
    ordinary_fixture = "region-reference.fits"

    assert any(fnmatch.fnmatch(stage_name, pattern) for pattern in patterns)
    assert any(fnmatch.fnmatch(backup_name, pattern) for pattern in patterns)
    assert not any(fnmatch.fnmatch(ordinary_fixture, pattern) for pattern in patterns)
