from __future__ import annotations

import hashlib
import sys

import pytest

from xmm_region_tool.sas_task_producer import (
    SasTaskProducerError,
    capture_sas_task_producer,
    validate_sas_task_producer_capture,
)


def _write_task(path, *, version: str, noisy: bool = False):
    noise = "print('x' * 70000)\n" if noisy else ""
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"{noise}"
        f"print('evselect ({version}) [xmmsas-test-build]')\n"
    )
    path.chmod(0o755)
    return path


def test_capture_sas_task_producer_binds_exact_bytes_and_version(tmp_path):
    executable = _write_task(tmp_path / "evselect", version="evselect-1.0")
    environment = {"PATH": str(tmp_path)}

    capture = capture_sas_task_producer("evselect", environment=environment)

    expected_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    assert capture.executable_path == executable.resolve()
    assert capture.identity.task_name == "evselect"
    assert capture.identity.task_version == "evselect-1.0 [xmmsas-test-build]"
    assert capture.identity.executable_sha256 == expected_sha
    assert capture.identity.sas_version is None
    assert capture.identity.canonical_record()["schema"] == "xmm-region-tool.sas-task-producer/v1"
    assert len(capture.identity.identity_sha256) == 64
    validate_sas_task_producer_capture(capture)


def test_replacing_task_bytes_changes_identity_and_invalidates_old_capture(tmp_path):
    executable = _write_task(tmp_path / "evselect", version="evselect-1.0")
    environment = {"PATH": str(tmp_path)}
    first = capture_sas_task_producer("evselect", environment=environment)

    _write_task(executable, version="evselect-2.0")
    second = capture_sas_task_producer("evselect", environment=environment)

    assert first.identity.identity_sha256 != second.identity.identity_sha256
    assert first.identity.executable_sha256 != second.identity.executable_sha256
    with pytest.raises(SasTaskProducerError, match="executable bytes changed"):
        validate_sas_task_producer_capture(first)


def test_noisy_task_version_probe_fails_closed_under_bounded_capture(tmp_path):
    _write_task(tmp_path / "evselect", version="evselect-1.0", noisy=True)
    environment = {"PATH": str(tmp_path)}

    with pytest.raises(SasTaskProducerError, match="truncated diagnostic output"):
        capture_sas_task_producer("evselect", environment=environment)


def test_malformed_task_version_probe_fails_explicitly(tmp_path):
    executable = tmp_path / "evselect"
    executable.write_text(
        f"#!{sys.executable}\n"
        "print('not a SAS task version banner')\n"
    )
    executable.chmod(0o755)

    with pytest.raises(SasTaskProducerError, match="stable evselect version/build"):
        capture_sas_task_producer("evselect", environment={"PATH": str(tmp_path)})
