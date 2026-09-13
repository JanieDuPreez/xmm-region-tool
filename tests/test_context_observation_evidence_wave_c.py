from __future__ import annotations

from pathlib import Path

from xmm_region_tool.calibration import CalibrationIdentity, SasProducerIdentity
from xmm_region_tool.context_suitability import ObservationRecord
from xmm_region_tool.execution import SasProjectionContext


def _context(tmp_path: Path, observation_evidence=None) -> SasProjectionContext:
    executable = tmp_path / "esky2det"
    executable.write_text("fake")
    cif = tmp_path / "ccf.cif"
    cif.write_text("fake")
    return SasProjectionContext(
        environment={"PATH": str(tmp_path), "SAS_CCF": str(cif)},
        calibration=CalibrationIdentity(
            cif_path=cif,
            cif_file_sha256="1" * 64,
            calindex_sha256="2" * 64,
            replacements=(),
            ccf_search_path=(),
        ),
        producer=SasProducerIdentity(
            esky2det_version="22",
            sas_version="22",
            esky2det_sha256="3" * 64,
        ),
        esky2det_path=executable,
        observation_evidence=observation_evidence,
    )


def test_declared_observation_evidence_is_nonsemantic_context_evidence(tmp_path):
    declared = ObservationRecord(
        obs_id="0723802001",
        scheduled_start="2013-06-08T14:52:32",
        scheduled_end="2013-06-09T16:34:12",
    )
    without = _context(tmp_path)
    with_evidence = _context(tmp_path, declared)

    assert with_evidence.observation_evidence == declared
    assert without.canonical_record() == with_evidence.canonical_record()
    assert without.identity_sha256 == with_evidence.identity_sha256
    assert "observation" not in str(with_evidence.canonical_record()).lower()
