from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_legacy_synthetic_projection_context(request, monkeypatch):
    """Keep fake-CIF geometry tests scoped to the behaviour they actually test.

    ``test_adaptive_projection`` and the projection-execution tests intentionally
    use plain-text stand-ins for CIF/executable state and already bypass real
    ``SasProjectionContext.validate()`` where projection execution is under test.
    They cannot satisfy the production observation/CIF suitability boundary
    without becoming calibration-fixture tests themselves.  Dedicated
    ``test_context_suitability_wave_c`` coverage exercises that boundary with
    structurally valid FITS CIF + ODF/SOSF fixtures.
    """
    module_name = request.module.__name__
    if module_name not in {
        "tests.test_adaptive_projection",
        "tests.test_execution",
        "test_adaptive_projection",
        "test_execution",
    }:
        return

    monkeypatch.setattr(
        "xmm_region_tool.sas.capture_context_suitability",
        lambda **kwargs: "synthetic-context-suitability-bypass",
    )
