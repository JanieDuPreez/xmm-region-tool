from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool.model import CelestialRegion, CelestialSelection
from xmm_region_tool.workflow import (
    WorkflowError,
    discover_all_event_files,
    discover_event_files,
    extraction_cells,
    pn_oot_event_file,
)


def _circle(center_ra: float, *, include: bool = True) -> CelestialRegion:
    selection = CelestialSelection.sector_annulus(
        SkyCoord(center_ra * u.deg, -9.0 * u.deg, frame="icrs"),
        0 * u.arcmin,
        1 * u.arcmin,
        0 * u.deg,
        360 * u.deg,
        include=include,
    )
    return selection.regions[0]


def test_positive_regions_split_into_independent_cells_with_shared_exclusions():
    first = _circle(10.0)
    second = _circle(10.1)
    exclusion = _circle(10.05, include=False)
    selection = CelestialSelection(regions=(first, second, exclusion))

    cells = extraction_cells(selection)

    assert [cell.label for cell in cells] == ["r001", "r002"]
    assert len(cells[0].selection.regions) == 2
    assert cells[0].selection.regions[0] == first
    assert cells[0].selection.regions[1] == exclusion
    assert cells[1].selection.regions[0] == second
    assert cells[1].selection.regions[1] == exclusion


def test_exclusion_only_selection_remains_one_cell():
    exclusion = _circle(10.0, include=False)
    selection = CelestialSelection(regions=(exclusion,))

    cells = extraction_cells(selection)

    assert len(cells) == 1
    assert cells[0].selection.geometry_sha256 == selection.geometry_sha256


def _event_file(
    path: Path,
    *,
    instrume: str,
    exposure: str = "S001",
    marker: str | None = None,
) -> None:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "XMM"
    if marker is not None:
        primary.header["TESTMARK"] = marker
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="X", format="D", array=np.asarray([1.0])),
            fits.Column(name="Y", format="D", array=np.asarray([1.0])),
            fits.Column(name="DETX", format="D", array=np.asarray([1.0])),
            fits.Column(name="DETY", format="D", array=np.asarray([1.0])),
        ],
        name="EVENTS",
    )
    table.header["INSTRUME"] = instrume
    table.header["OBS_ID"] = "001"
    table.header["EXP_ID"] = exposure
    fits.HDUList([primary, table]).writeto(path)


def test_discovery_finds_current_esas_allevc_products(tmp_path):
    _event_file(tmp_path / "mos1S001-allevc.fits", instrume="EMOS1")
    _event_file(tmp_path / "pnS003-allevc.fits", instrume="EPN", exposure="S003")

    found = discover_event_files(tmp_path, ("mos1", "pn"))

    assert [path.name for path in found] == ["mos1S001-allevc.fits", "pnS003-allevc.fits"]


def test_discovery_does_not_treat_old_clean_name_as_current_canonical_product(tmp_path):
    _event_file(tmp_path / "mos1S001-clean.fits", instrume="EMOS1")

    with pytest.raises(WorkflowError, match="no canonical mos1"):
        discover_event_files(tmp_path, ("mos1",))


def test_discovery_refuses_ambiguous_allevc_candidates(tmp_path):
    _event_file(tmp_path / "mos1S001-allevc.fits", instrume="EMOS1", exposure="S001")
    _event_file(tmp_path / "mos1S002-allevc.fits", instrume="EMOS1", exposure="S002")

    with pytest.raises(WorkflowError, match="multiple canonical mos1"):
        discover_event_files(tmp_path, ("mos1",))


def test_discovery_refuses_missing_requested_instrument(tmp_path):
    with pytest.raises(WorkflowError, match="no canonical pn"):
        discover_event_files(tmp_path, ("pn",))


def test_discover_all_uses_available_allevc_cameras(tmp_path):
    _event_file(tmp_path / "mos2S002-allevc.fits", instrume="EMOS2", exposure="S002")
    _event_file(tmp_path / "pnS003-allevc.fits", instrume="EPN", exposure="S003")

    found = discover_all_event_files(tmp_path)

    assert [path.name for path in found] == ["mos2S002-allevc.fits", "pnS003-allevc.fits"]


def test_pn_oot_pair_uses_current_esas_allevcoot_naming(tmp_path):
    pn = tmp_path / "pnS003-allevc.fits"
    oot = tmp_path / "pnS003-allevcoot.fits"
    _event_file(pn, instrume="EPN", exposure="S003")
    _event_file(oot, instrume="EPN", exposure="S003", marker="oot")

    assert pn_oot_event_file(pn) == oot.resolve()


def test_pn_oot_pair_accepts_hyphenated_compatibility_naming(tmp_path):
    pn = tmp_path / "pnS003-allevc.fits"
    oot = tmp_path / "pnS003-allevc-oot.fits"
    _event_file(pn, instrume="EPN", exposure="S003")
    _event_file(oot, instrume="EPN", exposure="S003", marker="oot")

    assert pn_oot_event_file(pn) == oot.resolve()


def test_pn_oot_pair_refuses_ambiguous_siblings(tmp_path):
    pn = tmp_path / "pnS003-allevc.fits"
    _event_file(pn, instrume="EPN", exposure="S003")
    _event_file(
        tmp_path / "pnS003-allevcoot.fits",
        instrume="EPN",
        exposure="S003",
    )
    _event_file(
        tmp_path / "pnS003-allevc-oot.fits",
        instrume="EPN",
        exposure="S003",
    )

    with pytest.raises(WorkflowError, match="multiple matching pn OOT"):
        pn_oot_event_file(pn)


def test_pn_oot_pair_refuses_same_physical_file_symlink(tmp_path):
    pn = tmp_path / "pnS003-allevc.fits"
    oot = tmp_path / "pnS003-allevcoot.fits"
    _event_file(pn, instrume="EPN", exposure="S003")
    oot.symlink_to(pn.name)

    with pytest.raises(WorkflowError, match="same physical file"):
        pn_oot_event_file(pn)


def test_pn_oot_pair_refuses_byte_identical_distinct_file(tmp_path):
    pn = tmp_path / "pnS003-allevc.fits"
    oot = tmp_path / "pnS003-allevcoot.fits"
    _event_file(pn, instrume="EPN", exposure="S003")
    oot.write_bytes(pn.read_bytes())

    with pytest.raises(WorkflowError, match="byte-identical"):
        pn_oot_event_file(pn)


def test_pn_oot_pair_refuses_malformed_named_sibling(tmp_path):
    pn = tmp_path / "pnS003-allevc.fits"
    oot = tmp_path / "pnS003-allevcoot.fits"
    _event_file(pn, instrume="EPN", exposure="S003")
    oot.write_bytes(b"not-fits")

    with pytest.raises(WorkflowError, match="not a readable XMM event product"):
        pn_oot_event_file(pn)


def test_pn_oot_pair_refuses_wrong_exposure(tmp_path):
    pn = tmp_path / "pnS003-allevc.fits"
    oot = tmp_path / "pnS003-allevcoot.fits"
    _event_file(pn, instrume="EPN", exposure="S003")
    _event_file(oot, instrume="EPN", exposure="S004")

    with pytest.raises(WorkflowError, match="does not belong to the science event"):
        pn_oot_event_file(pn)


def test_pn_oot_pair_is_optional_when_not_present(tmp_path):
    pn = tmp_path / "pnS003-allevc.fits"
    _event_file(pn, instrume="EPN", exposure="S003")

    assert pn_oot_event_file(pn) is None
