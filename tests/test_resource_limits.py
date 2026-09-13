"""Public resource limits reject amplification before allocation or execution."""
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from xmm_region_tool import load_ds9_selection
from xmm_region_tool.geometry import read_ds9_sky_regions
from xmm_region_tool.model import CelestialBoundary, CirclePath, DetectorBoundary, DetectorRegion
from xmm_region_tool.output import detector_region_components, write_esas_regionfile, write_fits_region
from xmm_region_tool.limits import MAX_SOURCE_BYTES, MAX_SOURCE_CELLS, bounded_text
from xmm_region_tool.sas_validation import fits_region_membership, SasRegionValidationError


def regions(count=1):
    boundary = DetectorBoundary(np.array([[0., 0.], [1., 0.], [0., 1.]]))
    return [DetectorRegion(True, (boundary,))] * count


@pytest.mark.parametrize("bad", [True, 16.0, float("nan"), float("inf"), "16"])
def test_public_limits_are_integers(tmp_path, bad):
    with pytest.raises(ValueError, match="samples"):
        load_ds9_selection(tmp_path / "missing", samples=bad)
    with pytest.raises(ValueError, match="sampling_hint"):
        CelestialBoundary(CirclePath(SkyCoord(0*u.deg, 0*u.deg), 1*u.deg), sampling_hint=bad)
    with pytest.raises(ValueError, match="max_components"):
        detector_region_components(regions(), max_components=bad)
    with pytest.raises(ValueError, match="inline_limit"):
        write_esas_regionfile(tmp_path / "out", regions(), inline_limit=bad)


@pytest.mark.parametrize("count", ["nan", "inf", "1e309", "1000000000"])
def test_source_subdivision_is_bounded(tmp_path, count):
    path = tmp_path / "source.reg"
    for text in (f'icrs\nannulus(10,-9,1",2",n={count})',
                 f'icrs\npanda(10,-9,0,90,{count},1",2",1)',
                 f'icrs\nepanda(10,-9,0,90,1,1",1",2",2",{count},0)'):
        path.write_text(text)
        with pytest.raises(ValueError, match="integer|resource limit"):
            load_ds9_selection(path)


def test_source_product_and_statement_limits(tmp_path):
    path = tmp_path / "source.reg"
    path.write_text('icrs\npanda(10,-9,0,90,100,1",2",100)')
    with pytest.raises(ValueError, match="resource limit"):
        load_ds9_selection(path)
    path.write_text('icrs\n' + 'circle(10,-9,1")\n' * MAX_SOURCE_CELLS)
    with pytest.raises(ValueError, match="resource limit"):
        load_ds9_selection(path)


def test_source_and_json_bytes_bounded(tmp_path):
    path = tmp_path / "oversized"
    with path.open("wb") as stream:
        stream.seek(MAX_SOURCE_BYTES)
        stream.write(b" ")
    for reader in (load_ds9_selection, read_ds9_sky_regions, bounded_text):
        with pytest.raises(ValueError, match="resource limit"):
            reader(path)


def test_initial_components_and_fits_integer_domain():
    with pytest.raises(ValueError, match="components"):
        detector_region_components(regions(3), max_components=2)
    with pytest.raises(ValueError, match="resource limit"):
        detector_region_components(regions(), max_components=2**31)


def test_fits_skips_expression_and_bounds_rows(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("FITS path built a discarded expression")
    monkeypatch.setattr("xmm_region_tool.output.selection_expression", forbidden)
    write_esas_regionfile(tmp_path / "good.txt", regions(), representation="fits")
    monkeypatch.setattr("xmm_region_tool.output.MAX_REGION_ROWS", 2)
    with pytest.raises(ValueError, match="row resource"):
        write_fits_region(tmp_path / "bad.fits", regions(3))


def test_aggregate_fits_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("xmm_region_tool.output.MAX_REGION_COORDINATES", 10)
    with pytest.raises(ValueError, match="aggregate"):
        write_fits_region(tmp_path / "bad.fits", regions(2))
    assert not (tmp_path / "bad.fits").exists()


def test_auto_expression_stops_at_inline_budget(tmp_path):
    result = write_esas_regionfile(tmp_path / "auto.txt", regions(3), representation="auto", inline_limit=10)
    assert result.representation == "fits"


@pytest.mark.parametrize("row_cap,vertex_cap", [(1, 16384), (16384, 2)])
def test_external_region_budget_precedes_polygon_evaluation(tmp_path, monkeypatch, row_cap, vertex_cap):
    path = write_fits_region(tmp_path / "input.fits", regions(2))
    monkeypatch.setattr("xmm_region_tool.limits.MAX_REGION_ROWS", row_cap)
    monkeypatch.setattr("xmm_region_tool.limits.MAX_REGION_VERTICES", vertex_cap)
    def forbidden(*args, **kwargs):
        pytest.fail("excessive REGION reached polygon evaluation")
    monkeypatch.setattr("xmm_region_tool.sas_validation.PolygonPixelRegion", forbidden)
    with pytest.raises(SasRegionValidationError, match="resource limit"):
        fits_region_membership(path, np.array([0.]), np.array([0.]))


def test_variable_length_region_vectors_rejected_before_access(tmp_path):
    from xmm_region_tool.limits import validate_region_table_budget
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="X", format="PD()", array=np.array([np.arange(4.)], dtype=object)),
        fits.Column(name="Y", format="PD()", array=np.array([np.arange(4.)], dtype=object)),
    ], name="REGION")
    with pytest.raises(ValueError, match="fixed-width"):
        validate_region_table_budget(table)
