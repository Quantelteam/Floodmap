"""
test_dem.py — Unit tests for terrain feature computation in floodmap.dem.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_dem_10x10():
    """
    A 10×10 perfectly flat DEM at 5 m resolution.

    Slope should be ~0° everywhere; TWI will be dominated by flow accumulation.
    """
    return np.zeros((10, 10), dtype=np.float32), 5.0


@pytest.fixture
def ramp_dem_10x10():
    """
    A 10×10 DEM with a uniform south-to-north elevation ramp (0–9 m).

    Slope should be constant; flow accumulates northward.
    """
    dem = np.tile(np.arange(10, dtype=np.float32), (10, 1)).T  # rows increase northward
    return dem, 5.0  # 5 m resolution


@pytest.fixture
def bowl_dem_10x10():
    """
    A 10×10 DEM shaped like a bowl — lowest in the centre.

    The centre cell should have the highest depression depth and TWI.
    """
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    dem = (X ** 2 + Y ** 2).astype(np.float32) * 5.0  # 0 at centre, ~10 at corners
    return dem, 5.0


# ---------------------------------------------------------------------------
# Slope tests
# ---------------------------------------------------------------------------

class TestSlopeComputation:
    """Validate slope computation on simple synthetic DEMs."""

    def test_flat_dem_slope_near_zero(self, flat_dem_10x10):
        """Slope of a flat DEM should be approximately 0° everywhere."""
        from floodmap.dem import _compute_slope_curvature

        dem, res = flat_dem_10x10
        slope, _ = _compute_slope_curvature(dem, res)
        # Allow small numerical noise
        assert np.nanmax(slope) < 0.01, f"Flat DEM max slope = {np.nanmax(slope)}"

    def test_ramp_slope_positive(self, ramp_dem_10x10):
        """Slope of a ramp DEM should be positive and roughly uniform."""
        from floodmap.dem import _compute_slope_curvature

        dem, res = ramp_dem_10x10
        slope, _ = _compute_slope_curvature(dem, res)
        interior = slope[1:-1, 1:-1]
        assert np.all(interior > 0), "Ramp DEM should produce positive slope everywhere"
        # For 1 m rise over 5 m run: arctan(0.2) ≈ 11.3°
        expected = np.degrees(np.arctan(1.0 / res))
        assert np.allclose(interior, expected, atol=1.0), (
            f"Ramp slope expected ~{expected:.1f}°, got {interior.mean():.1f}°"
        )


# ---------------------------------------------------------------------------
# Depression depth tests
# ---------------------------------------------------------------------------

class TestDepressionDepth:
    """Validate depression depth computation."""

    def test_flat_dem_no_depressions(self, flat_dem_10x10):
        """A flat DEM should have zero depression depth."""
        from floodmap.dem import _fill_pits

        dem, _ = flat_dem_10x10
        filled = _fill_pits(dem)
        depth = np.maximum(0.0, filled - dem)
        assert np.allclose(depth, 0, atol=1e-5), "Flat DEM should have no depressions"

    def test_pit_filled(self):
        """A single-cell pit should be filled to neighbour level."""
        from floodmap.dem import _fill_pits

        dem = np.ones((5, 5), dtype=np.float32) * 10.0
        dem[2, 2] = 0.0  # single pit in the centre

        filled = _fill_pits(dem)
        depth = filled[2, 2] - dem[2, 2]
        assert depth > 0, "Pit should be filled (depth > 0)"
        assert filled[2, 2] <= 10.0, "Filled pit should not exceed neighbour elevation"

    def test_bowl_depression_max_at_centre(self, bowl_dem_10x10):
        """Bowl DEM: depression depth should peak in the interior."""
        from floodmap.dem import _fill_pits

        dem, _ = bowl_dem_10x10
        filled = _fill_pits(dem)
        depth = np.maximum(0.0, filled - dem)
        # Interior cells should have more depression than corners
        interior_max = depth[2:8, 2:8].max()
        corner_max = max(depth[0, 0], depth[0, -1], depth[-1, 0], depth[-1, -1])
        assert interior_max >= corner_max, (
            "Bowl depression should be deepest in the interior"
        )


# ---------------------------------------------------------------------------
# Flow accumulation tests
# ---------------------------------------------------------------------------

class TestFlowAccumulation:
    """Validate D8 flow accumulation fallback."""

    def test_minimum_one_cell(self, ramp_dem_10x10):
        """Every cell should contribute at least 1 to flow accumulation."""
        from floodmap.dem import _simple_d8_accum

        dem, _ = ramp_dem_10x10
        accum = _simple_d8_accum(dem)
        assert np.all(accum >= 1.0), "All cells should have flow_accum >= 1"

    def test_downstream_larger(self, ramp_dem_10x10):
        """Downstream cells should have higher accumulation than upstream."""
        from floodmap.dem import _simple_d8_accum

        dem, _ = ramp_dem_10x10
        accum = _simple_d8_accum(dem)
        # On a N-S ramp, row 0 (lowest elevation) collects flow from all uphill rows
        # At least the bottom row should accumulate more than the top row on average
        bottom_row_mean = accum[0, :].mean()
        top_row_mean = accum[-1, :].mean()
        assert bottom_row_mean >= top_row_mean, (
            "Bottom of ramp should have higher flow accumulation"
        )


# ---------------------------------------------------------------------------
# TWI tests
# ---------------------------------------------------------------------------

class TestTWI:
    """Validate TWI computation."""

    def test_twi_finite(self, ramp_dem_10x10):
        """TWI should be finite for a ramp DEM (no infinities)."""
        from floodmap.dem import _compute_flow_accum_twi, _compute_slope_curvature

        dem, res = ramp_dem_10x10
        slope, _ = _compute_slope_curvature(dem, res)
        accum, twi = _compute_flow_accum_twi(dem, slope, res)
        # Clip to interior to avoid boundary effects
        interior_twi = twi[1:-1, 1:-1]
        assert np.all(np.isfinite(interior_twi)), "TWI must be finite in the interior"

    def test_twi_low_on_ridge(self, ramp_dem_10x10):
        """Upslope cells (top of ramp) should have lower TWI than downslope."""
        from floodmap.dem import _compute_flow_accum_twi, _compute_slope_curvature

        dem, res = ramp_dem_10x10
        slope, _ = _compute_slope_curvature(dem, res)
        accum, twi = _compute_flow_accum_twi(dem, slope, res)
        # Bottom rows collect more water → higher TWI
        bottom_twi = float(np.nanmean(twi[0:2, :]))
        top_twi = float(np.nanmean(twi[-2:, :]))
        assert bottom_twi >= top_twi, (
            f"Bottom TWI ({bottom_twi:.2f}) should be >= top TWI ({top_twi:.2f})"
        )


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------

class TestAggregation:
    """Validate fine-to-coarse aggregation."""

    def test_aggregate_shape(self):
        """Output shape should be (n_bands, fine_rows/factor, fine_cols/factor)."""
        from floodmap.dem import _aggregate_to_coarse
        import rasterio.transform

        n_bands, nrows, ncols = 3, 20, 20
        rng = np.random.default_rng(42)
        bands = rng.random((n_bands, nrows, ncols)).astype(np.float32)
        transform = rasterio.transform.from_origin(0, 100, 5, 5)

        result = _aggregate_to_coarse(bands, transform, "EPSG:32618", 5.0, 10.0)
        assert result.shape == (n_bands, nrows // 2, ncols // 2)

    def test_aggregate_mean(self):
        """Aggregation should compute the mean of the fine-resolution values."""
        from floodmap.dem import _aggregate_to_coarse
        import rasterio.transform

        # 1 band, 4×4, all values = 2.0 → coarse should also be 2.0
        bands = np.full((1, 4, 4), 2.0, dtype=np.float32)
        transform = rasterio.transform.from_origin(0, 20, 5, 5)
        result = _aggregate_to_coarse(bands, transform, "EPSG:32618", 5.0, 10.0)
        assert result.shape == (1, 2, 2)
        assert np.allclose(result, 2.0), "Uniform values should aggregate to same value"


# ---------------------------------------------------------------------------
# Tile mosaic: units, nodata, alignment
# ---------------------------------------------------------------------------

_TEST_BBOX = (40.650, -74.000, 40.655, -73.995)
_F32_NODATA = -3.4028230607370965e38  # NOAA value; not exactly representable in float32


def _write_tile(path, value_fn, nodata=_F32_NODATA):
    """Write a 10-ft-resolution EPSG:6539 tile (US survey feet) covering _TEST_BBOX."""
    import pyproj
    import rasterio
    from rasterio.transform import from_origin

    lat0, lon0, lat1, lon1 = _TEST_BBOX
    t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6539", always_xy=True)
    x0, y0 = t.transform(lon0 - 0.003, lat0 - 0.003)
    x1, y1 = t.transform(lon1 + 0.003, lat1 + 0.003)
    res = 10.0
    ncols, nrows = int((x1 - x0) / res), int((y1 - y0) / res)
    xs = x0 + (np.arange(ncols) + 0.5) * res
    arr = value_fn(xs, nrows, ncols).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=nrows, width=ncols, count=1, dtype="float32",
        crs="EPSG:6539", transform=from_origin(x0, y1, res, res), nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    return path


class TestMosaic:
    def test_feet_converted_to_metres_and_not_zero(self, tmp_path):
        import rasterio
        from floodmap.dem import merge_and_reproject_tiles

        tile = _write_tile(tmp_path / "a.tif", lambda xs, nr, nc: np.full((nr, nc), 100.0))
        out = merge_and_reproject_tiles([tile], _TEST_BBOX, tmp_path / "dem.tif", res_m=5, force=True)
        with rasterio.open(out) as src:
            dem = src.read(1)
        assert np.isfinite(dem).all()
        assert np.allclose(dem, 100 * 0.3048006096, atol=0.01)  # US survey foot -> m

    def test_nodata_becomes_nan_and_second_tile_fills_gap(self, tmp_path):
        import rasterio
        from floodmap.dem import merge_and_reproject_tiles

        def left_half_nodata(xs, nr, nc):
            a = np.full((nr, nc), 50.0)
            a[:, : nc // 2] = _F32_NODATA
            return a

        t1 = _write_tile(tmp_path / "a.tif", left_half_nodata)
        t2 = _write_tile(tmp_path / "b.tif", lambda xs, nr, nc: np.full((nr, nc), 200.0))
        only_first = merge_and_reproject_tiles([t1], _TEST_BBOX, tmp_path / "one.tif", res_m=5, force=True)
        both = merge_and_reproject_tiles([t1, t2], _TEST_BBOX, tmp_path / "two.tif", res_m=5, force=True)
        with rasterio.open(only_first) as s1, rasterio.open(both) as s2:
            d1, d2 = s1.read(1), s2.read(1)
        assert np.isnan(d1).any() and np.nanmax(d1) < 20  # no -3.4e38 leaking through
        assert np.isfinite(d2).all()
        assert np.nanmax(d2) == pytest.approx(200 * 0.3048006096, abs=0.05)  # gap filled by tile 2

    def test_output_aligned_to_100m_grid(self, tmp_path):
        import rasterio
        from floodmap.config import GRID_RES_M
        from floodmap.dem import merge_and_reproject_tiles

        tile = _write_tile(tmp_path / "a.tif", lambda xs, nr, nc: np.full((nr, nc), 10.0))
        out = merge_and_reproject_tiles([tile], _TEST_BBOX, tmp_path / "dem.tif", res_m=5, force=True)
        with rasterio.open(out) as src:
            t = src.transform
            assert t.c % GRID_RES_M == pytest.approx(0, abs=1e-6)
            assert t.f % GRID_RES_M == pytest.approx(0, abs=1e-6)
            assert src.width % (GRID_RES_M // 5) == 0 and src.height % (GRID_RES_M // 5) == 0

    def test_terrain_matches_prediction_grid(self, tmp_path, monkeypatch):
        """terrain_100m.tif must have exactly the cells of grid.build_grid."""
        import rasterio
        from floodmap.dem import compute_terrain_features, merge_and_reproject_tiles
        from floodmap.grid import build_grid

        tile = _write_tile(
            tmp_path / "a.tif",
            lambda xs, nr, nc: np.tile(np.linspace(0, 40, nc), (nr, 1)),
        )
        dem = merge_and_reproject_tiles([tile], _TEST_BBOX, tmp_path / "dem.tif", res_m=5, force=True)
        terrain = compute_terrain_features(dem, tmp_path / "t.tif", force=True)
        import floodmap.grid as grid_mod

        # keep the real data/raw/grid.* cache untouched
        monkeypatch.setattr(grid_mod, "_GRID_PARQUET_PATH", tmp_path / "grid.parquet")
        monkeypatch.setattr(grid_mod, "_GRID_GPKG_PATH", tmp_path / "grid.gpkg")
        grid = build_grid(_TEST_BBOX, force=True)
        with rasterio.open(terrain) as src:
            assert src.width * src.height == len(grid)

    def test_no_tiles_raises_without_synthetic_opt_in(self, tmp_path, monkeypatch):
        import floodmap.dem as dem_mod

        monkeypatch.setattr(dem_mod, "ALLOW_SYNTHETIC", False)
        with pytest.raises(RuntimeError):
            dem_mod.merge_and_reproject_tiles([], _TEST_BBOX, tmp_path / "x.tif", force=True)

