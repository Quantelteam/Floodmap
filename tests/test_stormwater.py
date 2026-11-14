"""
test_stormwater.py — Flood-zone assignment, comparison metrics and geodatabase extraction.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from floodmap.config import CRS_NYC


def _poly_gdf(lon0, lat0, lon1, lat1):
    """GeoDataFrame with one rectangle (given in WGS-84), stored in EPSG:2263."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"Flooding_Category": [1]}, geometry=[box(lon0, lat0, lon1, lat1)], crs="EPSG:4326"
    ).to_crs(CRS_NYC)


@pytest.fixture
def grid():
    return pd.DataFrame(
        {
            "cell_id": [10, 11, 12],
            "lat": [40.650, 40.660, 40.670],
            "lon": [-74.00, -74.00, -74.00],
        }
    )


class TestAssignFloodZones:
    def test_points_inside_polygon_flagged(self, grid):
        from floodmap.stormwater import assign_flood_zones

        moderate = _poly_gdf(-74.01, 40.645, -73.99, 40.655)  # contains cell 10 only
        extreme = _poly_gdf(-74.01, 40.645, -73.99, 40.665)  # cells 10 and 11
        out = assign_flood_zones(grid, moderate, extreme).set_index("cell_id")
        assert out["in_moderate_flood_zone"].to_dict() == {10: True, 11: False, 12: False}
        assert out["in_extreme_flood_zone"].to_dict() == {10: True, 11: True, 12: False}

    def test_none_or_empty_layer_gives_all_false(self, grid):
        import geopandas as gpd
        from floodmap.stormwater import assign_flood_zones

        out = assign_flood_zones(grid, None, gpd.GeoDataFrame(geometry=[], crs=CRS_NYC))
        assert not out["in_moderate_flood_zone"].any()
        assert not out["in_extreme_flood_zone"].any()
        assert list(out["cell_id"]) == [10, 11, 12]

    def test_layer_in_other_crs_is_reprojected(self, grid):
        from floodmap.stormwater import assign_flood_zones

        wgs = _poly_gdf(-74.01, 40.645, -73.99, 40.655).to_crs("EPSG:4326")
        out = assign_flood_zones(grid, wgs, None)
        assert out["in_moderate_flood_zone"].sum() == 1


class TestComparison:
    def test_confusion_matrix_metrics(self):
        from floodmap.stormwater import compare_flood_map_vs_labels

        feats = pd.DataFrame(
            {
                "flooded": [1, 1, 0, 0, 0, 0, 1, 0],
                "in_moderate_flood_zone": [True, False, True, False, False, False, True, False],
                "in_extreme_flood_zone": [True, True, True, True, False, False, True, False],
            }
        )
        res = compare_flood_map_vs_labels(feats).set_index("zone")
        m = res.loc["moderate"]
        assert (m.TP, m.FP, m.FN, m.TN) == (2, 1, 1, 4)
        assert m.precision == pytest.approx(2 / 3, abs=1e-3)
        assert m.recall == pytest.approx(2 / 3, abs=1e-3)
        assert res.loc["extreme"].recall == pytest.approx(1.0)


class TestExtraction:
    def _zip(self, tmp_path):
        zp = tmp_path / "sw.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Root/A_Moderate_x.gdb/a00000001.gdbtable", b"1")
            zf.writestr("Root/A_Moderate_x.gdb/gdb", b"2")
            zf.writestr("Root/B_Extreme_y.gdb/a00000001.gdbtable", b"3")
            zf.writestr("Root/C_Other.gdb/a00000001.gdbtable", b"4")
        return zp

    def test_only_requested_geodatabases_extracted(self, tmp_path):
        from floodmap.stormwater import _extract_scenarios

        root = tmp_path / "out"
        _extract_scenarios(self._zip(tmp_path), ("Moderate_x", "Extreme_y"), root)
        assert (root / "Moderate_x.gdb" / "a00000001.gdbtable").read_bytes() == b"1"
        assert (root / "Moderate_x.gdb" / "gdb").exists()
        assert (root / "Extreme_y.gdb" / "a00000001.gdbtable").read_bytes() == b"3"
        assert not (root / "Other.gdb").exists()

    def test_missing_scenario_raises(self, tmp_path):
        from floodmap.stormwater import _extract_scenarios

        with pytest.raises(RuntimeError):
            _extract_scenarios(self._zip(tmp_path), ("Nope",), tmp_path / "out")
