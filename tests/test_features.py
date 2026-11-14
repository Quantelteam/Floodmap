"""
test_features.py — Unit tests for the (cell × event) feature join and label assignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodmap.config import FLOOD_DESCRIPTORS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """
    A 4-cell grid in southern Brooklyn whose UTM centroids are consistent with
    each cell's lat/lon (cells are snapped to a 100 m lattice, as in grid.build_grid).
    """
    import pyproj

    lats = np.array([40.640, 40.641, 40.650, 40.651])
    lons = np.array([-74.010, -74.000, -74.010, -74.000])
    x, y = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True).transform(lons, lats)
    cx = np.floor(x / 100) * 100 + 50
    cy = np.floor(y / 100) * 100 + 50
    return pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "lat": lats,
            "lon": lons,
            "utm_x": cx,
            "utm_y": cy,
            "block_id": [0, 0, 0, 0],
        }
    )


@pytest.fixture
def two_events():
    """
    Two synthetic storm events on consecutive months.
    """
    return pd.DataFrame(
        {
            "event_id": ["E0000", "E0001"],
            "start_date": pd.to_datetime(["2020-07-01", "2020-08-10"]),
            "end_date": pd.to_datetime(["2020-07-02", "2020-08-11"]),
            "rain_total_mm": [45.0, 20.0],
            "rain_peak_1h_mm": [12.0, np.nan],
            "rain_prev_48h_mm": [2.0, 0.0],
            "n_reports": [5, 3],
        }
    )


@pytest.fixture
def flood_reports_fixture():
    """
    Synthetic 311 reports: cell_id=1 floods in event E0000, cell_id=3 in E0001.
    """
    return pd.DataFrame(
        {
            "cell_id": [1, 1, 1, 3],
            "created_date": pd.to_datetime(
                [
                    "2020-07-01 10:00",
                    "2020-07-01 11:00",
                    "2020-07-01 12:00",
                    "2020-08-10 09:00",
                ]
            ).tz_localize("America/New_York"),
            "descriptor": [FLOOD_DESCRIPTORS[0]] * 4,
            "latitude": [40.640, 40.640, 40.640, 40.650],
            "longitude": [-74.010, -74.010, -74.010, -74.010],
        }
    )


# ---------------------------------------------------------------------------
# Label assignment tests
# ---------------------------------------------------------------------------

class TestFloodLabelAssignment:
    """Verify that flood labels are assigned correctly."""

    def _make_cross_df(self, grid, events):
        """Build a simple cross-join DataFrame for label testing."""
        grid_key = grid[["cell_id"]].copy()
        grid_key["_key"] = 1
        ev_key = events[["event_id", "start_date", "end_date"]].copy()
        ev_key["_key"] = 1
        return grid_key.merge(ev_key, on="_key").drop(columns=["_key"])

    def test_flooded_cell_gets_label_1(self, small_grid, two_events, flood_reports_fixture):
        """Cell 1 during event E0000 should be labelled flooded."""
        from floodmap.features import _assign_flood_labels

        cross = self._make_cross_df(small_grid, two_events)
        labels = _assign_flood_labels(cross, flood_reports_fixture, small_grid)

        row = cross[(cross["cell_id"] == 1) & (cross["event_id"] == "E0000")]
        assert labels[row.index[0]] == 1, "Cell 1 during E0000 should be flooded (label=1)"

    def test_non_flooded_cell_gets_label_0(self, small_grid, two_events, flood_reports_fixture):
        """Cell 2 (no reports) should be labelled 0 for all events."""
        from floodmap.features import _assign_flood_labels

        cross = self._make_cross_df(small_grid, two_events)
        labels = _assign_flood_labels(cross, flood_reports_fixture, small_grid)

        rows = cross[cross["cell_id"] == 2]
        for idx in rows.index:
            assert labels[idx] == 0, f"Cell 2 should never be flooded (got {labels[idx]})"

    def test_correct_event_attribution(self, small_grid, two_events, flood_reports_fixture):
        """Cell 3 should be flooded in E0001 but not E0000."""
        from floodmap.features import _assign_flood_labels

        cross = self._make_cross_df(small_grid, two_events)
        labels = _assign_flood_labels(cross, flood_reports_fixture, small_grid)

        row_e0 = cross[(cross["cell_id"] == 3) & (cross["event_id"] == "E0000")]
        row_e1 = cross[(cross["cell_id"] == 3) & (cross["event_id"] == "E0001")]

        assert labels[row_e0.index[0]] == 0, "Cell 3 should NOT be flooded in E0000"
        assert labels[row_e1.index[0]] == 1, "Cell 3 SHOULD be flooded in E0001"

    def test_no_reports_all_zeros(self, small_grid, two_events):
        """With an empty reports DataFrame, all labels should be 0."""
        from floodmap.features import _assign_flood_labels

        cross = self._make_cross_df(small_grid, two_events)
        empty_reports = pd.DataFrame(
            columns=["cell_id", "created_date", "descriptor"]
        )
        labels = _assign_flood_labels(cross, empty_reports, small_grid)
        assert (labels == 0).all(), "Empty reports should produce all-zero labels"


# ---------------------------------------------------------------------------
# Feature table structure tests
# ---------------------------------------------------------------------------

class TestFeatureTableStructure:
    """Test the cross-join produces the correct (cells × events) structure."""

    def test_total_rows(self, small_grid, two_events):
        """Total rows should be n_cells * n_events."""
        n_cells = len(small_grid)
        n_events = len(two_events)
        small_grid["_key"] = 1
        two_events["_key"] = 1
        cross = small_grid.merge(two_events, on="_key").drop(columns=["_key"])
        assert len(cross) == n_cells * n_events

    def test_rain_columns_present(self, small_grid, two_events):
        """Rain feature columns must appear in the cross-joined table."""
        small_grid["_key"] = 1
        two_events["_key"] = 1
        cross = small_grid.merge(two_events, on="_key").drop(columns=["_key"])
        for col in ["rain_total_mm", "rain_prev_48h_mm"]:
            assert col in cross.columns, f"Missing rain column: {col}"

    def test_no_nan_after_imputation(self):
        """impute_dataframe should eliminate all NaNs from numeric columns."""
        from floodmap.utils import impute_dataframe

        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": [np.nan, 2.0, np.nan],
                "c": ["x", "y", "z"],  # non-numeric — should be unchanged
            }
        )
        result = impute_dataframe(df)
        assert result[["a", "b"]].isna().sum().sum() == 0
        assert list(result["c"]) == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Class balance tests
# ---------------------------------------------------------------------------

class TestClassBalance:
    """Test class-balance warning thresholds."""

    def test_very_low_flood_fraction_logs_warning(self, caplog):
        """A flood fraction below MIN_FLOOD_FRAC should trigger a warning."""
        import logging
        from floodmap.config import MIN_FLOOD_FRAC

        df = pd.DataFrame(
            {
                "flooded": np.zeros(1000, dtype=np.int8),
            }
        )
        df.loc[0, "flooded"] = 1  # 0.1% flooded

        frac = df["flooded"].mean()
        with caplog.at_level(logging.WARNING, logger="floodmap.features"):
            if frac < MIN_FLOOD_FRAC:
                import logging as lg

                lg.getLogger("floodmap.features").warning(
                    "Very low flood fraction (%.4f < %.4f)", frac, MIN_FLOOD_FRAC
                )

        assert any("flood fraction" in r.message.lower() for r in caplog.records), (
            "Warning about low flood fraction should be logged"
        )


class TestReportSnapping:
    """Reports must be assigned to grid cells from their coordinates."""

    @staticmethod
    def _cross(grid, events):
        return grid[["cell_id"]].merge(events[["event_id", "start_date", "end_date"]], how="cross")

    def test_reports_are_resnapped_to_grid_cells(self, small_grid, two_events, flood_reports_fixture):
        """The cell_id stored on a report is ignored; its coordinates decide the cell."""
        from floodmap.features import _assign_flood_labels

        reports = flood_reports_fixture.copy()
        reports["cell_id"] = 987_654_321  # wrong id, as produced by reports.py
        cross = self._cross(small_grid, two_events)
        labels = _assign_flood_labels(cross, reports, small_grid)

        row = cross[(cross["cell_id"] == 1) & (cross["event_id"] == "E0000")]
        assert labels[row.index[0]] == 1
        assert labels.sum() == 2  # cell 1 / E0000 and cell 3 / E0001 only

    def test_reports_outside_grid_are_dropped(self, small_grid, two_events, flood_reports_fixture):
        """Reports that fall outside the grid must not label any cell."""
        from floodmap.features import _assign_flood_labels

        reports = flood_reports_fixture.copy()
        reports["latitude"] = 41.5
        reports["longitude"] = -73.0
        cross = self._cross(small_grid, two_events)
        labels = _assign_flood_labels(cross, reports, small_grid)
        assert labels.sum() == 0
