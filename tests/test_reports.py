"""
test_reports.py — Unit tests for floodmap.reports module.
"""

from __future__ import annotations

import pandas as pd
import pytest

from floodmap.config import FLOOD_DESCRIPTORS


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_raw_reports(n: int = 20) -> pd.DataFrame:
    """
    Create a small synthetic raw 311 DataFrame that mimics Socrata response.

    Parameters
    ----------
    n: Number of rows to generate.

    Returns
    -------
    DataFrame with the same columns as the Socrata API response.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    # Mix of flood and non-flood descriptors
    descriptors = (
        FLOOD_DESCRIPTORS * (n // len(FLOOD_DESCRIPTORS) + 1)
    )[:n]
    # Inject a couple of non-flood descriptors
    descriptors[0] = "Noise - Street/Sidewalk (NP1)"
    descriptors[1] = "HEAT/HOT WATER"

    lats = rng.uniform(40.635, 40.680, size=n).astype(str)
    lons = rng.uniform(-74.025, -73.960, size=n).astype(str)

    return pd.DataFrame(
        {
            "unique_key": [str(i) for i in range(n)],
            "created_date": pd.date_range("2020-07-01", periods=n, freq="6h").astype(str),
            "descriptor": descriptors,
            "latitude": lats,
            "longitude": lons,
            "incident_zip": ["11201"] * n,
            "borough": ["BROOKLYN"] * n,
            "status": ["Closed"] * n,
        }
    )


@pytest.fixture
def raw_reports():
    """Synthetic raw 311 reports with mixed flood/non-flood descriptors."""
    return _make_raw_reports(20)


@pytest.fixture
def bbox():
    """Default southern Brooklyn bbox."""
    return (40.635, -74.025, 40.680, -73.960)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDescriptorFiltering:
    """Verify that only valid flood descriptors are kept."""

    def test_non_flood_descriptors_removed(self, raw_reports, bbox):
        """Non-flood complaint types must be filtered out."""
        from floodmap.reports import _clean_reports

        lat_min, lon_min, lat_max, lon_max = bbox
        cleaned = _clean_reports(raw_reports, lat_min, lon_min, lat_max, lon_max)
        assert cleaned["descriptor"].isin(FLOOD_DESCRIPTORS).all(), (
            "Non-flood descriptors survived cleaning"
        )

    def test_all_flood_descriptors_accepted(self, bbox):
        """All four flood descriptor types must be kept."""
        import numpy as np

        rng = np.random.default_rng(1)
        rows = []
        for desc in FLOOD_DESCRIPTORS:
            rows.append(
                {
                    "unique_key": desc[:5],
                    "created_date": "2021-08-22T10:00:00.000",
                    "descriptor": desc,
                    "latitude": str(rng.uniform(40.640, 40.675)),
                    "longitude": str(rng.uniform(-74.020, -73.965)),
                    "incident_zip": "11212",
                    "borough": "BROOKLYN",
                    "status": "Closed",
                }
            )
        df = pd.DataFrame(rows)
        lat_min, lon_min, lat_max, lon_max = bbox
        from floodmap.reports import _clean_reports

        cleaned = _clean_reports(df, lat_min, lon_min, lat_max, lon_max)
        assert len(cleaned) == len(FLOOD_DESCRIPTORS)


class TestBboxFiltering:
    """Verify that records outside the bbox are removed."""

    def test_out_of_bbox_removed(self, bbox):
        """Records with coordinates outside the bbox must be dropped."""
        import numpy as np
        from floodmap.reports import _clean_reports

        lat_min, lon_min, lat_max, lon_max = bbox
        # One in-bbox row, one outside
        df = pd.DataFrame(
            {
                "unique_key": ["in_bbox", "out_of_bbox"],
                "created_date": ["2021-08-22T10:00:00.000"] * 2,
                "descriptor": [FLOOD_DESCRIPTORS[0]] * 2,
                "latitude": [str(40.650), str(41.000)],  # 41.0 is outside
                "longitude": [str(-74.000), str(-73.950)],
                "incident_zip": ["11212", "11212"],
                "borough": ["BROOKLYN", "BROOKLYN"],
                "status": ["Closed", "Closed"],
            }
        )
        cleaned = _clean_reports(df, lat_min, lon_min, lat_max, lon_max)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["unique_key"] == "in_bbox"


class TestDeduplication:
    """Verify one-report-per-(cell, date) deduplication."""

    def test_duplicate_cell_date_collapsed(self, bbox):
        """Multiple reports in the same cell on the same day collapse to one."""
        import numpy as np
        from floodmap.reports import _clean_reports

        lat_min, lon_min, lat_max, lon_max = bbox
        # Two reports at the same location on the same day — only one should survive
        df = pd.DataFrame(
            {
                "unique_key": ["A", "B"],
                "created_date": [
                    "2021-08-22T09:00:00.000",
                    "2021-08-22T14:00:00.000",
                ],
                "descriptor": [FLOOD_DESCRIPTORS[0]] * 2,
                "latitude": ["40.650", "40.650"],
                "longitude": ["-74.000", "-74.000"],
                "incident_zip": ["11212", "11212"],
                "borough": ["BROOKLYN", "BROOKLYN"],
                "status": ["Closed", "Closed"],
            }
        )
        cleaned = _clean_reports(df, lat_min, lon_min, lat_max, lon_max)
        assert len(cleaned) == 1, "Duplicate (cell, date) rows should collapse to 1"

    def test_different_dates_not_collapsed(self, bbox):
        """Reports in the same cell but on different days should both be kept."""
        from floodmap.reports import _clean_reports

        lat_min, lon_min, lat_max, lon_max = bbox
        df = pd.DataFrame(
            {
                "unique_key": ["C", "D"],
                "created_date": [
                    "2021-08-22T10:00:00.000",
                    "2021-08-23T10:00:00.000",
                ],
                "descriptor": [FLOOD_DESCRIPTORS[0]] * 2,
                "latitude": ["40.650", "40.650"],
                "longitude": ["-74.000", "-74.000"],
                "incident_zip": ["11212", "11212"],
                "borough": ["BROOKLYN", "BROOKLYN"],
                "status": ["Closed", "Closed"],
            }
        )
        cleaned = _clean_reports(df, lat_min, lon_min, lat_max, lon_max)
        assert len(cleaned) == 2, "Different-day reports in same cell should both survive"

    def test_different_cells_not_collapsed(self, bbox):
        """Reports in different cells on the same day should both be kept."""
        from floodmap.reports import _clean_reports

        lat_min, lon_min, lat_max, lon_max = bbox
        # Place reports ~500 m apart (more than one 100-m cell apart)
        df = pd.DataFrame(
            {
                "unique_key": ["E", "F"],
                "created_date": [
                    "2021-08-22T10:00:00.000",
                    "2021-08-22T10:00:00.000",
                ],
                "descriptor": [FLOOD_DESCRIPTORS[0]] * 2,
                "latitude": ["40.640", "40.675"],   # ~3.9 km apart
                "longitude": ["-74.020", "-73.970"],
                "incident_zip": ["11212", "11212"],
                "borough": ["BROOKLYN", "BROOKLYN"],
                "status": ["Closed", "Closed"],
            }
        )
        cleaned = _clean_reports(df, lat_min, lon_min, lat_max, lon_max)
        assert len(cleaned) == 2, "Reports in different cells should both survive"


class TestTopFloodAreas:
    """Test the top_flood_areas helper."""

    def test_returns_top_n(self):
        """top_flood_areas should return exactly top_n rows."""
        from floodmap.reports import top_flood_areas

        df = pd.DataFrame(
            {
                "latitude": [40.640, 40.641, 40.650, 40.651, 40.660, 40.661],
                "longitude": [-74.000, -74.001, -73.990, -73.991, -73.980, -73.981],
            }
        )
        result = top_flood_areas(df, top_n=2)
        assert len(result) == 2

    def test_sorted_descending(self):
        """Areas should be returned from most to least reports."""
        from floodmap.reports import top_flood_areas

        df = pd.DataFrame(
            {
                "latitude": [40.640] * 5 + [40.660] * 2,
                "longitude": [-74.000] * 5 + [-73.980] * 2,
            }
        )
        result = top_flood_areas(df, top_n=2)
        assert result.iloc[0]["count"] >= result.iloc[1]["count"]
