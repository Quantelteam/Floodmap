"""
test_rain.py — Unit tests for storm-event identification and rain aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodmap.config import EVENT_GAP_DAYS, MIN_RAIN_MM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_rain_simple():
    """
    Simple 30-day synthetic daily rain series with two obvious storm events.

    Event 1: days 5–6 (heavy rain, 25 + 30 mm)
    Event 2: days 20–22 (heavy rain, 15 + 20 + 18 mm)
    All other days have < MIN_RAIN_MM.
    """
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    prcp = np.zeros(30, dtype=float)
    prcp[4] = 25.0   # day 5
    prcp[5] = 30.0   # day 6  → event 1 (consecutive)
    prcp[19] = 15.0  # day 20
    prcp[20] = 20.0  # day 21 → event 2 (consecutive)
    prcp[21] = 18.0  # day 22
    return pd.DataFrame({"date": dates, "prcp_mm": prcp})


@pytest.fixture
def flood_reports_matching():
    """
    Synthetic 311 reports that match the two storm events in *daily_rain_simple*.
    """
    return pd.DataFrame(
        {
            "created_date": pd.to_datetime(
                [
                    "2020-01-05 10:00",
                    "2020-01-05 11:00",
                    "2020-01-05 12:00",  # 3 reports in event 1 window
                    "2020-01-20 09:00",
                    "2020-01-20 10:00",
                    "2020-01-20 11:00",  # 3 reports in event 2 window
                ]
            ).tz_localize("America/New_York"),
            "descriptor": ["Street Flooding (SJ)"] * 6,
            "cell_id": [1] * 6,
        }
    )


@pytest.fixture
def daily_rain_no_events():
    """Daily rain series with no day exceeding MIN_RAIN_MM."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    prcp = np.full(30, MIN_RAIN_MM / 2.0)
    return pd.DataFrame({"date": dates, "prcp_mm": prcp})


# ---------------------------------------------------------------------------
# Event identification tests
# ---------------------------------------------------------------------------

class TestEventIdentification:
    """Test that storm events are correctly identified from daily rain."""

    def test_two_events_identified(self, daily_rain_simple):
        """The simple rain series should produce exactly two events."""
        from floodmap.rain import identify_storm_events

        events = identify_storm_events(daily_rain_simple, reports=None)
        assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    def test_event_ids_are_unique(self, daily_rain_simple):
        """Each event must have a distinct event_id."""
        from floodmap.rain import identify_storm_events

        events = identify_storm_events(daily_rain_simple, reports=None)
        assert events["event_id"].nunique() == len(events)

    def test_no_events_below_threshold(self, daily_rain_no_events):
        """No events should be returned when all rain is below the threshold."""
        from floodmap.rain import identify_storm_events

        events = identify_storm_events(daily_rain_no_events, reports=None)
        assert len(events) == 0

    def test_report_filter_applied(self, daily_rain_simple, flood_reports_matching):
        """With exactly 3 reports per event (== MIN_REPORTS_FOR_EVENT=3), both pass."""
        from floodmap.rain import identify_storm_events

        events = identify_storm_events(
            daily_rain_simple,
            reports=flood_reports_matching,
            min_reports=3,
        )
        assert len(events) == 2, "Both events should pass the 3-report threshold"

    def test_report_filter_too_strict(self, daily_rain_simple, flood_reports_matching):
        """With a higher report threshold, events should be filtered out."""
        from floodmap.rain import identify_storm_events

        events = identify_storm_events(
            daily_rain_simple,
            reports=flood_reports_matching,
            min_reports=10,  # more than the 3 synthetic reports
        )
        assert len(events) == 0, "No events should pass a threshold of 10 reports"

    def test_consecutive_days_merged(self):
        """Days within gap_days of each other should merge into a single event."""
        from floodmap.rain import identify_storm_events

        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prcp = np.zeros(10)
        prcp[0] = 20.0   # day 1
        prcp[2] = 20.0   # day 3 — 1 day gap → should merge with day 1
        df = pd.DataFrame({"date": dates, "prcp_mm": prcp})
        events = identify_storm_events(df, reports=None, gap_days=1)
        assert len(events) == 1, (
            "Days with gap <= 1 should merge into a single event"
        )

    def test_distant_days_not_merged(self):
        """Days separated by more than gap_days should produce separate events."""
        from floodmap.rain import identify_storm_events

        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        prcp = np.zeros(20)
        prcp[0] = 20.0   # day 1
        prcp[9] = 20.0   # day 10 — 8 day gap → separate event
        df = pd.DataFrame({"date": dates, "prcp_mm": prcp})
        events = identify_storm_events(df, reports=None, gap_days=1)
        assert len(events) == 2, (
            "Days with gap > 1 should not be merged"
        )


# ---------------------------------------------------------------------------
# Rain aggregation tests
# ---------------------------------------------------------------------------

class TestRainAggregation:
    """Test rain totals and prev_48h aggregation per event."""

    def test_rain_total_matches_sum(self, daily_rain_simple):
        """Rain total for each event should equal the sum of daily values in the window."""
        from floodmap.rain import identify_storm_events

        events = identify_storm_events(daily_rain_simple, reports=None)
        # Event 1: days 5–6 → 25 + 30 = 55 mm
        event1 = events[events["event_id"] == "E0000"].iloc[0]
        assert abs(event1["rain_total_mm"] - 55.0) < 0.1, (
            f"Event 1 rain total {event1['rain_total_mm']} != 55"
        )

    def test_prev_48h_uses_preceding_days(self, daily_rain_simple):
        """rain_prev_48h_mm should sum rain in the 2 days before the event."""
        from floodmap.rain import identify_storm_events

        # Add rain 2 days before event 1 (day 3 = index 2)
        df = daily_rain_simple.copy()
        df.loc[2, "prcp_mm"] = 8.0  # day 3 — 2 days before event1 (day 5)
        df.loc[3, "prcp_mm"] = 5.0  # day 4 — 1 day before event1

        events = identify_storm_events(df, reports=None)
        event1 = events[events["event_id"] == "E0000"].iloc[0]
        # Prev 48h = sum of days [3, 4] = 8 + 5 = 13 mm
        assert abs(event1["rain_prev_48h_mm"] - 13.0) < 0.1, (
            f"Prev 48h rain {event1['rain_prev_48h_mm']} != 13"
        )


# ---------------------------------------------------------------------------
# Hourly peak tests
# ---------------------------------------------------------------------------

class TestHourlyPeak:
    """Test addition of peak 1h rain from hourly data."""

    def test_peak_assigned_correctly(self, daily_rain_simple):
        """The highest 1h rain in the event window should be assigned as peak."""
        from floodmap.rain import add_hourly_peak, identify_storm_events

        events = identify_storm_events(daily_rain_simple, reports=None)

        # Synthetic hourly data: event 1 has max 1h of 15 mm
        hourly = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-01-05 10:00", "2020-01-05 11:00", "2020-01-05 12:00",
                     "2020-01-20 10:00"]
                ).tz_localize("UTC"),
                "prcp_1h_mm": [5.0, 15.0, 8.0, 10.0],
            }
        )

        events_with_peak = add_hourly_peak(events, hourly)
        event1 = events_with_peak[events_with_peak["event_id"] == "E0000"].iloc[0]
        assert abs(event1["rain_peak_1h_mm"] - 15.0) < 0.1

    def test_no_hourly_data_gives_nan(self, daily_rain_simple):
        """When hourly=None, rain_peak_1h_mm should be NaN."""
        from floodmap.rain import add_hourly_peak, identify_storm_events

        events = identify_storm_events(daily_rain_simple, reports=None)
        events_with_peak = add_hourly_peak(events, None)
        assert events_with_peak["rain_peak_1h_mm"].isna().all()


# ---------------------------------------------------------------------------
# GHCN parsing tests
# ---------------------------------------------------------------------------

class TestGHCNParsing:
    """Test GHCN-Daily CSV parsing with synthetic data."""

    def test_prcp_converted_from_tenths(self, tmp_path):
        """PRCP in tenths-of-mm should be divided by 10."""
        from floodmap.rain import _parse_ghcn_csv

        # Create a minimal GHCN CSV with ELEMENT column format
        csv_content = (
            "STATION,DATE,ELEMENT,DATA_VALUE,M_FLAG,Q_FLAG,S_FLAG,OBS_TIME\n"
            "USW00094728,2020-07-01,PRCP,100,,,,\n"  # 100 tenths = 10 mm
            "USW00094728,2020-07-02,PRCP,250,,,,\n"  # 250 tenths = 25 mm
        )
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)

        df = _parse_ghcn_csv(csv_path)
        assert len(df) == 2
        assert abs(df.iloc[0]["prcp_mm"] - 10.0) < 0.01
        assert abs(df.iloc[1]["prcp_mm"] - 25.0) < 0.01

    def test_negative_prcp_clipped_to_zero(self, tmp_path):
        """Negative PRCP values (trace/error) should be clipped to 0."""
        from floodmap.rain import _parse_ghcn_csv

        csv_content = (
            "STATION,DATE,ELEMENT,DATA_VALUE,M_FLAG,Q_FLAG,S_FLAG,OBS_TIME\n"
            "USW00094728,2020-07-01,PRCP,-10,,,,\n"
        )
        csv_path = tmp_path / "test_neg.csv"
        csv_path.write_text(csv_content)

        df = _parse_ghcn_csv(csv_path)
        assert df.iloc[0]["prcp_mm"] == 0.0


class TestISDParsing:
    """ISD AA1 field is ``period_h,depth_tenths_mm,condition,quality``."""

    def _write(self, tmp_path, rows):
        df = pd.DataFrame(rows, columns=["DATE", "AA1", "AA2"])
        p = tmp_path / "isd.csv"
        df.to_csv(p, index=False)
        return p

    def test_depth_and_period_not_swapped(self, tmp_path):
        from floodmap.rain import _parse_isd_csv

        p = self._write(tmp_path, [["2023-01-01T00:51:00", "01,0100,9,5", None]])
        out = _parse_isd_csv(p)
        assert out["prcp_1h_mm"].tolist() == [10.0]

    def test_only_hourly_valid_records_kept(self, tmp_path):
        from floodmap.rain import _parse_isd_csv

        p = self._write(
            tmp_path,
            [
                ["2023-01-01T00:51:00", "06,0100,3,1", None],  # 6 h accumulation -> ignored
                ["2023-01-01T01:51:00", "01,9999,9,5", None],  # missing depth -> ignored
                ["2023-01-01T02:51:00", "01,0050,9,2", None],  # suspect quality -> ignored
                ["2023-01-01T03:51:00", "01,0020,2,1", None],  # trace -> 0 mm
                ["2023-01-01T04:51:00", "01,0030,9,5", None],
            ],
        )
        out = _parse_isd_csv(p)
        assert out["prcp_1h_mm"].tolist() == [0.0, 3.0]

    def test_reports_in_same_hour_reduced_with_max(self, tmp_path):
        from floodmap.rain import _parse_isd_csv

        p = self._write(
            tmp_path,
            [["2023-01-01T00:10:00", "01,0010,9,5", None], ["2023-01-01T00:51:00", "01,0040,9,5", None]],
        )
        assert _parse_isd_csv(p)["prcp_1h_mm"].tolist() == [4.0]

    def test_html_error_page_returns_none(self, tmp_path):
        from floodmap.rain import _parse_isd_csv

        p = tmp_path / "bad.csv"
        p.write_text("<!DOCTYPE html><html><body>not found</body></html>")
        assert _parse_isd_csv(p) is None


class TestHourlyPeakEstimation:
    """Events beyond the ISD record get an estimated, not a NaN/median, peak."""

    def test_missing_peak_estimated_from_total(self):
        from floodmap.rain import add_hourly_peak

        n = 12
        events = pd.DataFrame(
            {
                "start_date": pd.date_range("2020-01-01", periods=n + 1, freq="30D"),
                "rain_total_mm": list(np.linspace(20, 80, n)) + [50.0],
            }
        )
        events["end_date"] = events["start_date"]
        hourly = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(events["start_date"][:n]).dt.tz_localize("UTC")
                + pd.Timedelta(hours=5),
                "prcp_1h_mm": events["rain_total_mm"][:n].values * 0.25,
            }
        )
        out = add_hourly_peak(events, hourly)
        assert out["rain_peak_1h_mm"].notna().all()
        assert out["rain_peak_1h_mm"].iloc[-1] == pytest.approx(12.5, rel=0.05)
