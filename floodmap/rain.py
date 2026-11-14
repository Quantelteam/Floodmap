from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from floodmap.config import (
    DATA_RAW,
    EVENT_GAP_DAYS,
    GHCN_BASE_URL,
    ISD_BASE_URL,
    ISD_STATION_ID,
    MIN_RAIN_MM,
    MIN_REPORTS_FOR_EVENT,
    RAIN_STATION_ALT,
    RAIN_STATION_ID,
)
from floodmap.utils import download_file

logger = logging.getLogger(__name__)

_RAIN_DAILY_PATH = DATA_RAW / "rain_daily.parquet"
_RAIN_HOURLY_PATH = DATA_RAW / "rain_hourly.parquet"
_EVENTS_PATH = DATA_RAW / "events.parquet"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_ghcn_daily(
    station_id: str = RAIN_STATION_ID,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download GHCN-Daily CSV for *station_id* from NOAA and return a parsed DataFrame.

    Precipitation (PRCP) is in tenths-of-mm; this function converts to mm.
    Also extracts SNOW and SNWD fields where available.

    Parameters
    ----------
    station_id: GHCN station identifier (default: Central Park).
    force:      Re-download even if the cache exists.

    Returns
    -------
    DataFrame with columns: date (datetime), prcp_mm (float), snow_mm (float).
    """
    csv_path = DATA_RAW / f"ghcn_{station_id}.csv"
    url = f"{GHCN_BASE_URL}{station_id}.csv"

    download_file(url, csv_path, force=force, desc=f"GHCN {station_id}")

    df = _parse_ghcn_csv(csv_path)
    logger.info(
        "GHCN daily: %d rows, date range %s → %s",
        len(df),
        df["date"].min().date() if len(df) else "N/A",
        df["date"].max().date() if len(df) else "N/A",
    )
    return df


def _parse_ghcn_csv(csv_path: Path) -> pd.DataFrame:
    """
    Parse a GHCN-Daily CSV file.

    GHCN CSVs have one row per (station, date, element) with a VALUE column.
    We pivot on the ELEMENT field and extract PRCP (and optionally SNOW).

    Parameters
    ----------
    csv_path: Path to the downloaded CSV.

    Returns
    -------
    DataFrame indexed by date with 'prcp_mm' and optionally 'snow_mm'.
    """
    raw = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names (GHCN headers vary slightly)
    raw.columns = raw.columns.str.strip().str.upper()

    if "ELEMENT" in raw.columns:
        # Wide-format with ELEMENT column (standard GHCN-Daily access CSV)
        raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")
        raw = raw.dropna(subset=["DATE"])

        # Extract PRCP
        prcp = raw[raw["ELEMENT"] == "PRCP"][["DATE", "DATA_VALUE"]].copy()
        prcp = prcp.rename(columns={"DATA_VALUE": "prcp_tenths"})
        prcp["prcp_mm"] = pd.to_numeric(prcp["prcp_tenths"], errors="coerce") / 10.0
        prcp["date"] = prcp["DATE"]
        result = prcp[["date", "prcp_mm"]].copy()

        # Extract SNOW if present
        snow = raw[raw["ELEMENT"] == "SNOW"][["DATE", "DATA_VALUE"]].copy()
        if not snow.empty:
            snow["snow_mm"] = pd.to_numeric(snow["DATA_VALUE"], errors="coerce") / 10.0
            snow = snow[["DATE", "snow_mm"]].rename(columns={"DATE": "date"})
            result = result.merge(snow, on="date", how="left")
        else:
            result["snow_mm"] = np.nan

    else:
        # Columnar format (some GHCN access files are pre-pivoted)
        raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")
        raw = raw.dropna(subset=["DATE"])
        result = raw[["DATE"]].copy().rename(columns={"DATE": "date"})
        if "PRCP" in raw.columns:
            result["prcp_mm"] = pd.to_numeric(raw["PRCP"], errors="coerce") / 10.0
        else:
            result["prcp_mm"] = np.nan
        result["snow_mm"] = np.nan

    result = result.sort_values("date").reset_index(drop=True)
    result["prcp_mm"] = result["prcp_mm"].clip(lower=0).fillna(0.0)
    result["snow_mm"] = result["snow_mm"].clip(lower=0).fillna(0.0)
    return result


def download_isd_hourly(
    station_id: str = ISD_STATION_ID,
    start_year: int = 2016,
    end_year: int | None = None,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Download NOAA ISD (Integrated Surface Database) hourly data for *station_id*.

    Downloads one CSV per year, caches each file individually, then concatenates.
    Returns None if no hourly data is available.

    Parameters
    ----------
    station_id: ISD station identifier (USAF+WBAN).
    start_year: First year to download (inclusive).
    end_year:   Last year to download (inclusive).  Defaults to current year.
    force:      Re-download cached files.

    Returns
    -------
    DataFrame with columns: timestamp (datetime tz-aware), prcp_1h_mm (float),
    or None if download fails for all years.
    """
    import datetime

    if end_year is None:
        end_year = datetime.date.today().year

    frames: list[pd.DataFrame] = []
    years = list(range(start_year, end_year + 1))

    for year in tqdm(years, desc="ISD hourly years", unit="yr"):
        cached = DATA_RAW / f"isd_{station_id}_{year}.csv"
        url = f"{ISD_BASE_URL}{year}/{station_id}.csv"
        try:
            download_file(url, cached, force=force, desc=f"ISD {year}", retries=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ISD year %d unavailable: %s", year, exc)
            continue

        try:
            parsed = _parse_isd_csv(cached)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse ISD %d: %s", year, exc)
            continue
        if parsed is None:
            # Not a CSV (e.g. an HTML error page) -- drop it so it is never cached as data.
            logger.warning("ISD %d file is not valid CSV; removing %s", year, cached.name)
            cached.unlink(missing_ok=True)
        elif len(parsed):
            frames.append(parsed)

    if not frames:
        logger.warning("No ISD hourly data available; hourly rain features will be NaN")
        return None

    hourly = pd.concat(frames, ignore_index=True)
    hourly = hourly.sort_values("timestamp").reset_index(drop=True)
    logger.info("ISD hourly: %d rows loaded", len(hourly))
    return hourly


def _parse_isd_csv(csv_path: Path) -> pd.DataFrame | None:
    """
    Parse an ISD hourly CSV into hourly precipitation depth.

    Precipitation lives in the ``AA1``..``AA3`` fields, each encoded
    ``period_hours,depth_tenths_mm,condition,quality``.  Only 1-hour periods are
    used; a depth of 9999 means missing, condition ``2`` is a trace (counted as 0),
    and quality flags 2/3/6/7 (suspect/erroneous) are dropped.  Reports from
    several sources in the same clock hour are reduced with ``max``.

    Parameters
    ----------
    csv_path: Path to the ISD CSV for one year.

    Returns
    -------
    DataFrame with UTC ``timestamp`` (hour start) and ``prcp_1h_mm``, or None if the
    file is not a valid ISD CSV (e.g. an HTML error page).
    """
    try:
        raw = pd.read_csv(csv_path, low_memory=False)
    except Exception:  # noqa: BLE001
        return None

    raw.columns = raw.columns.str.strip().str.upper()
    if "DATE" not in raw.columns:
        return None

    raw["timestamp"] = pd.to_datetime(raw["DATE"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["timestamp"])

    pieces = []
    for col in ("AA1", "AA2", "AA3"):
        if col not in raw.columns:
            continue
        parts = raw[col].astype(str).str.split(",", expand=True)
        if parts.shape[1] < 4:
            continue
        period = pd.to_numeric(parts[0], errors="coerce")
        depth = pd.to_numeric(parts[1], errors="coerce")
        cond = parts[2].str.strip()
        qual = parts[3].str.strip()
        ok = (period == 1) & depth.notna() & (depth != 9999) & ~qual.isin(["2", "3", "6", "7"])
        mm = (depth / 10.0).where(cond != "2", 0.0)
        pieces.append(
            pd.DataFrame(
                {"timestamp": raw.loc[ok, "timestamp"], "prcp_1h_mm": mm[ok].clip(lower=0)}
            )
        )

    if not pieces:
        return pd.DataFrame(columns=["timestamp", "prcp_1h_mm"])

    out = pd.concat(pieces, ignore_index=True)
    out["timestamp"] = out["timestamp"].dt.floor("h")
    return out.groupby("timestamp", as_index=False)["prcp_1h_mm"].max()


# ---------------------------------------------------------------------------
# Storm event identification
# ---------------------------------------------------------------------------

def identify_storm_events(
    daily: pd.DataFrame,
    reports: pd.DataFrame | None = None,
    gap_days: int = EVENT_GAP_DAYS,
    min_rain_mm: float = MIN_RAIN_MM,
    min_reports: int = MIN_REPORTS_FOR_EVENT,
) -> pd.DataFrame:
    """
    Identify storm events from daily rain data and (optionally) 311 reports.

    A *candidate rain day* is any day where daily precipitation >= *min_rain_mm*.
    Consecutive candidate days within *gap_days* of each other are merged into
    a single event.

    If *reports* is provided, an event is only kept when at least *min_reports*
    311 flood reports fall in the event window.  When *reports* is None, all
    rainy periods are retained.

    Parameters
    ----------
    daily:       DataFrame with 'date' and 'prcp_mm' columns.
    reports:     Optional cleaned 311 reports DataFrame (column 'created_date').
    gap_days:    Max gap in days between consecutive rainy days to merge.
    min_rain_mm: Rain threshold in mm.
    min_reports: Min 311 reports within the event window to confirm it.

    Returns
    -------
    DataFrame with columns: event_id, start_date, end_date, rain_total_mm,
    rain_prev_48h_mm, n_reports.
    """
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])

    rainy = daily[daily["prcp_mm"] >= min_rain_mm].copy()

    if rainy.empty:
        logger.warning("No days with rain >= %.1f mm found", min_rain_mm)
        return _empty_events_df()

    # Group consecutive rainy days
    rainy = rainy.sort_values("date").reset_index(drop=True)
    rainy["gap"] = (rainy["date"].diff().dt.days > gap_days + 1).cumsum()

    events = (
        rainy.groupby("gap")
        .agg(start_date=("date", "min"), end_date=("date", "max"))
        .reset_index(drop=True)
    )

    # For each event aggregate rain stats
    rows = []
    for _, ev in tqdm(events.iterrows(), total=len(events), desc="Storm events"):
        mask = (daily["date"] >= ev["start_date"]) & (daily["date"] <= ev["end_date"])
        rain_total = daily.loc[mask, "prcp_mm"].sum()

        # Previous 48h rain (2 days before start)
        prev_start = ev["start_date"] - pd.Timedelta(days=2)
        prev_mask = (daily["date"] >= prev_start) & (daily["date"] < ev["start_date"])
        rain_prev = daily.loc[prev_mask, "prcp_mm"].sum()

        # Count reports in window
        n_rep = 0
        if reports is not None and len(reports) > 0:
            start_ts = pd.Timestamp(ev["start_date"]).tz_localize("America/New_York")
            end_ts = (
                pd.Timestamp(ev["end_date"]) + pd.Timedelta(days=1)
            ).tz_localize("America/New_York")
            rep_dates = reports["created_date"]
            n_rep = int(((rep_dates >= start_ts) & (rep_dates <= end_ts)).sum())

        rows.append(
            {
                "start_date": ev["start_date"],
                "end_date": ev["end_date"],
                "rain_total_mm": rain_total,
                "rain_prev_48h_mm": rain_prev,
                "n_reports": n_rep,
            }
        )

    result = pd.DataFrame(rows)

    # Filter by report count if reports provided
    if reports is not None:
        before = len(result)
        result = result[result["n_reports"] >= min_reports].reset_index(drop=True)
        logger.info(
            "Events after report-count filter (>=%d): %d / %d",
            min_reports,
            len(result),
            before,
        )

    # Assign event IDs
    result.insert(0, "event_id", [f"E{i:04d}" for i in range(len(result))])
    result = result.reset_index(drop=True)
    logger.info("Identified %d storm events", len(result))
    return result


def add_hourly_peak(
    events: pd.DataFrame,
    hourly: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Add ``rain_peak_1h_mm`` column to *events* from hourly data.

    If *hourly* is None or empty, the column is filled with NaN.

    Parameters
    ----------
    events: Storm events DataFrame (from :func:`identify_storm_events`).
    hourly: Optional hourly rain DataFrame with 'timestamp' and 'prcp_1h_mm'.

    Returns
    -------
    Events DataFrame with an additional 'rain_peak_1h_mm' column.
    """
    events = events.copy()
    events["rain_peak_1h_mm"] = np.nan

    if hourly is None or hourly.empty:
        return events

    for idx, row in events.iterrows():
        start_ts = pd.Timestamp(row["start_date"]).tz_localize("UTC")
        end_ts = (
            pd.Timestamp(row["end_date"]) + pd.Timedelta(days=1)
        ).tz_localize("UTC")
        mask = (hourly["timestamp"] >= start_ts) & (hourly["timestamp"] <= end_ts)
        window = hourly.loc[mask, "prcp_1h_mm"]
        if not window.empty:
            events.at[idx, "rain_peak_1h_mm"] = window.max()

    # The public ISD archive lags by months, so the newest events may have no
    # hourly record.  Estimate their peak from the event total with a log-log fit
    # on events that do have one (peak hourly rain scales tightly with the total),
    # rather than letting the later median-imputation flatten them all to one value.
    missing = events["rain_peak_1h_mm"].isna()
    known = ~missing & (events["rain_total_mm"] > 0) & (events["rain_peak_1h_mm"] > 0)
    if missing.any() and known.sum() >= 10:
        slope, intercept = np.polyfit(
            np.log(events.loc[known, "rain_total_mm"]),
            np.log(events.loc[known, "rain_peak_1h_mm"]),
            1,
        )
        est = np.exp(intercept + slope * np.log(events.loc[missing, "rain_total_mm"].clip(lower=0.1)))
        events.loc[missing, "rain_peak_1h_mm"] = est.round(1)
        logger.warning(
            "%d event(s) have no hourly rain record (ISD archive ends %s); "
            "rain_peak_1h_mm estimated from rain_total_mm (log-log fit, n=%d)",
            int(missing.sum()), hourly["timestamp"].max().date(), int(known.sum()),
        )

    return events


def _empty_events_df() -> pd.DataFrame:
    """Return an empty events DataFrame with the correct schema."""
    return pd.DataFrame(
        columns=[
            "event_id", "start_date", "end_date",
            "rain_total_mm", "rain_peak_1h_mm", "rain_prev_48h_mm", "n_reports",
        ]
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_rain(
    bbox: Tuple[float, float, float, float] | None = None,
    force: bool = False,
    reports: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full rain pipeline: download daily + hourly data, identify storm events.

    Saves to ``data/raw/rain_daily.parquet``, ``data/raw/rain_hourly.parquet``,
    and ``data/raw/events.parquet``.

    Parameters
    ----------
    bbox:    Bounding box (unused directly, passed for consistency).
    force:   Re-download cached files.
    reports: Cleaned 311 reports for event confirmation.

    Returns
    -------
    Tuple of (rain_daily DataFrame, events DataFrame).
    """
    # Daily rain
    if Path(_RAIN_DAILY_PATH).exists() and not force:
        logger.info("Loading cached daily rain from %s", _RAIN_DAILY_PATH)
        daily = pd.read_parquet(_RAIN_DAILY_PATH)
    else:
        daily = download_ghcn_daily(force=force)
        daily.to_parquet(_RAIN_DAILY_PATH, index=False)
        logger.info("Saved daily rain to %s", _RAIN_DAILY_PATH)

    # Hourly rain (best-effort)
    if Path(_RAIN_HOURLY_PATH).exists() and not force:
        logger.info("Loading cached hourly rain from %s", _RAIN_HOURLY_PATH)
        hourly = pd.read_parquet(_RAIN_HOURLY_PATH)
    else:
        hourly_raw = download_isd_hourly(force=force)
        if hourly_raw is not None:
            hourly_raw.to_parquet(_RAIN_HOURLY_PATH, index=False)
            logger.info("Saved hourly rain to %s", _RAIN_HOURLY_PATH)
            hourly = hourly_raw
        else:
            hourly = pd.DataFrame(columns=["timestamp", "prcp_1h_mm"])

    # Events
    if Path(_EVENTS_PATH).exists() and not force:
        logger.info("Loading cached events from %s", _EVENTS_PATH)
        events = pd.read_parquet(_EVENTS_PATH)
    else:
        events = identify_storm_events(daily, reports=reports)
        events = add_hourly_peak(events, hourly if len(hourly) else None)
        events.to_parquet(_EVENTS_PATH, index=False)
        logger.info("Saved events to %s", _EVENTS_PATH)

    return daily, events
