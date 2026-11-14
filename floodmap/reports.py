"""
reports.py — Download and clean NYC 311 flood-complaint reports via Socrata API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from floodmap.config import (
    DATA_RAW,
    DEFAULT_BBOX,
    FLOOD_DESCRIPTORS,
    GRID_RES_M,
    SOCRATA_APP_TOKEN,
    SOCRATA_DATASET,
    SOCRATA_DOMAIN,
)

logger = logging.getLogger(__name__)

_REPORTS_PATH = DATA_RAW / "reports_311.parquet"


def _socrata_client():
    """
    Return an authenticated (or anonymous) Socrata client.

    Uses the NYC_APP_TOKEN environment variable if set.
    """
    from sodapy import Socrata

    token = SOCRATA_APP_TOKEN or None
    return Socrata(SOCRATA_DOMAIN, token, timeout=60)


def _descriptor_filter() -> str:
    """
    Build a Socrata SoQL WHERE clause that matches all FLOOD_DESCRIPTORS.

    Returns
    -------
    SQL fragment suitable for the ``where`` parameter of ``client.get()``.
    """
    quoted = [f"'{d}'" for d in FLOOD_DESCRIPTORS]
    return f"descriptor in ({', '.join(quoted)})"


def download_reports(
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    force: bool = False,
    start_year: int = 2016,
) -> pd.DataFrame:
    """
    Download all 311 flood reports for the bounding box from *start_year* to present.

    Uses paginated Socrata calls (offset/limit) so the full history is fetched
    regardless of size.  Results are cached to ``data/raw/reports_311.parquet``.

    Parameters
    ----------
    bbox:       (lat_min, lon_min, lat_max, lon_max) in WGS-84.
    force:      Re-download even if the cache file already exists.
    start_year: First year to include (inclusive).

    Returns
    -------
    DataFrame with columns: unique_key, created_date, descriptor, latitude,
    longitude, incident_zip, borough, status.
    """
    out_path = Path(_REPORTS_PATH)
    if out_path.exists() and not force:
        logger.info("Loading cached 311 reports from %s", out_path)
        return pd.read_parquet(out_path)

    lat_min, lon_min, lat_max, lon_max = bbox
    logger.info(
        "Downloading 311 flood reports for bbox (%.4f,%.4f)→(%.4f,%.4f) from %d",
        lat_min, lon_min, lat_max, lon_max, start_year,
    )

    client = _socrata_client()

    # Build query components
    desc_clause = _descriptor_filter()
    date_clause = f"created_date >= '{start_year}-01-01T00:00:00'"
    lat_clause = f"latitude >= '{lat_min}' AND latitude <= '{lat_max}'"
    lon_clause = f"longitude >= '{lon_min}' AND longitude <= '{lon_max}'"
    where_clause = f"{desc_clause} AND {date_clause} AND {lat_clause} AND {lon_clause}"

    limit = 50_000
    offset = 0
    frames: list[pd.DataFrame] = []

    pbar = tqdm(desc="311 flood reports (pages)", unit="page")
    while True:
        try:
            records = client.get(
                SOCRATA_DATASET,
                where=where_clause,
                select=(
                    "unique_key,created_date,descriptor,"
                    "latitude,longitude,incident_zip,borough,status"
                ),
                limit=limit,
                offset=offset,
                order="created_date ASC",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Socrata API error at offset %d: %s", offset, exc)
            break

        if not records:
            break

        frames.append(pd.DataFrame.from_records(records))
        logger.debug("Fetched %d records (offset=%d)", len(records), offset)
        pbar.update(1)
        offset += limit

        if len(records) < limit:
            break

    pbar.close()

    if not frames:
        logger.warning("No 311 records returned — returning empty DataFrame")
        df = pd.DataFrame(
            columns=[
                "unique_key", "created_date", "descriptor",
                "latitude", "longitude", "incident_zip", "borough", "status",
            ]
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return df

    df = pd.concat(frames, ignore_index=True)
    logger.info("Raw records fetched: %d", len(df))

    df = _clean_reports(df, lat_min, lon_min, lat_max, lon_max)
    logger.info("Records after cleaning: %d", len(df))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved 311 reports to %s", out_path)
    return df


def _clean_reports(
    df: pd.DataFrame,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
) -> pd.DataFrame:
    """
    Clean raw 311 records: parse types, filter bbox, filter descriptors, deduplicate.

    Parameters
    ----------
    df:                     Raw DataFrame from Socrata.
    lat_min/lon_min/…:      Bounding box in WGS-84.

    Returns
    -------
    Cleaned DataFrame.
    """
    # Parse numeric coordinates
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Parse datetime
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce", utc=True)
    df["created_date"] = df["created_date"].dt.tz_convert("America/New_York")
    df = df.dropna(subset=["created_date"])

    # Bbox filter (redundant with API filter but defensively re-applied)
    df = df[
        (df["latitude"] >= lat_min)
        & (df["latitude"] <= lat_max)
        & (df["longitude"] >= lon_min)
        & (df["longitude"] <= lon_max)
    ]

    # Descriptor filter
    df = df[df["descriptor"].isin(FLOOD_DESCRIPTORS)]

    # Assign grid cell IDs at GRID_RES_M resolution using UTM 18N
    df = _assign_cell_ids(df)

    # Deduplication: one report per (cell_id, date) — the earliest created_date wins
    df["date"] = df["created_date"].dt.date
    df = df.sort_values("created_date")
    df = df.drop_duplicates(subset=["cell_id", "date"], keep="first")
    df = df.drop(columns=["date"])

    df = df.reset_index(drop=True)
    return df


def _assign_cell_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``cell_id`` column to *df* by snapping each report to the nearest
    GRID_RES_M × GRID_RES_M cell in UTM 18N.

    Cell IDs are encoded as ``row * 100000 + col`` (simple integer key).

    Parameters
    ----------
    df: DataFrame with 'latitude' and 'longitude' columns.

    Returns
    -------
    DataFrame with an additional 'cell_id' column (int64).
    """
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    xs, ys = transformer.transform(df["longitude"].values, df["latitude"].values)

    col_ids = (xs // GRID_RES_M).astype(np.int64)
    row_ids = (ys // GRID_RES_M).astype(np.int64)
    df = df.copy()
    df["cell_id"] = row_ids * 100_000 + col_ids
    return df


def top_flood_areas(
    df: pd.DataFrame,
    top_n: int = 5,
    block_km: float = 1.0,
) -> pd.DataFrame:
    """
    Count 311 flood reports per *block_km* × *block_km* block and return the
    top-*n* blocks sorted by report count.

    Can be used to auto-select a high-density sub-area bounding box.

    Parameters
    ----------
    df:        Cleaned reports DataFrame (must have 'latitude', 'longitude').
    top_n:     Number of top blocks to return.
    block_km:  Block size in kilometres.

    Returns
    -------
    DataFrame with columns: block_lat, block_lon, count, bbox_tuple.
    """
    block_deg_lat = block_km / 111.0  # ~1 km in degrees latitude
    block_deg_lon = block_km / 85.0   # ~1 km in degrees longitude at NYC latitude

    df = df.copy()
    df["block_lat"] = (df["latitude"] // block_deg_lat) * block_deg_lat
    df["block_lon"] = (df["longitude"] // block_deg_lon) * block_deg_lon

    counts = (
        df.groupby(["block_lat", "block_lon"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    counts["bbox_tuple"] = counts.apply(
        lambda r: (
            r["block_lat"],
            r["block_lon"],
            r["block_lat"] + block_deg_lat,
            r["block_lon"] + block_deg_lon,
        ),
        axis=1,
    )
    return counts.reset_index(drop=True)


def load_reports(bbox: Tuple[float, float, float, float] = DEFAULT_BBOX) -> pd.DataFrame:
    """
    Load cached 311 reports, downloading if necessary.

    Parameters
    ----------
    bbox: Bounding box passed to :func:`download_reports` if cache is absent.

    Returns
    -------
    Cleaned reports DataFrame.
    """
    path = Path(_REPORTS_PATH)
    if path.exists():
        logger.info("Loading 311 reports from cache: %s", path)
        return pd.read_parquet(path)
    return download_reports(bbox=bbox)
