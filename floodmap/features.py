"""
features.py — Join all data sources into a (grid cell × storm event) feature table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from floodmap.config import (
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_BBOX,
    FLOOD_LABEL_BUFFER_H,
    GRID_RES_M,
    MAX_FLOOD_FRAC,
    MIN_FLOOD_FRAC,
)
from floodmap.dem import TERRAIN_BAND_NAMES
from floodmap.landcover import LC_BAND_NAMES
from floodmap.utils import impute_dataframe, sample_raster_at_points

logger = logging.getLogger(__name__)

_FEATURES_RAW_PATH = DATA_PROCESSED / "features_raw.parquet"


# ---------------------------------------------------------------------------
# Raster sampling helpers
# ---------------------------------------------------------------------------

def _sample_terrain(
    grid: pd.DataFrame,
    terrain_path: Path,
) -> pd.DataFrame:
    """
    Sample the multi-band terrain raster at each grid cell centroid.

    Parameters
    ----------
    grid:         DataFrame with 'lat' and 'lon' columns.
    terrain_path: Path to multi-band terrain GeoTIFF.

    Returns
    -------
    DataFrame with one terrain feature column per band (names from TERRAIN_BAND_NAMES).
    """
    import rasterio

    logger.info("Sampling terrain features from %s", terrain_path)
    with rasterio.open(terrain_path) as src:
        n_bands = src.count

    result = grid[["cell_id"]].copy()
    for band_idx in range(1, n_bands + 1):
        name = TERRAIN_BAND_NAMES[band_idx - 1] if band_idx <= len(TERRAIN_BAND_NAMES) else f"band_{band_idx}"
        values = sample_raster_at_points(
            terrain_path, grid["lon"].values, grid["lat"].values, band=band_idx
        )
        result[name] = values

    return result


def _sample_landcover(
    grid: pd.DataFrame,
    lc_path: Path,
    dist_water_path: Path,
) -> pd.DataFrame:
    """
    Sample the multi-band landcover raster and distance-to-water raster.

    Parameters
    ----------
    grid:           DataFrame with 'lat' and 'lon' columns.
    lc_path:        Multi-band landcover fraction raster.
    dist_water_path: Single-band distance-to-water raster.

    Returns
    -------
    DataFrame with landcover feature columns.
    """
    import rasterio

    logger.info("Sampling landcover features from %s", lc_path)
    with rasterio.open(lc_path) as src:
        n_bands = src.count

    result = grid[["cell_id"]].copy()
    for band_idx in range(1, n_bands + 1):
        name = LC_BAND_NAMES[band_idx - 1] if band_idx <= len(LC_BAND_NAMES) else f"lc_band_{band_idx}"
        values = sample_raster_at_points(
            lc_path, grid["lon"].values, grid["lat"].values, band=band_idx
        )
        result[name] = values

    dist_values = sample_raster_at_points(
        dist_water_path, grid["lon"].values, grid["lat"].values, band=1
    )
    result["dist_water"] = dist_values
    return result


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def _assign_flood_labels(
    cell_event_df: pd.DataFrame,
    reports: pd.DataFrame,
    grid: pd.DataFrame,
    buffer_h: int = FLOOD_LABEL_BUFFER_H,
) -> pd.Series:
    """
    Assign binary flood labels to each (cell_id, event_id) row.

    A cell is labelled *flooded* (1) if at least one deduplicated 311 report
    falls within that cell during ``[event_start − buffer_h hours,
    event_end + buffer_h hours]``.

    Parameters
    ----------
    cell_event_df: Cross-joined (cell × event) DataFrame with columns
                   cell_id, event_id, start_date, end_date.
    reports:       Cleaned 311 reports with 'cell_id' and 'created_date'.
    grid:          Grid DataFrame (unused here, kept for API consistency).
    buffer_h:      Hours to extend the event window on each side.

    Returns
    -------
    Series of int8 flood labels (0/1) aligned with *cell_event_df*.
    """
    if reports is None or len(reports) == 0:
        logger.warning("No flood reports available — all labels will be 0")
        return pd.Series(np.zeros(len(cell_event_df), dtype=np.int8), index=cell_event_df.index)

    # Re-snap reports to grid cell IDs using the actual grid geometry.
    # The cell_id stored in reports was computed with a different origin than
    # the sequential cell_ids in the grid — recompute the mapping here.
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    rep_x, rep_y = transformer.transform(
        reports["longitude"].values, reports["latitude"].values
    )

    # Grid cell corners: utm_x/utm_y are centroids; cell spans ±res/2
    res = GRID_RES_M
    grid_x = grid["utm_x"].values  # shape (n_cells,)
    grid_y = grid["utm_y"].values
    grid_ids = grid["cell_id"].values

    # For each report, find the grid cell whose centroid is nearest (within 1 cell)
    # Use integer bin arithmetic: col = floor((rep_x - x_min + res/2) / res), etc.
    x_min = grid_x.min() - res / 2  # left edge of leftmost cell
    y_min = grid_y.min() - res / 2
    ncols = int(round((grid_x.max() - grid_x.min()) / res)) + 1

    rep_col = np.floor((rep_x - x_min) / res).astype(np.int64)
    rep_row = np.floor((rep_y - y_min) / res).astype(np.int64)

    # Build a fast (row, col) → cell_id lookup from the grid
    grid_col = np.round((grid_x - grid_x.min()) / res).astype(np.int64)
    grid_row = np.round((grid_y - grid_y.min()) / res).astype(np.int64)
    rc_to_cid: dict[tuple, int] = {
        (int(r), int(c)): int(cid)
        for r, c, cid in zip(grid_row, grid_col, grid_ids)
    }

    snapped_cids = np.array(
        [rc_to_cid.get((int(r), int(c)), -1) for r, c in zip(rep_row, rep_col)],
        dtype=np.int64,
    )
    n_matched = int((snapped_cids >= 0).sum())
    logger.info(
        "Snapped %d / %d reports to grid cells", n_matched, len(reports)
    )

    reports_copy = reports.copy()
    reports_copy["cell_id"] = snapped_cids
    reports_copy = reports_copy[reports_copy["cell_id"] >= 0]

    # Normalise timestamps to timezone-naive UTC datetime64 for fast numpy comparisons.
    if hasattr(reports_copy["created_date"].dtype, "tz") and reports_copy["created_date"].dt.tz is not None:
        reports_copy["created_date"] = reports_copy["created_date"].dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        reports_copy["created_date"] = pd.to_datetime(reports_copy["created_date"]).dt.tz_localize(None)

    cell_times: dict[int, np.ndarray] = {}
    for cid, grp in reports_copy.groupby("cell_id"):
        cell_times[int(cid)] = grp["created_date"].values.astype("datetime64[ns]")

    buffer_td = np.timedelta64(buffer_h * 3_600, "s")

    labels = np.zeros(len(cell_event_df), dtype=np.int8)

    for i, row in enumerate(
        tqdm(
            cell_event_df.itertuples(index=False),
            total=len(cell_event_df),
            desc="Flood labels",
            leave=False,
        )
    ):
        cid = int(row.cell_id)
        if cid not in cell_times:
            continue

        # Window: [event_start − buffer, event_end + 1 day + buffer] in datetime64[ns]
        start_ts = np.datetime64(pd.Timestamp(row.start_date), "ns") - buffer_td
        end_ts = np.datetime64(pd.Timestamp(row.end_date) + pd.Timedelta(days=1), "ns") + buffer_td

        times = cell_times[cid]
        if np.any((times >= start_ts) & (times <= end_ts)):
            labels[i] = 1

    return pd.Series(labels, index=cell_event_df.index, dtype=np.int8)


# ---------------------------------------------------------------------------
# Main feature join
# ---------------------------------------------------------------------------

def build_feature_table(
    grid: pd.DataFrame,
    events: pd.DataFrame,
    terrain_path: Path,
    lc_path: Path,
    dist_water_path: Path,
    reports: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build the full (n_cells × n_events) feature DataFrame.

    Steps:
        1. Sample terrain features at each cell centroid.
        2. Sample landcover fractions at each cell centroid.
        3. Cross-join cells × events.
        4. Attach event rain features.
        5. Assign flood labels.
        6. Impute any remaining NaNs.
        7. Validate class balance.

    Parameters
    ----------
    grid:           Grid DataFrame (cell_id, lat, lon, utm_x, utm_y, block_id).
    events:         Storm events DataFrame.
    terrain_path:   Multi-band terrain feature raster path.
    lc_path:        Multi-band landcover fraction raster path.
    dist_water_path: Single-band distance-to-water raster path.
    reports:        Cleaned 311 flood reports.
    force:          Recompute even if cached.

    Returns
    -------
    Feature DataFrame saved to ``data/processed/features_raw.parquet``.
    """
    out_path = Path(_FEATURES_RAW_PATH)
    if out_path.exists() and not force:
        logger.info("Loading cached features from %s", out_path)
        return pd.read_parquet(out_path)

    logger.info("Building feature table (%d cells × %d events)", len(grid), len(events))

    # --- Static features ---
    terrain_df = _sample_terrain(grid, terrain_path)
    lc_df = _sample_landcover(grid, lc_path, dist_water_path)

    static_df = grid[["cell_id", "lat", "lon", "block_id"]].copy()
    static_df = static_df.merge(terrain_df, on="cell_id", how="left")
    static_df = static_df.merge(lc_df, on="cell_id", how="left")

    # --- Cross join: every (cell, event) pair ---
    if len(events) == 0:
        logger.warning(
            "No storm events found — creating a dummy event for feature schema"
        )
        events = _dummy_events()

    static_df["_key"] = 1
    events_key = events.copy()
    events_key["_key"] = 1
    cross = static_df.merge(events_key, on="_key").drop(columns=["_key"])

    logger.info("Cross-join size: %d rows (cells × events)", len(cross))

    # --- Rain features (from events table) ---
    rain_cols = ["event_id", "rain_total_mm", "rain_peak_1h_mm", "rain_prev_48h_mm"]
    available_rain = [c for c in rain_cols if c in events.columns]
    # Already joined via cross; just ensure columns exist
    if "rain_total_mm" not in cross.columns:
        cross["rain_total_mm"] = np.nan
    if "rain_peak_1h_mm" not in cross.columns:
        cross["rain_peak_1h_mm"] = np.nan
    if "rain_prev_48h_mm" not in cross.columns:
        cross["rain_prev_48h_mm"] = np.nan

    # --- Flood labels ---
    cross["flooded"] = _assign_flood_labels(cross, reports, grid)

    # --- Select and order columns per schema ---
    ordered_cols = [
        "cell_id", "lat", "lon", "event_id",
        # terrain
        "elev", "slope", "curvature", "twi", "flow_accum", "depression_depth",
        "dist_water",
        # landcover
        "impervious_frac", "landcover_built", "landcover_lowveg",
        "landcover_tree", "landcover_water",
        # rain
        "rain_total_mm", "rain_peak_1h_mm", "rain_prev_48h_mm",
        # label
        "flooded",
        # splits helper
        "block_id",
        # dates (kept for split logic)
        "start_date", "end_date",
    ]

    # Keep only columns that exist
    present = [c for c in ordered_cols if c in cross.columns]
    result = cross[present].copy()

    # --- Impute NaNs ---
    result = impute_dataframe(result)

    # Cast flooded to int8
    result["flooded"] = result["flooded"].astype(np.int8)

    # --- Class balance check ---
    flooded_frac = result["flooded"].mean()
    logger.info(
        "Class balance: %d flooded / %d total = %.3f%%",
        result["flooded"].sum(),
        len(result),
        flooded_frac * 100,
    )
    if flooded_frac < MIN_FLOOD_FRAC:
        logger.warning(
            "Very low flood fraction (%.4f < %.4f) — labels may be sparse",
            flooded_frac,
            MIN_FLOOD_FRAC,
        )
    elif flooded_frac > MAX_FLOOD_FRAC:
        logger.warning(
            "High flood fraction (%.4f > %.4f) — check label logic",
            flooded_frac,
            MAX_FLOOD_FRAC,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    logger.info("Raw features saved to %s (%d rows)", out_path, len(result))
    return result


def _dummy_events() -> pd.DataFrame:
    """Return a minimal single-row events DataFrame with zero rain."""
    return pd.DataFrame(
        {
            "event_id": ["E0000"],
            "start_date": [pd.Timestamp("2020-01-01")],
            "end_date": [pd.Timestamp("2020-01-01")],
            "rain_total_mm": [0.0],
            "rain_peak_1h_mm": [np.nan],
            "rain_prev_48h_mm": [0.0],
            "n_reports": [0],
        }
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_features(
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load all intermediate outputs and build the feature table.

    Expects the following files to already exist (run earlier pipeline steps):
        * data/raw/grid.parquet
        * data/raw/events.parquet
        * data/raw/terrain_100m.tif
        * data/raw/landcover_100m.tif
        * data/raw/dist_water_100m.tif
        * data/raw/reports_311.parquet

    Parameters
    ----------
    bbox:  Bounding box (for consistency; not used if all files exist).
    force: Recompute even if cached.

    Returns
    -------
    Feature DataFrame.
    """
    from floodmap.grid import run_grid

    grid = run_grid(bbox=bbox, force=False)

    events_path = DATA_RAW / "events.parquet"
    if events_path.exists():
        events = pd.read_parquet(events_path)
    else:
        logger.warning("events.parquet not found — running rain step first")
        from floodmap.rain import run_rain

        _, events = run_rain(bbox=bbox, force=force)

    terrain_path = DATA_RAW / "terrain_100m.tif"
    lc_path = DATA_RAW / "landcover_100m.tif"
    dist_water_path = DATA_RAW / "dist_water_100m.tif"

    if not terrain_path.exists():
        logger.warning("terrain_100m.tif not found — running DEM step")
        from floodmap.dem import run_dem

        run_dem(bbox=bbox, force=force)

    if not lc_path.exists() or not dist_water_path.exists():
        logger.warning("landcover rasters not found — running landcover step")
        from floodmap.landcover import run_landcover

        run_landcover(bbox=bbox, force=force)

    reports_path = DATA_RAW / "reports_311.parquet"
    if reports_path.exists():
        reports = pd.read_parquet(reports_path)
    else:
        logger.warning("reports_311.parquet not found — running reports step")
        from floodmap.reports import download_reports

        reports = download_reports(bbox=bbox, force=force)

    return build_feature_table(
        grid=grid,
        events=events,
        terrain_path=terrain_path,
        lc_path=lc_path,
        dist_water_path=dist_water_path,
        reports=reports,
        force=force,
    )
