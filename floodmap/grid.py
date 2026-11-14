"""
grid.py — Generate the 100 m prediction grid over the study sub-area.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from floodmap.config import (
    CRS_METRIC,
    CRS_WGS84,
    DATA_RAW,
    DEFAULT_BBOX,
    GRID_RES_M,
)

logger = logging.getLogger(__name__)

_GRID_PARQUET_PATH = DATA_RAW / "grid.parquet"
_GRID_GPKG_PATH = DATA_RAW / "grid.gpkg"


def build_grid(
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    res_m: float = GRID_RES_M,
    force: bool = False,
) -> pd.DataFrame:
    """
    Create a regular 100 m grid covering *bbox* in UTM 18N.

    Each row represents one grid cell and stores:
        * ``cell_id``  — unique sequential integer (int64)
        * ``lat``      — WGS-84 latitude of cell centroid (float64)
        * ``lon``      — WGS-84 longitude of cell centroid (float64)
        * ``utm_x``    — easting of centroid in UTM 18N (metres)
        * ``utm_y``    — northing of centroid in UTM 18N (metres)
        * ``geometry`` — Shapely polygon (UTM 18N) — stored in the GeoPackage

    The DataFrame (without geometry) is saved to ``data/raw/grid.parquet``.
    The GeoDataFrame (with geometry) is saved to ``data/raw/grid.gpkg``.

    Parameters
    ----------
    bbox:  (lat_min, lon_min, lat_max, lon_max) in WGS-84 degrees.
    res_m: Cell edge length in metres (default 100).
    force: Regenerate even if cached files exist.

    Returns
    -------
    DataFrame with cell_id, lat, lon, utm_x, utm_y columns.
    """
    parquet_path = Path(_GRID_PARQUET_PATH)
    gpkg_path = Path(_GRID_GPKG_PATH)

    if parquet_path.exists() and not force:
        logger.info("Loading cached grid from %s", parquet_path)
        return pd.read_parquet(parquet_path)

    import pyproj
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    lat_min, lon_min, lat_max, lon_max = bbox

    # Transform bbox corners to UTM 18N
    transformer_fwd = pyproj.Transformer.from_crs(
        CRS_WGS84, CRS_METRIC, always_xy=True
    )
    transformer_inv = pyproj.Transformer.from_crs(
        CRS_METRIC, CRS_WGS84, always_xy=True
    )

    x_min, y_min = transformer_fwd.transform(lon_min, lat_min)
    x_max, y_max = transformer_fwd.transform(lon_max, lat_max)

    # Snap to grid
    x_min = np.floor(x_min / res_m) * res_m
    y_min = np.floor(y_min / res_m) * res_m
    x_max = np.ceil(x_max / res_m) * res_m
    y_max = np.ceil(y_max / res_m) * res_m

    xs = np.arange(x_min, x_max, res_m)
    ys = np.arange(y_min, y_max, res_m)

    logger.info(
        "Grid: %d columns × %d rows = %d cells at %.0f m",
        len(xs), len(ys), len(xs) * len(ys), res_m,
    )

    # Build centroid arrays
    cx = xs + res_m / 2.0
    cy = ys + res_m / 2.0
    CX, CY = np.meshgrid(cx, cy)
    CX_flat = CX.ravel()
    CY_flat = CY.ravel()

    # Back-project centroids to WGS-84
    lons, lats = transformer_inv.transform(CX_flat, CY_flat)

    cell_ids = np.arange(len(CX_flat), dtype=np.int64)

    # Build polygon geometries for GeoPackage
    polys = [
        shapely_box(x, y, x + res_m, y + res_m)
        for x, y in zip(CX_flat - res_m / 2.0, CY_flat - res_m / 2.0)
    ]

    df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "lat": lats,
            "lon": lons,
            "utm_x": CX_flat,
            "utm_y": CY_flat,
        }
    )

    # Save parquet (no geometry)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    logger.info("Grid parquet saved to %s (%d cells)", parquet_path, len(df))

    # Save GeoPackage (with geometry)
    gdf = gpd.GeoDataFrame(df, geometry=polys, crs=CRS_METRIC)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(gpkg_path, driver="GPKG", layer="grid")
    logger.info("Grid GeoPackage saved to %s", gpkg_path)

    return df


def assign_spatial_blocks(
    grid: pd.DataFrame,
    block_size_m: float = 1000.0,
    res_m: float = GRID_RES_M,
) -> pd.DataFrame:
    """
    Assign a ``block_id`` to each grid cell for spatial cross-validation.

    Cells are grouped into *block_size_m* × *block_size_m* blocks (default 1 km²).
    Block IDs are sequential integers.

    Parameters
    ----------
    grid:         Grid DataFrame with 'utm_x', 'utm_y' columns.
    block_size_m: Block edge length in metres.
    res_m:        Grid cell resolution (used for block ID encoding).

    Returns
    -------
    Grid DataFrame with an additional 'block_id' column (int64).
    """
    grid = grid.copy()
    block_col = (grid["utm_x"] // block_size_m).astype(np.int64)
    block_row = (grid["utm_y"] // block_size_m).astype(np.int64)
    # Encode as a unique integer
    block_col_range = block_col.max() - block_col.min() + 1
    grid["block_id"] = (block_row - block_row.min()) * block_col_range + (
        block_col - block_col.min()
    )
    return grid


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_grid(
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build (or load cached) grid and return it with block_id assignments.

    Parameters
    ----------
    bbox:  (lat_min, lon_min, lat_max, lon_max).
    force: Regenerate even if cached.

    Returns
    -------
    DataFrame with cell_id, lat, lon, utm_x, utm_y, block_id.
    """
    grid = build_grid(bbox=bbox, force=force)
    grid = assign_spatial_blocks(grid)
    return grid
