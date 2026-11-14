"""
utils.py — Shared helper utilities for reprojection, raster sampling, I/O, etc.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(
    url: str,
    dest: Path,
    force: bool = False,
    chunk_size: int = 1 << 20,
    desc: str | None = None,
    timeout: int = 120,
    retries: int = 3,
) -> Path:
    """
    Download *url* to *dest*, skipping if the file already exists and *force* is False.

    Uses streaming download with a tqdm progress bar.  Retries up to *retries*
    times on transient errors.

    Parameters
    ----------
    url:        Remote URL to fetch.
    dest:       Local path to write.
    force:      Re-download even if *dest* already exists.
    chunk_size: Streaming chunk size in bytes (default 1 MiB).
    desc:       Label shown in the progress bar (defaults to filename).
    timeout:    Per-request timeout in seconds.
    retries:    Number of retry attempts on failure.

    Returns
    -------
    Path of the downloaded file.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        logger.debug("Cache hit — skipping download of %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None

            with (
                open(dest, "wb") as fh,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=label,
                    leave=False,
                ) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    pbar.update(len(chunk))

            logger.info("Downloaded %s → %s", url, dest)
            return dest

        except Exception as exc:  # noqa: BLE001
            logger.warning("Download attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)

    return dest  # unreachable but satisfies type checkers


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

def bbox_to_polygon(lat_min: float, lon_min: float, lat_max: float, lon_max: float):
    """
    Return a Shapely Polygon from WGS-84 bounding-box coordinates.

    Parameters
    ----------
    lat_min, lon_min, lat_max, lon_max: Bounding box in decimal degrees.

    Returns
    -------
    shapely.geometry.Polygon
    """
    from shapely.geometry import box as shapely_box

    return shapely_box(lon_min, lat_min, lon_max, lat_max)


def bbox_to_wkt(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> str:
    """
    Return a WKT POLYGON string from WGS-84 bounding-box coordinates.

    Useful for Socrata spatial queries.
    """
    return (
        f"POLYGON (({lon_min} {lat_min}, {lon_max} {lat_min}, "
        f"{lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"
    )


def bbox_to_utm(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    crs_metric: str = "EPSG:32618",
) -> Tuple[float, float, float, float]:
    """
    Reproject a WGS-84 bounding box to a metric CRS.

    Returns
    -------
    (x_min, y_min, x_max, y_max) in the target CRS units.
    """
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_metric, always_xy=True)
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------

def sample_raster_at_points(
    raster_path: Path,
    lons: Iterable[float],
    lats: Iterable[float],
    band: int = 1,
    nodata_fill: float = np.nan,
) -> np.ndarray:
    """
    Sample a raster at an array of (lon, lat) WGS-84 points.

    Uses nearest-neighbour lookup (no interpolation) which is appropriate for
    both continuous and categorical rasters.

    Parameters
    ----------
    raster_path:  Path to any rasterio-readable file.
    lons:         Sequence of longitudes (WGS-84).
    lats:         Sequence of latitudes (WGS-84).
    band:         Band index (1-based).
    nodata_fill:  Value to substitute when the sample falls on nodata.

    Returns
    -------
    numpy array of sampled values, length == len(lons).
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import rowcol
    import pyproj

    lons_arr = np.asarray(lons, dtype=float)
    lats_arr = np.asarray(lats, dtype=float)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs.to_epsg()
        # Transform points to the raster CRS if needed
        if raster_crs != 4326:
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326", f"EPSG:{raster_crs}", always_xy=True
            )
            xs, ys = transformer.transform(lons_arr, lats_arr)
        else:
            xs, ys = lons_arr, lats_arr

        rows, cols = rowcol(src.transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        height, width = src.height, src.width
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

        data = src.read(band)
        nd = src.nodata

        result = np.full(len(lons_arr), nodata_fill, dtype=float)
        result[valid] = data[rows[valid], cols[valid]].astype(float)

        if nd is not None:
            result[result == nd] = nodata_fill

    return result


def reproject_raster(
    src_path: Path,
    dst_path: Path,
    dst_crs: str,
    res_m: float | None = None,
    resampling_name: str = "bilinear",
    force: bool = False,
) -> Path:
    """
    Reproject (and optionally resample) a raster to *dst_crs*.

    Parameters
    ----------
    src_path:        Input raster path.
    dst_path:        Output raster path.
    dst_crs:         Target CRS string (e.g. 'EPSG:32618').
    res_m:           Target pixel size in destination CRS units.  If None,
                     the native resolution is preserved.
    resampling_name: Rasterio resampling algorithm name.
    force:           Overwrite even if *dst_path* exists.

    Returns
    -------
    Path of the reprojected raster.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    dst_path = Path(dst_path)
    if dst_path.exists() and not force:
        logger.debug("Cache hit — skipping reproject of %s", dst_path.name)
        return dst_path

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    resampling = Resampling[resampling_name]

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=res_m,
        )
        kwargs = src.meta.copy()
        kwargs.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            compress="lzw",
        )

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )

    logger.info("Reprojected %s → %s (%s)", src_path.name, dst_path.name, dst_crs)
    return dst_path


def clip_raster_to_bbox(
    src_path: Path,
    dst_path: Path,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    force: bool = False,
) -> Path:
    """
    Clip a raster to a WGS-84 bounding box.

    If the raster is not in WGS-84 the bounding box is reprojected first.

    Parameters
    ----------
    src_path:               Input raster.
    dst_path:               Output clipped raster.
    lat_min/lon_min/…:      Bounding box in WGS-84 degrees.
    force:                  Overwrite even if *dst_path* exists.

    Returns
    -------
    Path of the clipped raster.
    """
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    from shapely.geometry import mapping

    dst_path = Path(dst_path)
    if dst_path.exists() and not force:
        logger.debug("Cache hit — skipping clip of %s", dst_path.name)
        return dst_path

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    geom = bbox_to_polygon(lat_min, lon_min, lat_max, lon_max)

    with rasterio.open(src_path) as src:
        # Transform the clip polygon to the raster CRS
        import geopandas as gpd

        clip_gdf = gpd.GeoDataFrame(
            geometry=[geom], crs="EPSG:4326"
        ).to_crs(src.crs)
        shapes = [mapping(g) for g in clip_gdf.geometry]

        out_image, out_transform = rasterio_mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            compress="lzw",
        )

        with rasterio.open(dst_path, "w", **out_meta) as dst:
            dst.write(out_image)

    logger.info("Clipped %s → %s", src_path.name, dst_path.name)
    return dst_path


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def impute_dataframe(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing values in a DataFrame using the specified strategy.

    Numeric columns are imputed with column median (or mean).
    Non-numeric columns are left unchanged.

    Parameters
    ----------
    df:       Input DataFrame (possibly containing NaNs).
    strategy: 'median' or 'mean'.

    Returns
    -------
    DataFrame with no NaN values in numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    result = df.copy()
    for col in numeric_cols:
        if result[col].isna().any():
            fill_val = (
                result[col].median() if strategy == "median" else result[col].mean()
            )
            if np.isnan(fill_val):
                fill_val = 0.0
            result[col] = result[col].fillna(fill_val)
            logger.debug("Imputed column '%s' with %s=%.4f", col, strategy, fill_val)
    return result


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """
    Configure root logger to write to stdout and optionally to *log_file*.

    Parameters
    ----------
    log_file: Optional path for a file handler.  Parent directories are
              created automatically.
    level:    Logging level (default INFO).
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
