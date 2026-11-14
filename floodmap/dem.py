"""
dem.py — NOAA LiDAR DEM tile discovery, download, merge, resample, and terrain features.
"""

from __future__ import annotations

import logging
import math
import tempfile
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

from floodmap.config import (
    ALLOW_SYNTHETIC,
    CRS_METRIC,
    DATA_RAW,
    DEFAULT_BBOX,
    DEM_RES_M,
    GRID_RES_M,
    NOAA_LIDAR_BUCKET,
    NOAA_LIDAR_PREFIX,
)

logger = logging.getLogger(__name__)

_DEM_5M_PATH = DATA_RAW / "dem_5m.tif"
_TERRAIN_100M_PATH = DATA_RAW / "terrain_100m.tif"

# Band names stored in the multi-band terrain raster (in order)
TERRAIN_BAND_NAMES = [
    "elev",
    "slope",
    "curvature",
    "twi",
    "flow_accum",
    "depression_depth",
]


# ---------------------------------------------------------------------------
# S3 tile discovery
# ---------------------------------------------------------------------------

def _s3_resource():
    """
    Return a boto3 S3 resource configured for anonymous (unsigned) access.

    Returns
    -------
    boto3 S3 resource object.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.resource("s3", config=Config(signature_version=UNSIGNED))


def list_lidar_tiles(
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
) -> List[str]:
    """
    List NOAA LiDAR S3 object keys whose filenames overlap *bbox*.

    The NYC topobathy 2017 tiles are 1-degree × 1-degree GeoTIFFs named with
    their south-west corner (e.g. ``n40w074_...tif``).  This function lists
    all keys under the prefix and filters by the bounding-box extents.

    Parameters
    ----------
    bbox: (lat_min, lon_min, lat_max, lon_max) in WGS-84.

    Returns
    -------
    List of S3 object keys that could overlap the bounding box.
    """
    lat_min, lon_min, lat_max, lon_max = bbox

    try:
        s3 = _s3_resource()
        bucket = s3.Bucket(NOAA_LIDAR_BUCKET)
        all_keys = [
            obj.key
            for obj in bucket.objects.filter(Prefix=NOAA_LIDAR_PREFIX)
            if obj.key.endswith(".tif") or obj.key.endswith(".img")
        ]
        logger.info("Found %d tiles under %s", len(all_keys), NOAA_LIDAR_PREFIX)
    except Exception as exc:  # noqa: BLE001
        logger.warning("S3 listing failed: %s — using fallback tile list", exc)
        all_keys = _fallback_tile_keys(lat_min, lon_min, lat_max, lon_max)

    # Filter by spatial overlap with bbox
    matching = []
    for key in all_keys:
        fname = Path(key).name.lower()
        tile_lat, tile_lon = _parse_tile_coords(fname)
        if tile_lat is None:
            # Name carries no coordinates (e.g. be_NYC_001.tif): read the
            # georeferencing from the remote file header instead.
            if _remote_tile_overlaps_bbox(key, bbox) is not False:
                matching.append(key)
            continue
        # Tile covers 1° square starting at (tile_lat, tile_lon)
        if (
            tile_lat <= lat_max
            and tile_lat + 1 >= lat_min
            and tile_lon <= lon_max
            and tile_lon + 1 >= lon_min
        ):
            matching.append(key)

    logger.info("Tiles intersecting bbox: %d", len(matching))
    return matching


def _remote_tile_overlaps_bbox(
    key: str,
    bbox: Tuple[float, float, float, float],
) -> bool | None:
    """
    Check whether the S3 raster *key* overlaps *bbox* by reading only its header.

    Parameters
    ----------
    key:  S3 object key inside the NOAA LiDAR bucket.
    bbox: (lat_min, lon_min, lat_max, lon_max) in WGS-84.

    Returns
    -------
    True/False for overlap, or None if the header could not be read (the caller
    then keeps the tile to be safe).
    """
    import rasterio
    from rasterio.coords import disjoint_bounds
    from rasterio.warp import transform_bounds

    lat_min, lon_min, lat_max, lon_max = bbox
    url = f"/vsicurl/https://{NOAA_LIDAR_BUCKET}.s3.amazonaws.com/{key}"
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(url) as src:
                west, south, east, north = transform_bounds(
                    "EPSG:4326", src.crs, lon_min, lat_min, lon_max, lat_max
                )
                return not disjoint_bounds(src.bounds, (west, south, east, north))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read header of %s (%s) — keeping tile", key, exc)
        return None


def _parse_tile_coords(filename: str) -> Tuple[float | None, float | None]:
    """
    Parse the SW corner latitude/longitude from a NOAA tile filename.

    Expected pattern: ``n<lat>w<lon>`` or ``s<lat>e<lon>`` etc.

    Parameters
    ----------
    filename: Tile filename (lowercase).

    Returns
    -------
    (lat, lon) floats or (None, None) if parsing fails.
    """
    import re

    match = re.search(r"([ns])(\d+)([ew])(\d+)", filename)
    if not match:
        return None, None
    lat = float(match.group(2)) * (1 if match.group(1) == "n" else -1)
    lon = float(match.group(4)) * (-1 if match.group(3) == "w" else 1)
    return lat, lon


def _fallback_tile_keys(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
) -> List[str]:
    """
    Generate plausible S3 key patterns for the NYC topobathy dataset when
    automatic listing fails.

    Parameters
    ----------
    lat_min/lon_min/lat_max/lon_max: Bounding box.

    Returns
    -------
    List of candidate S3 keys.
    """
    keys = []
    for lat in range(int(lat_min), int(lat_max) + 1):
        for lon_neg in range(int(-lon_max), int(-lon_min) + 1):
            fname = f"n{lat:02d}w{lon_neg:03d}_NYC_topobathy_hydroEnforced_DEM_2017.tif"
            keys.append(f"{NOAA_LIDAR_PREFIX}{fname}")
    return keys


# ---------------------------------------------------------------------------
# Download & merge
# ---------------------------------------------------------------------------

def download_tiles(
    tile_keys: List[str],
    force: bool = False,
) -> List[Path]:
    """
    Download S3 tiles to ``data/raw/lidar/`` and return local paths.

    Parameters
    ----------
    tile_keys: S3 object keys to download.
    force:     Re-download even if cached.

    Returns
    -------
    List of paths for successfully downloaded tiles.
    """
    from tqdm import tqdm

    out_dir = DATA_RAW / "lidar"
    out_dir.mkdir(parents=True, exist_ok=True)

    s3 = _s3_resource()
    local_paths = []

    for key in tqdm(tile_keys, desc="LiDAR tiles", unit="tile"):
        dest = out_dir / Path(key).name
        if dest.exists() and not force:
            logger.debug("Cache hit: %s", dest.name)
            local_paths.append(dest)
            continue
        try:
            s3.Bucket(NOAA_LIDAR_BUCKET).download_file(key, str(dest))
            logger.info("Downloaded %s → %s", key, dest)
            local_paths.append(dest)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to download %s: %s", key, exc)

    return local_paths


def merge_and_reproject_tiles(
    tile_paths: List[Path],
    bbox: Tuple[float, float, float, float],
    out_path: Path,
    res_m: float = DEM_RES_M,
    force: bool = False,
) -> Path:
    """
    Mosaic LiDAR tiles onto one UTM 18N grid at *res_m*, in metres.

    Each tile is warped straight onto a shared destination grid whose extent is
    the bbox snapped outward to a multiple of ``GRID_RES_M`` -- the same snapping
    ``grid.build_grid`` uses -- so aggregating the 5 m DEM by an integer factor
    yields 100 m cells that coincide exactly with the prediction grid.  Peak RAM
    is one destination array (not the 1 ft source tiles).  Where tiles overlap,
    the first tile with a valid value wins.  Source elevations are converted to
    metres using the tile CRS's linear unit (US survey feet for the NYC LiDAR).

    ``rasterio.merge`` is deliberately not used: with a float32 nodata value that
    is not exactly representable (-3.4028230607370965e+38) it silently returns an
    all-zero mosaic.

    Parameters
    ----------
    tile_paths: Local raster file paths.
    bbox:       (lat_min, lon_min, lat_max, lon_max) extent.
    out_path:   Output GeoTIFF path.
    res_m:      Target pixel size in metres.
    force:      Overwrite existing output.

    Returns
    -------
    Path of the mosaicked DEM (float32, NaN = no data, metres).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject as rio_reproject

    from floodmap.utils import bbox_to_utm

    out_path = Path(out_path)
    if out_path.exists() and not force:
        logger.debug("Cache hit — skipping DEM merge/reproject: %s", out_path.name)
        return out_path

    if not tile_paths:
        if ALLOW_SYNTHETIC:
            logger.warning("No LiDAR tiles — generating SYNTHETIC DEM (FLOODMAP_ALLOW_SYNTHETIC=1)")
            return _generate_synthetic_dem(bbox, out_path, res_m)
        raise RuntimeError(
            "No LiDAR tiles were downloaded. Check your network / the S3 bucket, "
            "or set FLOODMAP_ALLOW_SYNTHETIC=1 for a fake DEM (tests/demos only)."
        )

    lat_min, lon_min, lat_max, lon_max = bbox
    x_min, y_min, x_max, y_max = bbox_to_utm(lat_min, lon_min, lat_max, lon_max, CRS_METRIC)
    x0 = math.floor(x_min / GRID_RES_M) * GRID_RES_M
    y0 = math.floor(y_min / GRID_RES_M) * GRID_RES_M
    x1 = math.ceil(x_max / GRID_RES_M) * GRID_RES_M
    y1 = math.ceil(y_max / GRID_RES_M) * GRID_RES_M
    ncols = int(round((x1 - x0) / res_m))
    nrows = int(round((y1 - y0) / res_m))
    dst_transform = from_origin(x0, y1, res_m, res_m)

    mosaic = np.full((nrows, ncols), np.nan, dtype=np.float32)
    n_used = 0
    for tp in tile_paths:
        try:
            with rasterio.open(tp) as src:
                nodata = src.nodata
                # Linear unit -> metres (vertical unit equals horizontal unit for this product)
                unit_factor = 1.0
                try:
                    unit_factor = float(src.crs.linear_units_factor[1])
                except Exception:  # noqa: BLE001
                    logger.warning("Unknown CRS unit for %s; assuming metres", tp.name)
                tile = np.full((nrows, ncols), np.nan, dtype=np.float32)
                rio_reproject(
                    source=rasterio.band(src, 1),
                    destination=tile,
                    src_nodata=None if nodata is None else float(np.float32(nodata)),
                    dst_transform=dst_transform,
                    dst_crs=CRS_METRIC,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping tile %s: %s", tp.name, exc)
            continue
        tile = tile * np.float32(unit_factor)
        take = np.isnan(mosaic) & np.isfinite(tile)
        mosaic[take] = tile[take]
        n_used += 1
        logger.info("Tile %s covered %d px (unit factor %.7f)", tp.name, int(take.sum()), unit_factor)

    valid = np.isfinite(mosaic)
    if n_used == 0 or not valid.any():
        if ALLOW_SYNTHETIC:
            logger.warning("No tile overlaps bbox — generating SYNTHETIC DEM (FLOODMAP_ALLOW_SYNTHETIC=1)")
            return _generate_synthetic_dem(bbox, out_path, res_m)
        raise RuntimeError(
            f"None of the LiDAR tiles contain valid data inside the bounding box {bbox}. "
            "See the warnings above for per-tile errors."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=nrows, width=ncols, count=1,
        dtype="float32", crs=CRS_METRIC, transform=dst_transform,
        nodata=float("nan"), compress="lzw",
    ) as dst:
        dst.write(mosaic, 1)

    logger.info(
        "DEM mosaic -> %s (%d×%d px @ %.0f m, %.1f%% valid, elev %.1f..%.1f m)",
        out_path, ncols, nrows, res_m, 100 * valid.mean(),
        float(np.nanmin(mosaic)), float(np.nanmax(mosaic)),
    )
    return out_path


def _generate_synthetic_dem(
    bbox: Tuple[float, float, float, float],
    out_path: Path,
    res_m: float,
) -> Path:
    """
    Generate a plausible synthetic DEM for the sub-area when LiDAR tiles are
    unavailable (e.g. in CI or offline environments).

    The synthetic DEM uses a mild south-to-north elevation gradient with
    additive Gaussian noise, mimicking flat coastal terrain.

    Parameters
    ----------
    bbox:     (lat_min, lon_min, lat_max, lon_max).
    out_path: Output GeoTIFF path.
    res_m:    Pixel size in metres.

    Returns
    -------
    Path of the generated raster.
    """
    import rasterio
    from rasterio.transform import from_origin
    from floodmap.utils import bbox_to_utm

    lat_min, lon_min, lat_max, lon_max = bbox
    x_min, y_min, x_max, y_max = bbox_to_utm(lat_min, lon_min, lat_max, lon_max, CRS_METRIC)

    ncols = max(10, int((x_max - x_min) / res_m))
    nrows = max(10, int((y_max - y_min) / res_m))

    rng = np.random.default_rng(42)
    # Mild S→N gradient (0–10 m) + noise (σ = 1 m)
    gradient = np.linspace(0.0, 10.0, nrows)[:, None] * np.ones((1, ncols))
    noise = rng.normal(0, 1.0, size=(nrows, ncols))
    dem = (gradient + noise).astype(np.float32)

    transform = from_origin(x_min, y_max, res_m, res_m)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=np.float32,
        crs=CRS_METRIC,
        transform=transform,
        nodata=-9999,
        compress="lzw",
    ) as dst:
        dst.write(dem[np.newaxis, :, :])

    logger.warning("Synthetic DEM written to %s (%d×%d)", out_path, ncols, nrows)
    return out_path


# ---------------------------------------------------------------------------
# Terrain feature computation
# ---------------------------------------------------------------------------

def compute_terrain_features(
    dem_path: Path,
    out_100m_path: Path,
    res_coarse_m: float = GRID_RES_M,
    force: bool = False,
) -> Path:
    """
    Compute terrain features from a DEM and save a multi-band 100m raster.

    Features computed:
        Band 1 — elevation (mean)
        Band 2 — slope (degrees)
        Band 3 — curvature (planform, 1/m)
        Band 4 — TWI = ln(flow_accum / tan(slope))
        Band 5 — flow accumulation (cells)
        Band 6 — depression depth (m) = filled DEM − original DEM

    Uses ``richdem`` for slope, curvature, and TWI; falls back to numpy
    finite-difference approximations if richdem is unavailable.

    Parameters
    ----------
    dem_path:      Path to 5m DEM GeoTIFF in UTM 18N.
    out_100m_path: Output path for multi-band 100m terrain raster.
    res_coarse_m:  Target coarse resolution (default 100 m).
    force:         Overwrite existing output.

    Returns
    -------
    Path of the terrain feature raster.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    out_100m_path = Path(out_100m_path)
    if out_100m_path.exists() and not force:
        logger.debug("Cache hit — skipping terrain features: %s", out_100m_path.name)
        return out_100m_path

    logger.info("Computing terrain features from %s", dem_path)

    with rasterio.open(dem_path) as src:
        dem_arr = src.read(1).astype(np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            dem_arr[dem_arr == np.float32(src.nodata)] = np.nan
        res = src.res[0]  # pixel size in metres (UTM)
        transform_fine = src.transform
        crs = src.crs
        nodata_val = -9999.0  # output sentinel (the 5 m DEM itself uses NaN)

    logger.info("DEM shape: %s, resolution: %.1f m", dem_arr.shape, res)

    # Replace NaN with interpolated or zero fill for processing
    nan_mask = ~np.isfinite(dem_arr)
    if nan_mask.any():
        # Simple fill: replace NaN with nearest valid or 0
        from scipy.ndimage import distance_transform_edt

        indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        dem_filled_for_proc = dem_arr.copy()
        dem_filled_for_proc[nan_mask] = dem_arr[tuple(indices[:, nan_mask])]
        dem_filled_for_proc = np.where(np.isfinite(dem_filled_for_proc), dem_filled_for_proc, 0.0)
    else:
        dem_filled_for_proc = dem_arr.copy()

    # --- Slope and curvature via richdem or numpy ---
    slope_arr, curv_arr = _compute_slope_curvature(dem_filled_for_proc, res)

    # --- Flow accumulation & TWI ---
    flow_accum_arr, twi_arr = _compute_flow_accum_twi(dem_filled_for_proc, slope_arr, res)

    # --- Depression depth ---
    dem_pit_filled = _fill_pits(dem_filled_for_proc)
    depression_arr = np.maximum(0.0, dem_pit_filled - dem_filled_for_proc).astype(np.float32)

    # Restore NaN in original no-data areas
    for arr in (slope_arr, curv_arr, twi_arr, flow_accum_arr, depression_arr):
        arr[nan_mask] = np.nan

    bands_5m = np.stack(
        [dem_arr, slope_arr, curv_arr, twi_arr, flow_accum_arr, depression_arr],
        axis=0,
    ).astype(np.float32)

    # --- Aggregate to coarse resolution ---
    bands_100m = _aggregate_to_coarse(
        bands_5m, transform_fine, crs, res, res_coarse_m
    )

    # Write output
    out_100m_path.parent.mkdir(parents=True, exist_ok=True)
    n_bands, nrows, ncols = bands_100m.shape

    scale = res_coarse_m / res
    new_transform = rasterio.transform.Affine(
        transform_fine.a * scale,
        transform_fine.b,
        transform_fine.c,
        transform_fine.d,
        transform_fine.e * scale,
        transform_fine.f,
    )

    with rasterio.open(
        out_100m_path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=n_bands,
        dtype=np.float32,
        crs=crs,
        transform=new_transform,
        nodata=nodata_val,
        compress="lzw",
    ) as dst:
        for b in range(n_bands):
            band_data = bands_100m[b]
            band_data = np.where(np.isfinite(band_data), band_data, nodata_val)
            dst.write(band_data.astype(np.float32), b + 1)
        # Write band descriptions
        for i, name in enumerate(TERRAIN_BAND_NAMES, 1):
            dst.update_tags(i, name=name)

    logger.info(
        "Terrain features saved to %s (%d bands, %d×%d px)",
        out_100m_path, n_bands, ncols, nrows,
    )
    return out_100m_path


def _compute_slope_curvature(
    dem: np.ndarray, res: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and planform curvature (1/m) from a DEM array.

    Tries richdem first; falls back to numpy finite-difference stencils.

    Parameters
    ----------
    dem: 2-D float32 elevation array (metres).
    res: Pixel size in metres.

    Returns
    -------
    (slope_degrees, curvature) arrays of the same shape as *dem*.
    """
    try:
        import richdem as rd

        rd_dem = rd.rdarray(dem.astype(np.float64), no_data=-9999)
        slope_rd = rd.TerrainAttribute(rd_dem, attrib="slope_degrees")
        curv_rd = rd.TerrainAttribute(rd_dem, attrib="curvature")
        slope = np.array(slope_rd, dtype=np.float32)
        curv = np.array(curv_rd, dtype=np.float32)
        logger.debug("Slope/curvature via richdem")
        return slope, curv

    except Exception as exc:  # noqa: BLE001
        logger.debug("richdem unavailable (%s), using numpy finite differences", exc)

    # Finite-difference fallback (Zevenbergen & Thorne, 1987)
    dz_dy, dz_dx = np.gradient(dem, res, res)
    p = dz_dx
    q = dz_dy
    slope_rad = np.arctan(np.sqrt(p ** 2 + q ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Second derivatives for curvature
    d2z_dx2 = np.gradient(dz_dx, res, axis=1)
    d2z_dy2 = np.gradient(dz_dy, res, axis=0)
    curvature = -(d2z_dx2 + d2z_dy2).astype(np.float32)
    return slope_deg, curvature


def _compute_flow_accum_twi(
    dem: np.ndarray, slope_deg: np.ndarray, res: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute flow accumulation and TWI from a DEM.

    Attempts to use pysheds; falls back to a simple D8 upslope area
    approximation implemented in numpy.

    Parameters
    ----------
    dem:       Filled DEM array (metres).
    slope_deg: Slope in degrees.
    res:       Pixel size in metres.

    Returns
    -------
    (flow_accum, twi) arrays.
    """
    try:
        from pysheds.grid import Grid
        import tempfile, rasterio
        from rasterio.transform import from_origin

        # Write DEM to temp GeoTIFF for pysheds
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=dem.shape[0],
            width=dem.shape[1],
            count=1,
            dtype=np.float64,
            crs="EPSG:32618",
            transform=from_origin(0, dem.shape[0] * res, res, res),
            nodata=-9999,
        ) as dst:
            dst.write(dem[np.newaxis].astype(np.float64))

        grid = Grid.from_raster(tmp_path)
        dem_py = grid.read_raster(tmp_path)
        pit_filled = grid.fill_pits(dem_py)
        flooded = grid.fill_depressions(pit_filled)
        inflated = grid.resolve_flats(flooded)
        fdir = grid.flowdir(inflated)
        acc = grid.accumulation(fdir)

        import os

        os.unlink(tmp_path)

        flow_accum = np.array(acc, dtype=np.float32)
        logger.debug("Flow accumulation via pysheds")

    except Exception as exc:  # noqa: BLE001
        logger.debug("pysheds unavailable (%s), using simple D8 approximation", exc)
        flow_accum = _simple_d8_accum(dem).astype(np.float32)

    # TWI = ln(A / tan(beta))
    slope_rad = np.radians(np.clip(slope_deg, 0.1, 89.9))
    tan_slope = np.tan(slope_rad)
    tan_slope = np.where(tan_slope < 1e-6, 1e-6, tan_slope)
    area = np.maximum(flow_accum, 1.0) * (res ** 2)  # specific catchment area m²
    twi = np.log(area / tan_slope).astype(np.float32)
    return flow_accum, twi


def _simple_d8_accum(dem: np.ndarray) -> np.ndarray:
    """
    Compute a simple D8 flow accumulation raster from a DEM using
    iterative upstream counting.

    Not hydrologically perfect but sufficient as a fallback for feature
    engineering when pysheds is unavailable.

    Parameters
    ----------
    dem: 2-D elevation array.

    Returns
    -------
    Flow accumulation array (number of upstream cells + 1).
    """
    rows, cols = dem.shape
    accum = np.ones((rows, cols), dtype=np.float32)

    # D8 neighbour offsets (dr, dc)
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Build flow direction as index of steepest descent neighbour
    # Process cells in elevation order (lowest to highest for accumulation)
    flat = dem.ravel()
    order = np.argsort(flat)[::-1]  # process high → low, accumulate downslope

    for idx in order:
        r, c = divmod(int(idx), cols)
        best_drop = 0.0
        best_nr, best_nc = -1, -1
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                drop = dem[r, c] - dem[nr, nc]
                if drop > best_drop:
                    best_drop = drop
                    best_nr, best_nc = nr, nc
        if best_nr >= 0:
            accum[best_nr, best_nc] += accum[r, c]

    return accum


def _fill_pits(dem: np.ndarray) -> np.ndarray:
    """
    Fill single-cell pits in a DEM using a simple iterative approach.

    A pit is a cell that is strictly lower than the minimum of all its 8
    neighbours.  We raise it to the minimum-neighbour value.

    ``scipy.ndimage.minimum_filter`` with size=3 returns the minimum of the
    full 3×3 window *including* the centre cell, so a pit (which is itself the
    window minimum) would never be detected.  Instead we compute the minimum of
    the 8 surrounding neighbours by dilating with an annular structuring
    element that excludes the centre.

    Parameters
    ----------
    dem: 2-D elevation array.

    Returns
    -------
    Pit-filled elevation array (same dtype as input, cast to float32).
    """
    from scipy.ndimage import minimum_filter

    # 3×3 footprint that excludes the centre cell
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False  # exclude centre

    filled = dem.astype(np.float32).copy()
    for _ in range(10):  # iterate to handle chained pits
        # Minimum of the 8 surrounding neighbours (centre excluded)
        neighbour_min = minimum_filter(filled, footprint=footprint, mode="nearest")
        # A pit: centre strictly below all neighbours
        is_pit = filled < neighbour_min
        if not is_pit.any():
            break
        # Raise pits to the level of the lowest neighbour
        filled = np.where(is_pit, neighbour_min, filled)
    return filled


def _aggregate_to_coarse(
    bands_fine: np.ndarray,
    transform_fine,
    crs,
    res_fine: float,
    res_coarse: float,
) -> np.ndarray:
    """
    Aggregate fine-resolution multi-band raster to a coarser resolution by averaging.

    Parameters
    ----------
    bands_fine:     (n_bands, nrows_fine, ncols_fine) array.
    transform_fine: Affine transform of the fine raster.
    crs:            CRS of the fine raster.
    res_fine:       Fine pixel size in metres.
    res_coarse:     Coarse pixel size in metres.

    Returns
    -------
    (n_bands, nrows_coarse, ncols_coarse) array.
    """
    factor = max(1, int(round(res_coarse / res_fine)))
    n_bands, nrows, ncols = bands_fine.shape
    nrows_c = nrows // factor
    ncols_c = ncols // factor

    if nrows_c < 1 or ncols_c < 1:
        logger.warning("DEM too small to aggregate; returning as-is")
        return bands_fine

    trimmed = bands_fine[:, : nrows_c * factor, : ncols_c * factor]
    coarse = trimmed.reshape(n_bands, nrows_c, factor, ncols_c, factor)

    # Use nanmean so single bad pixels don't contaminate the cell
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells -> NaN is intended
        result = np.nanmean(coarse, axis=(2, 4)).astype(np.float32)
    logger.debug(
        "Aggregated %d×%d → %d×%d (factor=%d)",
        ncols, nrows, ncols_c, nrows_c, factor,
    )
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dem(
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    force: bool = False,
) -> Path:
    """
    Full DEM pipeline: discover → download → merge → compute terrain features.

    Parameters
    ----------
    bbox:  (lat_min, lon_min, lat_max, lon_max).
    force: Re-download/recompute even if cached.

    Returns
    -------
    Path of the multi-band 100m terrain feature raster.
    """
    if Path(_TERRAIN_100M_PATH).exists() and not force:
        logger.info("Loading cached terrain features from %s", _TERRAIN_100M_PATH)
        return Path(_TERRAIN_100M_PATH)

    tile_keys = list_lidar_tiles(bbox)
    tile_paths = download_tiles(tile_keys, force=force)

    dem_path = merge_and_reproject_tiles(
        tile_paths, bbox, Path(_DEM_5M_PATH), res_m=DEM_RES_M, force=force
    )

    terrain_path = compute_terrain_features(
        dem_path, Path(_TERRAIN_100M_PATH), res_coarse_m=GRID_RES_M, force=force
    )
    return terrain_path
