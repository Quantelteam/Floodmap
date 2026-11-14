"""
landcover.py — ESA WorldCover 10m download, clip, and per-cell fraction computation.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from floodmap.config import (
    CRS_METRIC,
    DATA_RAW,
    DEFAULT_BBOX,
    GRID_RES_M,
    LC_BUILT,
    LC_GRASSLAND,
    LC_SHRUB,
    LC_TREE,
    LC_WATER,
    WORLDCOVER_TILE_DEG,
    WORLDCOVER_URL,
)
from floodmap.utils import reproject_raster

logger = logging.getLogger(__name__)

_WC_CLIPPED_PATH = DATA_RAW / "worldcover_clipped.tif"
_WC_UTM_PATH = DATA_RAW / "worldcover_utm.tif"
_LC_100M_PATH = DATA_RAW / "landcover_100m.tif"

# Multi-band output band names (in order)
LC_BAND_NAMES = [
    "impervious_frac",
    "landcover_built",
    "landcover_lowveg",
    "landcover_tree",
    "landcover_water",
]


def worldcover_tiles_for_bbox(
    bbox: Tuple[float, float, float, float],
    tile_deg: int = WORLDCOVER_TILE_DEG,
) -> List[str]:
    """
    Return the WorldCover tile names (e.g. ``'N39W075'``) that cover *bbox*.

    WorldCover tiles are ``tile_deg`` x ``tile_deg`` degrees and are named after
    their south-west corner, which is always a multiple of ``tile_deg``.

    Parameters
    ----------
    bbox:     (lat_min, lon_min, lat_max, lon_max) in WGS-84.
    tile_deg: Tile size in degrees (3 for WorldCover).

    Returns
    -------
    Sorted list of tile identifiers.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    eps = 1e-9  # a bbox edge exactly on a tile boundary must not pull in the next tile
    lat0s = range(
        int(math.floor(lat_min / tile_deg)) * tile_deg,
        int(math.floor((lat_max - eps) / tile_deg)) * tile_deg + 1,
        tile_deg,
    )
    lon0s = range(
        int(math.floor(lon_min / tile_deg)) * tile_deg,
        int(math.floor((lon_max - eps) / tile_deg)) * tile_deg + 1,
        tile_deg,
    )
    tiles = []
    for lat0 in lat0s:
        for lon0 in lon0s:
            ns = "N" if lat0 >= 0 else "S"
            ew = "E" if lon0 >= 0 else "W"
            tiles.append(f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}")
    return sorted(tiles)


def _read_worldcover_window(
    tile: str,
    bbox: Tuple[float, float, float, float],
    out_path: Path,
    retries: int = 3,
) -> Path:
    """
    Read only the *bbox* window of a remote WorldCover COG and save it locally.

    The tiles are Cloud-Optimised GeoTIFFs, so GDAL fetches just the needed
    1024x1024 blocks over HTTP range requests (a few hundred kB instead of the
    whole ~100 MB tile).

    Parameters
    ----------
    tile:     Tile identifier, e.g. ``'N39W075'``.
    bbox:     (lat_min, lon_min, lat_max, lon_max) in WGS-84.
    out_path: Destination GeoTIFF (EPSG:4326, uint8).
    retries:  Attempts on transient network errors.

    Returns
    -------
    Path of the written window raster.
    """
    import rasterio
    from rasterio.windows import from_bounds

    lat_min, lon_min, lat_max, lon_max = bbox
    url = "/vsicurl/" + WORLDCOVER_URL.format(tile=tile)
    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "2",
    }
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with rasterio.Env(**env), rasterio.open(url) as src:
                window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
                window = window.round_offsets().round_lengths()
                data = src.read(1, window=window)
                transform = src.window_transform(window)
                profile = {
                    "driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
                    "count": 1, "dtype": data.dtype, "crs": src.crs,
                    "transform": transform, "compress": "lzw",
                }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
            logger.info("Read WorldCover %s window %s → %s", tile, data.shape, out_path.name)
            return out_path
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("WorldCover %s read attempt %d/%d failed: %s", tile, attempt, retries, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not read WorldCover tile {tile} from {url}: {last_exc}")


def download_worldcover(
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    out_path: Path = _WC_CLIPPED_PATH,
    force: bool = False,
) -> Path:
    """
    Fetch the ESA WorldCover 10 m land cover for *bbox* (WGS-84 GeoTIFF).

    Works out which 3-degree tile(s) cover the bbox, reads just the bbox window
    from each remote COG and mosaics them if the bbox straddles several tiles.

    Parameters
    ----------
    bbox:     (lat_min, lon_min, lat_max, lon_max).
    out_path: Output GeoTIFF path.
    force:    Re-fetch even if *out_path* exists.

    Returns
    -------
    Path of the clipped WorldCover raster.
    """
    import rasterio
    from rasterio.merge import merge as rio_merge

    out_path = Path(out_path)
    if out_path.exists() and not force:
        logger.debug("Cache hit — WorldCover window: %s", out_path)
        return out_path

    tiles = worldcover_tiles_for_bbox(bbox)
    logger.info("WorldCover tiles covering bbox: %s", tiles)
    parts = [
        _read_worldcover_window(t, bbox, out_path.with_name(f"_wc_part_{t}.tif"))
        for t in tiles
    ]
    if len(parts) == 1:
        parts[0].replace(out_path)
        return out_path

    datasets = [rasterio.open(p) for p in parts]
    try:
        mosaic, transform = rio_merge(datasets)
        profile = datasets[0].profile.copy()
    finally:
        for ds in datasets:
            ds.close()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    for p in parts:
        p.unlink(missing_ok=True)
    return out_path


def prepare_worldcover(
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    force: bool = False,
) -> Path:
    """
    Fetch WorldCover for *bbox* and reproject it to UTM 18N at its native 10 m.

    Parameters
    ----------
    bbox:  (lat_min, lon_min, lat_max, lon_max).
    force: Reprocess even if cached.

    Returns
    -------
    Path of the UTM-reprojected clipped WorldCover raster.
    """
    if Path(_WC_UTM_PATH).exists() and not force:
        logger.debug("Cache hit — WorldCover UTM raster: %s", _WC_UTM_PATH)
        return Path(_WC_UTM_PATH)

    clipped = download_worldcover(bbox=bbox, out_path=_WC_CLIPPED_PATH, force=force)
    reproject_raster(
        clipped,
        Path(_WC_UTM_PATH),
        CRS_METRIC,
        res_m=10.0,  # keep at native 10 m
        resampling_name="nearest",  # categorical data
        force=True,
    )
    return Path(_WC_UTM_PATH)


def compute_landcover_fractions(
    wc_path: Path,
    out_path: Path,
    res_coarse_m: float = GRID_RES_M,
    force: bool = False,
) -> Path:
    """
    Aggregate WorldCover categories to per-cell fractions at *res_coarse_m*.

    Output is a multi-band GeoTIFF with bands:
        1 — impervious_frac   (fraction of built-up pixels, class 50)
        2 — landcover_built   (same as impervious_frac — kept for feature schema)
        3 — landcover_lowveg  (fraction of shrub [20] + grassland [30])
        4 — landcover_tree    (fraction of tree cover [10])
        5 — landcover_water   (fraction of water [80])

    Parameters
    ----------
    wc_path:      UTM-reprojected WorldCover raster.
    out_path:     Output path for the multi-band 100 m fraction raster.
    res_coarse_m: Target coarse resolution (default 100 m).
    force:        Overwrite existing output.

    Returns
    -------
    Path of the landcover fraction raster.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    out_path = Path(out_path)
    if out_path.exists() and not force:
        logger.debug("Cache hit — landcover fractions: %s", out_path.name)
        return out_path

    logger.info("Computing landcover fractions from %s", wc_path)

    with rasterio.open(wc_path) as src:
        lc = src.read(1).astype(np.uint8)
        res_fine = src.res[0]
        transform_fine = src.transform
        crs = src.crs
        nodata = src.nodata

    logger.info("WorldCover raster shape: %s, res: %.1f m", lc.shape, res_fine)

    # Create binary masks
    mask_built = (lc == LC_BUILT).astype(np.float32)
    mask_lowveg = ((lc == LC_SHRUB) | (lc == LC_GRASSLAND)).astype(np.float32)
    mask_tree = (lc == LC_TREE).astype(np.float32)
    mask_water = (lc == LC_WATER).astype(np.float32)

    factor = max(1, int(round(res_coarse_m / res_fine)))
    nrows, ncols = lc.shape
    nrows_c = nrows // factor
    ncols_c = ncols // factor

    def _aggregate(mask: np.ndarray) -> np.ndarray:
        """Reshape-aggregate a 2-D binary mask to fraction at coarse resolution."""
        trimmed = mask[: nrows_c * factor, : ncols_c * factor]
        return trimmed.reshape(nrows_c, factor, ncols_c, factor).mean(axis=(1, 3))

    if nrows_c < 1 or ncols_c < 1:
        logger.warning("WorldCover raster too small; using full-raster averages")
        nrows_c, ncols_c = 1, 1
        fracs = {
            "impervious": np.array([[mask_built.mean()]]),
            "built": np.array([[mask_built.mean()]]),
            "lowveg": np.array([[mask_lowveg.mean()]]),
            "tree": np.array([[mask_tree.mean()]]),
            "water": np.array([[mask_water.mean()]]),
        }
    else:
        fracs = {
            "impervious": _aggregate(mask_built),
            "built": _aggregate(mask_built),
            "lowveg": _aggregate(mask_lowveg),
            "tree": _aggregate(mask_tree),
            "water": _aggregate(mask_water),
        }

    bands = np.stack(
        [
            fracs["impervious"],
            fracs["built"],
            fracs["lowveg"],
            fracs["tree"],
            fracs["water"],
        ],
        axis=0,
    ).astype(np.float32)

    # Build coarse transform
    new_transform = rasterio.transform.Affine(
        transform_fine.a * factor,
        transform_fine.b,
        transform_fine.c,
        transform_fine.d,
        transform_fine.e * factor,
        transform_fine.f,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=nrows_c,
        width=ncols_c,
        count=bands.shape[0],
        dtype=np.float32,
        crs=crs,
        transform=new_transform,
        nodata=-9999,
        compress="lzw",
    ) as dst:
        dst.write(bands)
        for i, name in enumerate(LC_BAND_NAMES, 1):
            dst.update_tags(i, name=name)

    logger.info(
        "Landcover fractions saved to %s (%d×%d px)", out_path, ncols_c, nrows_c
    )
    return out_path


def compute_dist_water(
    wc_path: Path,
    out_path: Path,
    res_coarse_m: float = GRID_RES_M,
    force: bool = False,
) -> Path:
    """
    Compute distance-to-water raster (metres) from the WorldCover water class.

    Uses Euclidean distance transform on the water-class binary mask, then
    aggregates to 100 m resolution by taking the minimum distance within each
    coarse cell.

    Parameters
    ----------
    wc_path:      UTM WorldCover raster.
    out_path:     Output single-band 100 m distance raster.
    res_coarse_m: Coarse resolution in metres.
    force:        Overwrite existing output.

    Returns
    -------
    Path of the distance-to-water raster.
    """
    import rasterio
    from scipy.ndimage import distance_transform_edt

    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path

    with rasterio.open(wc_path) as src:
        lc = src.read(1).astype(np.uint8)
        res_fine = src.res[0]
        transform_fine = src.transform
        crs = src.crs

    is_water = (lc == LC_WATER)

    if not is_water.any():
        logger.warning("No water pixels in WorldCover — dist_water will be constant 9999")
        dist_fine = np.full(lc.shape, 9999.0, dtype=np.float32)
    else:
        # distance_transform_edt returns distances in pixels; multiply by resolution
        dist_fine = distance_transform_edt(~is_water).astype(np.float32) * res_fine

    factor = max(1, int(round(res_coarse_m / res_fine)))
    nrows, ncols = dist_fine.shape
    nrows_c = nrows // factor
    ncols_c = ncols // factor

    if nrows_c >= 1 and ncols_c >= 1:
        trimmed = dist_fine[: nrows_c * factor, : ncols_c * factor]
        dist_coarse = trimmed.reshape(nrows_c, factor, ncols_c, factor).min(axis=(1, 3))
    else:
        dist_coarse = np.array([[dist_fine.min()]])
        nrows_c, ncols_c = 1, 1

    new_transform = rasterio.transform.Affine(
        transform_fine.a * factor,
        transform_fine.b,
        transform_fine.c,
        transform_fine.d,
        transform_fine.e * factor,
        transform_fine.f,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=nrows_c,
        width=ncols_c,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=new_transform,
        nodata=-9999,
        compress="lzw",
    ) as dst:
        dst.write(dist_coarse[np.newaxis].astype(np.float32))

    logger.info("Distance-to-water saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_landcover(
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    force: bool = False,
) -> Tuple[Path, Path]:
    """
    Full landcover pipeline: download → clip/reproject → compute fractions + dist_water.

    Parameters
    ----------
    bbox:  (lat_min, lon_min, lat_max, lon_max).
    force: Re-download/recompute even if cached.

    Returns
    -------
    Tuple of (landcover_100m_path, dist_water_100m_path).
    """
    dist_water_path = DATA_RAW / "dist_water_100m.tif"

    if (
        Path(_LC_100M_PATH).exists()
        and Path(dist_water_path).exists()
        and not force
    ):
        return Path(_LC_100M_PATH), Path(dist_water_path)

    wc_path = prepare_worldcover(bbox=bbox, force=force)

    lc_path = compute_landcover_fractions(
        wc_path, Path(_LC_100M_PATH), res_coarse_m=GRID_RES_M, force=force
    )
    dw_path = compute_dist_water(
        wc_path, Path(dist_water_path), res_coarse_m=GRID_RES_M, force=force
    )
    return lc_path, dw_path
