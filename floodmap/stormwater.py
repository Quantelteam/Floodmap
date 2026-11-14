"""
stormwater.py — NYC Stormwater Flood Map download, comparison, and feature join.

Source: NYC Open Data file asset ``9i7c-xyvv`` ("NYC Stormwater Flood Maps"), a
~34 MB zip holding one ESRI file geodatabase per rainfall scenario.  We use:

* moderate: 2.13 in/hr, current sea levels
* extreme:  3.66 in/hr, 2080 sea-level rise (the only extreme scenario published)

Each layer is a MultiPolygon in EPSG:2263 with a ``Flooding_Category`` attribute
(1 = nuisance, 2 = deep & contiguous, 3 = tidally influenced).  All categories
count as "inside the flood zone".
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

from floodmap.config import (
    CRS_NYC,
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_BBOX,
    STORMWATER_DATASET_ID,
    STORMWATER_EXTREME_KEY,
    STORMWATER_FILE_URL,
    STORMWATER_META_URL,
    STORMWATER_MODERATE_KEY,
)
from floodmap.utils import download_file

logger = logging.getLogger(__name__)

_STORMWATER_DIR = DATA_RAW / "stormwater"


def resolve_stormwater_file_url(timeout: int = 60) -> Tuple[str, str]:
    """
    Look up the current download URL of the stormwater zip from Socrata metadata.

    Parameters
    ----------
    timeout: Request timeout in seconds.

    Returns
    -------
    (download_url, filename) of the file attached to dataset ``9i7c-xyvv``.
    """
    resp = requests.get(STORMWATER_META_URL, timeout=timeout)
    resp.raise_for_status()
    meta = resp.json()
    blob, filename = meta["blobId"], meta["blobFilename"]
    url = STORMWATER_FILE_URL.format(
        dataset=STORMWATER_DATASET_ID, blob=blob, filename=filename
    )
    return url, filename


def _gdb_dir(key: str, root: Path = _STORMWATER_DIR) -> Path:
    """Return the local extraction directory for the scenario geodatabase *key*."""
    return Path(root) / f"{key}.gdb"


def _extract_scenarios(zip_path: Path, keys: Tuple[str, ...], root: Path) -> None:
    """
    Extract only the geodatabases matching *keys* from the stormwater zip.

    Files are written to ``root/<key>.gdb/`` (short names — the original paths
    exceed Windows' 260-character limit).  The other scenario geodatabases use
    a compressed table format the open-source OpenFileGDB driver cannot read, so
    they are skipped.

    Parameters
    ----------
    zip_path: Downloaded zip.
    keys:     Substrings identifying the wanted ``.gdb`` folders.
    root:     Output directory.
    """
    with zipfile.ZipFile(zip_path) as zf:
        for key in keys:
            dest = _gdb_dir(key, root)
            if dest.exists():
                shutil.rmtree(dest)
            members = [
                i for i in zf.infolist()
                if ".gdb/" in i.filename and key in i.filename and not i.is_dir()
            ]
            if not members:
                raise RuntimeError(f"Scenario '{key}' not found in {zip_path.name}")
            for info in members:
                target = dest / info.filename.split(".gdb/", 1)[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
            logger.info("Extracted %s (%d files)", dest.name, len(members))


def download_stormwater_map(force: bool = False) -> Dict[str, Path]:
    """
    Download the NYC Stormwater Flood Maps zip and extract the needed scenarios.

    Parameters
    ----------
    force: Re-download and re-extract even if the geodatabases already exist.

    Returns
    -------
    ``{"moderate": <gdb path>, "extreme": <gdb path>}``.
    """
    keys = (STORMWATER_MODERATE_KEY, STORMWATER_EXTREME_KEY)
    paths = {
        "moderate": _gdb_dir(STORMWATER_MODERATE_KEY),
        "extreme": _gdb_dir(STORMWATER_EXTREME_KEY),
    }
    if all(p.exists() for p in paths.values()) and not force:
        logger.debug("Cache hit — stormwater geodatabases")
        return paths

    url, filename = resolve_stormwater_file_url()
    zip_path = _STORMWATER_DIR / filename
    download_file(url, zip_path, force=True, desc="NYC Stormwater Flood Maps", timeout=300)
    _extract_scenarios(zip_path, keys, _STORMWATER_DIR)
    zip_path.unlink(missing_ok=True)  # ~34 MB, no need to keep once extracted
    return paths


def load_flood_layer(
    gdb_path: Path,
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
) -> "geopandas.GeoDataFrame":
    """
    Read a flood scenario geodatabase, clipped to *bbox*, with valid geometries.

    The source layers contain invalid polygons, so geometries are repaired with
    ``make_valid``.  Data stay in EPSG:2263 (NY State Plane, US feet).

    Parameters
    ----------
    gdb_path: Path to the extracted ``.gdb`` directory.
    bbox:     (lat_min, lon_min, lat_max, lon_max) in WGS-84.

    Returns
    -------
    GeoDataFrame in EPSG:2263 (possibly empty if the bbox has no flood zone).
    """
    import geopandas as gpd
    import pyogrio
    from shapely.geometry import box as shapely_box
    from shapely.validation import make_valid

    lat_min, lon_min, lat_max, lon_max = bbox
    bounds_2263 = (
        gpd.GeoSeries([shapely_box(lon_min, lat_min, lon_max, lat_max)], crs="EPSG:4326")
        .to_crs(CRS_NYC)
        .total_bounds
    )
    layer = pyogrio.list_layers(gdb_path)[0][0]
    gdf = gpd.read_file(gdb_path, layer=layer, bbox=tuple(bounds_2263))
    if len(gdf):
        # The layers are a handful of giant MultiPolygons (>100k parts).  Validating
        # or point-testing those as single geometries takes many minutes, so split
        # them into single polygons first and repair only the invalid ones.
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        bad = ~gdf.geometry.is_valid
        if bad.any():
            gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].apply(make_valid)
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        gdf = gdf[gdf.geometry.geom_type == "Polygon"].reset_index(drop=True)
    logger.info("Loaded %s: %d polygons in study area", Path(gdb_path).name, len(gdf))
    return gdf


def assign_flood_zones(
    grid: pd.DataFrame,
    moderate_gdf: "geopandas.GeoDataFrame | None",
    extreme_gdf: "geopandas.GeoDataFrame | None",
) -> pd.DataFrame:
    """
    Flag every grid cell whose centroid lies inside an official flood polygon.

    Parameters
    ----------
    grid:         DataFrame with ``cell_id``, ``lat`` and ``lon`` (WGS-84).
    moderate_gdf: Moderate-scenario polygons (any CRS), or None.
    extreme_gdf:  Extreme-scenario polygons (any CRS), or None.

    Returns
    -------
    DataFrame with ``cell_id``, ``in_moderate_flood_zone``, ``in_extreme_flood_zone``.
    """
    import geopandas as gpd

    result = grid[["cell_id"]].copy()
    points = gpd.GeoDataFrame(
        grid[["cell_id"]].copy(),
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
        crs="EPSG:4326",
    ).to_crs(CRS_NYC)

    for col, layer in (
        ("in_moderate_flood_zone", moderate_gdf),
        ("in_extreme_flood_zone", extreme_gdf),
    ):
        inside = np.zeros(len(grid), dtype=bool)
        if layer is not None and len(layer) > 0:
            polys = layer.to_crs(CRS_NYC)[["geometry"]]
            hit = gpd.sjoin(points, polys, how="inner", predicate="within")
            inside = result["cell_id"].isin(hit["cell_id"]).values
        result[col] = inside
        logger.info("%s: %.1f%% of cells", col, 100 * inside.mean())
    return result


def compare_flood_map_vs_labels(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare official stormwater flood zones against observed 311 flood labels.

    Computes confusion-matrix metrics for how well each zone predicts flooding.

    Parameters
    ----------
    features: Feature DataFrame with 'flooded', 'in_moderate_flood_zone',
              'in_extreme_flood_zone' columns.

    Returns
    -------
    DataFrame of comparison metrics with rows for 'moderate' and 'extreme' zones.
    """
    rows = []
    for zone in ["moderate", "extreme"]:
        col = f"in_{zone}_flood_zone"
        if col not in features.columns:
            continue

        y_true = features["flooded"].values.astype(int)
        y_pred = features[col].astype(int).values

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / max(len(y_true), 1)

        rows.append(
            {
                "zone": zone,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
            }
        )
        logger.info(
            "Stormwater %s zone vs flood labels: P=%.3f R=%.3f F1=%.3f Acc=%.3f",
            zone, precision, recall, f1, accuracy,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_stormwater(
    features_path: Path | None = None,
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download stormwater map, join flood-zone flags to features, compare vs labels.

    Updates ``data/processed/features.parquet`` in-place with two new columns:
    ``in_moderate_flood_zone`` and ``in_extreme_flood_zone``.  Raises on download
    or read failure so an all-False flood-zone column is never written silently.

    Parameters
    ----------
    features_path: Path to features parquet (defaults to ``data/processed/features.parquet``).
    bbox:          Study-area bounding box.
    force:         Re-download stormwater data.

    Returns
    -------
    Updated features DataFrame.
    """
    fp = features_path or DATA_PROCESSED / "features.parquet"
    if not Path(fp).exists():
        raise FileNotFoundError(
            f"features.parquet not found at {fp}. Run splits step first."
        )

    features = pd.read_parquet(fp)

    gdbs = download_stormwater_map(force=force)
    moderate = load_flood_layer(gdbs["moderate"], bbox)
    extreme = load_flood_layer(gdbs["extreme"], bbox)

    grid_cols = ["cell_id", "lat", "lon"]
    if not all(c in features.columns for c in grid_cols):
        raise KeyError("Required columns (cell_id, lat, lon) missing from features")
    grid = features[grid_cols].drop_duplicates("cell_id").reset_index(drop=True)

    zone_df = assign_flood_zones(grid, moderate, extreme)

    # Merge flood zone columns back
    for col in ["in_moderate_flood_zone", "in_extreme_flood_zone"]:
        if col in features.columns:
            features = features.drop(columns=[col])

    features = features.merge(zone_df, on="cell_id", how="left")
    features["in_moderate_flood_zone"] = features["in_moderate_flood_zone"].fillna(False)
    features["in_extreme_flood_zone"] = features["in_extreme_flood_zone"].fillna(False)

    # Compare flood map vs labels
    comparison = compare_flood_map_vs_labels(features)
    if len(comparison) > 0:
        logger.info("Flood map comparison:\n%s", comparison.to_string(index=False))
        comparison.to_parquet(DATA_PROCESSED / "flood_map_comparison.parquet", index=False)
        logger.info(
            "Comparison metrics saved to %s",
            DATA_PROCESSED / "flood_map_comparison.parquet",
        )

    # Save updated features
    features.to_parquet(fp, index=False)
    logger.info("Updated features (with stormwater zones) saved to %s", fp)
    return features
