#!/usr/bin/env python3
"""
make_dataset.py — NYC flood prediction data pipeline orchestrator.

Usage
-----
Run the full pipeline::

    python make_dataset.py --step all

Run individual steps::

    python make_dataset.py --step reports
    python make_dataset.py --step dem
    python make_dataset.py --step rain
    python make_dataset.py --step landcover
    python make_dataset.py --step grid
    python make_dataset.py --step features
    python make_dataset.py --step splits
    python make_dataset.py --step stormwater

Override the study-area bounding box::

    python make_dataset.py --step all --bbox 40.635 -74.025 40.680 -73.960

Force re-download of all cached files::

    python make_dataset.py --step all --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add project root to sys.path so ``import floodmap`` always works
# regardless of how the script is invoked.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# dotenv — load .env if present (for API tokens, etc.)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

from floodmap.config import DATA_PROCESSED, DATA_RAW, DEFAULT_BBOX, LOG_FILE
from floodmap.utils import setup_logging

logger = logging.getLogger(__name__)

STEPS = ["reports", "dem", "rain", "landcover", "grid", "features", "splits", "stormwater"]


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def step_reports(bbox, force: bool) -> None:
    """Download and clean NYC 311 flood reports for the study area."""
    from floodmap.reports import download_reports

    logger.info("=== STEP: reports ===")
    t0 = time.perf_counter()
    df = download_reports(bbox=bbox, force=force)
    logger.info(
        "Reports complete: %d records in %.1fs",
        len(df),
        time.perf_counter() - t0,
    )
    if len(df) > 0:
        logger.info("Date range: %s → %s", df["created_date"].min(), df["created_date"].max())
        logger.info("Top descriptors:\n%s", df["descriptor"].value_counts().head())


def step_dem(bbox, force: bool) -> None:
    """Discover, download, and process NOAA LiDAR DEM tiles."""
    from floodmap.dem import run_dem

    logger.info("=== STEP: dem ===")
    t0 = time.perf_counter()
    terrain_path = run_dem(bbox=bbox, force=force)
    logger.info(
        "DEM/terrain complete: %s in %.1fs", terrain_path, time.perf_counter() - t0
    )


def step_rain(bbox, force: bool) -> None:
    """Download GHCN daily + ISD hourly rain and identify storm events."""
    from floodmap.rain import run_rain

    logger.info("=== STEP: rain ===")
    t0 = time.perf_counter()

    # Pass reports if available for event confirmation
    reports = None
    reports_path = DATA_RAW / "reports_311.parquet"
    if reports_path.exists():
        import pandas as pd

        reports = pd.read_parquet(reports_path)

    daily, events = run_rain(bbox=bbox, force=force, reports=reports)
    logger.info(
        "Rain complete: %d daily rows, %d storm events in %.1fs",
        len(daily),
        len(events),
        time.perf_counter() - t0,
    )
    if len(events) > 0:
        logger.info("Event date range: %s → %s", events["start_date"].min(), events["end_date"].max())


def step_landcover(bbox, force: bool) -> None:
    """Download and process ESA WorldCover land cover data."""
    from floodmap.landcover import run_landcover

    logger.info("=== STEP: landcover ===")
    t0 = time.perf_counter()
    lc_path, dw_path = run_landcover(bbox=bbox, force=force)
    logger.info(
        "Landcover complete: %s, %s in %.1fs",
        lc_path,
        dw_path,
        time.perf_counter() - t0,
    )


def step_grid(bbox, force: bool) -> None:
    """Generate the 100 m prediction grid over the study area."""
    from floodmap.grid import run_grid

    logger.info("=== STEP: grid ===")
    t0 = time.perf_counter()
    grid = run_grid(bbox=bbox, force=force)
    logger.info(
        "Grid complete: %d cells in %.1fs", len(grid), time.perf_counter() - t0
    )


def step_features(bbox, force: bool) -> None:
    """Join all data sources into the (cell × event) feature table."""
    from floodmap.features import run_features

    logger.info("=== STEP: features ===")
    t0 = time.perf_counter()
    features = run_features(bbox=bbox, force=force)
    logger.info(
        "Features complete: %d rows (%d cells × events) in %.1fs",
        len(features),
        len(features),
        time.perf_counter() - t0,
    )
    _print_feature_summary(features)


def step_splits(force: bool) -> None:
    """Apply train/val/test splits and save the final features.parquet."""
    from floodmap.splits import run_splits

    logger.info("=== STEP: splits ===")
    t0 = time.perf_counter()
    features = run_splits(force=force)
    logger.info(
        "Splits complete: %d rows in %.1fs", len(features), time.perf_counter() - t0
    )
    _print_final_summary(features)


def step_stormwater(bbox, force: bool) -> None:
    """Download NYC Stormwater Flood Map and join flood-zone features."""
    from floodmap.stormwater import run_stormwater

    logger.info("=== STEP: stormwater ===")
    t0 = time.perf_counter()
    features = run_stormwater(bbox=bbox, force=force)
    logger.info(
        "Stormwater complete: %d rows in %.1fs",
        len(features),
        time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_feature_summary(features) -> None:
    """Print a brief summary of the raw feature table."""
    import pandas as pd
    n_cells = features["cell_id"].nunique() if "cell_id" in features.columns else "N/A"
    n_events = features["event_id"].nunique() if "event_id" in features.columns else "N/A"
    n_flooded = int(features["flooded"].sum()) if "flooded" in features.columns else "N/A"
    frac = features["flooded"].mean() if "flooded" in features.columns else float("nan")
    nan_count = features.isna().sum().sum()

    print("\n--- Feature Table Summary ---")
    print(f"  Total rows    : {len(features):,}")
    print(f"  Unique cells  : {n_cells}")
    print(f"  Unique events : {n_events}")
    print(f"  Flooded rows  : {n_flooded} ({frac:.3%})")
    print(f"  NaN values    : {nan_count}")
    print(f"  Columns       : {list(features.columns)}\n")


def _print_final_summary(features) -> None:
    """Print the final pipeline summary statistics."""
    n_cells = features["cell_id"].nunique() if "cell_id" in features.columns else "N/A"
    n_events = features["event_id"].nunique() if "event_id" in features.columns else "N/A"
    n_flooded = int(features["flooded"].sum()) if "flooded" in features.columns else "N/A"
    frac = features["flooded"].mean() if "flooded" in features.columns else float("nan")
    nan_count = features.isna().sum().sum()

    output_path = DATA_PROCESSED / "features.parquet"
    size_mb = output_path.stat().st_size / 1e6 if output_path.exists() else 0

    print("\n" + "=" * 60)
    print("  FINAL PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Output file   : {output_path}")
    print(f"  File size     : {size_mb:.2f} MB")
    print(f"  Total rows    : {len(features):,}")
    print(f"  Unique cells  : {n_cells}")
    print(f"  Unique events : {n_events}")
    print(f"  Flooded rows  : {n_flooded} ({frac:.3%})")
    print(f"  NaN values    : {nan_count}")
    if "split" in features.columns:
        for sname in ["train", "val", "test"]:
            sub = features[features["split"] == sname]
            sf = sub["flooded"].mean() if len(sub) else float("nan")
            print(f"  {sname:<8}      : {len(sub):>8,} rows, {int(sub['flooded'].sum()):>6} flooded ({sf:.3%})")
    print("=" * 60 + "\n")

    logger.info(
        "Pipeline complete — n_cells=%s, n_events=%s, n_flooded=%s, balance=%.4f",
        n_cells, n_events, n_flooded, frac,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """
    Parse command-line arguments for the pipeline.

    Parameters
    ----------
    argv: Argument list (defaults to sys.argv).

    Returns
    -------
    argparse.Namespace with attributes: step, bbox, force.
    """
    parser = argparse.ArgumentParser(
        prog="make_dataset.py",
        description="NYC flood prediction data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        choices=["all"] + STEPS,
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
        default=None,
        help="Override study-area bounding box (default: southern Brooklyn)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download/recompute even if cached files exist",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    Main entry point for the pipeline.

    Parameters
    ----------
    argv: Argument list (defaults to sys.argv).

    Returns
    -------
    Exit code (0 = success, 1 = error).
    """
    args = parse_args(argv)

    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_file=LOG_FILE, level=log_level)

    bbox = tuple(args.bbox) if args.bbox else DEFAULT_BBOX
    force = args.force

    logger.info("=" * 60)
    logger.info("NYC Flood Prediction Pipeline")
    logger.info("Step      : %s", args.step)
    logger.info("Bbox      : %s", bbox)
    logger.info("Force     : %s", force)
    logger.info("=" * 60)

    t_start = time.perf_counter()

    try:
        steps_to_run = STEPS if args.step == "all" else [args.step]

        for step in steps_to_run:
            if step == "reports":
                step_reports(bbox, force)
            elif step == "dem":
                step_dem(bbox, force)
            elif step == "rain":
                step_rain(bbox, force)
            elif step == "landcover":
                step_landcover(bbox, force)
            elif step == "grid":
                step_grid(bbox, force)
            elif step == "features":
                step_features(bbox, force)
            elif step == "splits":
                step_splits(force)
            elif step == "stormwater":
                step_stormwater(bbox, force)
            else:
                logger.error("Unknown step: %s", step)
                return 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        return 1

    elapsed = time.perf_counter() - t_start
    logger.info("Pipeline finished in %.1f seconds", elapsed)
    print(f"\nPipeline finished in {elapsed:.1f}s")

    # Print final summary if features.parquet exists
    final_path = DATA_PROCESSED / "features.parquet"
    if final_path.exists():
        import pandas as pd

        try:
            features = pd.read_parquet(final_path)
            _print_final_summary(features)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
