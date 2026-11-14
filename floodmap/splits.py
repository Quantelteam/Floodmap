"""
splits.py — Train/val/test splits by storm event chronology with spatial block IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from floodmap.config import (
    DATA_PROCESSED,
    MAX_NEG_RATIO,
    TRAIN_FRAC,
    VAL_FRAC,
)

logger = logging.getLogger(__name__)

_FEATURES_PATH = DATA_PROCESSED / "features.parquet"


def chronological_event_splits(
    events: pd.Series,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> dict[str, list]:
    """
    Split a list of event IDs into train/val/test by chronological order.

    Events are assumed to be sorted oldest-first (order preserved in *events*).
    The oldest *train_frac* go to train, the next *val_frac* to val, the rest
    to test.

    Parameters
    ----------
    events:     Series or list of unique event IDs in chronological order.
    train_frac: Fraction allocated to training (default 0.70).
    val_frac:   Fraction allocated to validation (default 0.15).

    Returns
    -------
    Dict with keys 'train', 'val', 'test' mapping to lists of event IDs.
    """
    events_list = list(events)
    n = len(events_list)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    # Ensure at least 1 event in test when total > 2
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 3:
        n_val = max(1, n_val - 1)
        n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0

    train_ev = events_list[:n_train]
    val_ev = events_list[n_train: n_train + n_val]
    test_ev = events_list[n_train + n_val:]

    return {"train": train_ev, "val": val_ev, "test": test_ev}


def downsample_negatives(
    df: pd.DataFrame,
    split: str = "train",
    max_ratio: int = MAX_NEG_RATIO,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Downsample non-flooded rows in the *train* split to cap the negative-to-positive ratio.

    Only the **training split** is modified; val and test sets are returned as-is
    to preserve the natural class distribution for unbiased evaluation.

    Parameters
    ----------
    df:           Feature DataFrame with 'split' and 'flooded' columns.
    split:        Which split to apply downsampling to (default 'train').
    max_ratio:    Maximum negative:positive ratio after downsampling.
    random_state: Random seed for reproducibility.

    Returns
    -------
    DataFrame with the train split downsampled, other splits unchanged.
    """
    train_mask = df["split"] == split
    other = df[~train_mask].copy()
    train = df[train_mask].copy()

    pos = train[train["flooded"] == 1]
    neg = train[train["flooded"] == 0]

    n_pos = len(pos)
    if n_pos == 0:
        logger.warning("No positive examples in %s split — skipping downsampling", split)
        return df

    max_neg = n_pos * max_ratio
    if len(neg) > max_neg:
        neg_sampled = neg.sample(n=int(max_neg), random_state=random_state)
        logger.info(
            "Downsampled negatives in %s: %d → %d (ratio %.1f:1)",
            split,
            len(neg),
            len(neg_sampled),
            len(neg_sampled) / n_pos,
        )
        train = pd.concat([pos, neg_sampled], ignore_index=True)
    else:
        logger.info(
            "No downsampling needed in %s (neg=%d, pos=%d, ratio=%.1f:1)",
            split,
            len(neg),
            n_pos,
            len(neg) / max(n_pos, 1),
        )

    return pd.concat([train, other], ignore_index=True)


def apply_splits(
    features: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    """
    Add a ``split`` column to *features* and apply train-only downsampling.

    Split assignment is deterministic — running twice with the same input
    produces identical output.

    Parameters
    ----------
    features: Raw feature DataFrame (from ``features_raw.parquet``).
    force:    Overwrite ``features.parquet`` even if it already exists.

    Returns
    -------
    Feature DataFrame with 'split' column, saved to ``data/processed/features.parquet``.
    """
    out_path = Path(_FEATURES_PATH)
    if out_path.exists() and not force:
        logger.info("Loading cached split features from %s", out_path)
        return pd.read_parquet(out_path)

    if len(features) == 0:
        logger.error("Empty features DataFrame — cannot create splits")
        raise ValueError("Cannot split an empty DataFrame")

    # Determine chronological event order
    events_sorted = _sort_events_chronologically(features)
    split_map = chronological_event_splits(pd.Series(events_sorted))

    logger.info(
        "Event split sizes — train: %d, val: %d, test: %d",
        len(split_map["train"]),
        len(split_map["val"]),
        len(split_map["test"]),
    )

    event_to_split: dict[str, str] = {}
    for sname, ev_list in split_map.items():
        for eid in ev_list:
            event_to_split[eid] = sname

    features = features.copy()
    features["split"] = features["event_id"].map(event_to_split).fillna("test")

    # Apply negative downsampling to train only
    features = downsample_negatives(features, split="train")

    # Ensure no NaN values in final output
    from floodmap.utils import impute_dataframe

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if features[col].isna().any():
            fill = features[col].median()
            fill = fill if np.isfinite(fill) else 0.0
            features[col] = features[col].fillna(fill)

    # Cast types
    features["cell_id"] = features["cell_id"].astype(np.int64)
    features["flooded"] = features["flooded"].astype(np.int8)
    if "block_id" in features.columns:
        features["block_id"] = features["block_id"].astype(np.int64)

    # --- Summary statistics ---
    _log_split_summary(features)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    logger.info("Split features saved to %s", out_path)
    return features


def _sort_events_chronologically(features: pd.DataFrame) -> list[str]:
    """
    Return a list of unique event IDs sorted by their start date.

    Parameters
    ----------
    features: Feature DataFrame with 'event_id' and optionally 'start_date'.

    Returns
    -------
    Sorted list of unique event IDs.
    """
    if "start_date" in features.columns:
        event_dates = (
            features.groupby("event_id")["start_date"]
            .min()
            .sort_values()
        )
        return event_dates.index.tolist()
    else:
        # Fall back to lexicographic order (event IDs are E0000, E0001, …)
        return sorted(features["event_id"].unique().tolist())


def _log_split_summary(features: pd.DataFrame) -> None:
    """
    Log and print summary statistics for each split.

    Parameters
    ----------
    features: Feature DataFrame with 'split' and 'flooded' columns.
    """
    header = f"{'Split':<8} {'N rows':>10} {'N events':>10} {'N flooded':>10} {'Frac':>8}"
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("=" * len(header))
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for split in ["train", "val", "test"]:
        sub = features[features["split"] == split]
        if len(sub) == 0:
            continue
        n_events = sub["event_id"].nunique()
        n_flooded = int(sub["flooded"].sum())
        frac = sub["flooded"].mean()
        line = f"{split:<8} {len(sub):>10,} {n_events:>10} {n_flooded:>10,} {frac:>7.3%}"
        logger.info(line)
        print(line)

    total = len(features)
    total_flooded = int(features["flooded"].sum())
    total_frac = features["flooded"].mean()
    footer = f"{'TOTAL':<8} {total:>10,} {features['event_id'].nunique():>10} {total_flooded:>10,} {total_frac:>7.3%}"
    logger.info("=" * len(header))
    logger.info(footer)
    print("=" * len(header))
    print(footer)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_splits(force: bool = False) -> pd.DataFrame:
    """
    Load ``features_raw.parquet`` and produce ``features.parquet`` with splits.

    Parameters
    ----------
    force: Overwrite even if ``features.parquet`` exists.

    Returns
    -------
    Split feature DataFrame.
    """
    raw_path = DATA_PROCESSED / "features_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"features_raw.parquet not found at {raw_path}. "
            "Run the 'features' step first."
        )
    features = pd.read_parquet(raw_path)
    return apply_splits(features, force=force)
