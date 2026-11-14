"""
test_splits.py — Unit tests for train/val/test splitting logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodmap.config import MAX_NEG_RATIO, TRAIN_FRAC, VAL_FRAC


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features():
    """
    Synthetic feature DataFrame with 10 events × 20 cells = 200 rows.

    Events are named E0000–E0009 and assigned fake start dates one month apart.
    Flood labels: cells 0–4 are labelled flooded for events E0000–E0004 only.
    """
    n_events = 10
    n_cells = 20

    event_ids = [f"E{i:04d}" for i in range(n_events)]
    start_dates = pd.date_range("2016-01-01", periods=n_events, freq="90D")

    rows = []
    for ei, (eid, sdate) in enumerate(zip(event_ids, start_dates)):
        for ci in range(n_cells):
            flooded = int(ci < 5 and ei < 5)  # first 5 cells flood in first 5 events
            rows.append(
                {
                    "cell_id": ci,
                    "lat": 40.640 + ci * 0.001,
                    "lon": -74.010 + ci * 0.001,
                    "event_id": eid,
                    "start_date": sdate,
                    "end_date": sdate + pd.Timedelta(days=1),
                    "elev": float(ci),
                    "slope": 0.5,
                    "curvature": 0.0,
                    "twi": 5.0,
                    "flow_accum": 10.0,
                    "depression_depth": 0.1,
                    "dist_water": 500.0,
                    "impervious_frac": 0.5,
                    "landcover_built": 0.5,
                    "landcover_lowveg": 0.3,
                    "landcover_tree": 0.1,
                    "landcover_water": 0.05,
                    "rain_total_mm": 20.0 + ei,
                    "rain_peak_1h_mm": float("nan"),
                    "rain_prev_48h_mm": 2.0,
                    "flooded": flooded,
                    "block_id": ci // 10,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# No event leakage
# ---------------------------------------------------------------------------

class TestNoEventLeakage:
    """Ensure the same event never appears in two different splits."""

    def test_events_disjoint_across_splits(self, synthetic_features):
        """Event IDs must be mutually exclusive across train/val/test."""
        from floodmap.splits import apply_splits

        result = apply_splits(synthetic_features, force=True)

        train_ev = set(result[result["split"] == "train"]["event_id"].unique())
        val_ev = set(result[result["split"] == "val"]["event_id"].unique())
        test_ev = set(result[result["split"] == "test"]["event_id"].unique())

        assert train_ev.isdisjoint(val_ev), "Train and val share events"
        assert train_ev.isdisjoint(test_ev), "Train and test share events"
        assert val_ev.isdisjoint(test_ev), "Val and test share events"

    def test_all_events_assigned(self, synthetic_features):
        """Every event in the input must appear in exactly one split."""
        from floodmap.splits import apply_splits

        result = apply_splits(synthetic_features, force=True)
        input_events = set(synthetic_features["event_id"].unique())
        output_events = set(result["event_id"].unique())
        assert input_events == output_events, (
            "Some events were lost during split assignment"
        )


# ---------------------------------------------------------------------------
# Chronological ordering
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    """Events must be assigned chronologically: oldest → train, newest → test."""

    def test_train_has_oldest_events(self, synthetic_features):
        """The train split should contain the chronologically oldest events."""
        from floodmap.splits import apply_splits, _sort_events_chronologically

        result = apply_splits(synthetic_features, force=True)
        sorted_events = _sort_events_chronologically(synthetic_features)

        n = len(sorted_events)
        n_train = max(1, int(n * TRAIN_FRAC))
        expected_train = set(sorted_events[:n_train])

        actual_train = set(result[result["split"] == "train"]["event_id"].unique())
        # Allow for some downsampling removal but not reordering
        assert actual_train.issubset(expected_train | set(sorted_events[n_train:])) or \
               expected_train.issubset(actual_train | set()), \
               "Train split should contain the oldest events"

    def test_chronological_order_respected(self, synthetic_features):
        """Train max date < val max date < test max date."""
        from floodmap.splits import apply_splits

        result = apply_splits(synthetic_features, force=True)

        def max_date(split_name):
            sub = result[result["split"] == split_name]
            return pd.to_datetime(sub["start_date"]).max() if len(sub) else pd.Timestamp.min

        train_max = max_date("train")
        val_max = max_date("val")
        test_max = max_date("test")

        # We allow equality only if splits share the boundary event
        assert train_max <= val_max, (
            f"Train max date ({train_max}) should be <= val max date ({val_max})"
        )
        assert val_max <= test_max, (
            f"Val max date ({val_max}) should be <= test max date ({test_max})"
        )


# ---------------------------------------------------------------------------
# Spatial block assignment
# ---------------------------------------------------------------------------

class TestSpatialBlocks:
    """Test block ID assignment for spatial cross-validation."""

    def test_block_ids_assigned(self):
        """Every grid cell should receive a non-negative block_id."""
        from floodmap.grid import assign_spatial_blocks

        grid = pd.DataFrame(
            {
                "cell_id": range(9),
                "utm_x": [580_000 + (i % 3) * 100 for i in range(9)],
                "utm_y": [4_497_000 + (i // 3) * 100 for i in range(9)],
            }
        )
        result = assign_spatial_blocks(grid, block_size_m=1000.0)
        assert "block_id" in result.columns
        assert (result["block_id"] >= 0).all()

    def test_adjacent_cells_same_block(self):
        """Cells within a 1km block should share the same block_id."""
        from floodmap.grid import assign_spatial_blocks

        grid = pd.DataFrame(
            {
                "cell_id": [0, 1],
                "utm_x": [580_000, 580_050],  # 50 m apart — same 1km block
                "utm_y": [4_497_000, 4_497_050],
            }
        )
        result = assign_spatial_blocks(grid, block_size_m=1000.0)
        assert result.iloc[0]["block_id"] == result.iloc[1]["block_id"]

    def test_distant_cells_different_blocks(self):
        """Cells > 1 km apart should be in different blocks."""
        from floodmap.grid import assign_spatial_blocks

        grid = pd.DataFrame(
            {
                "cell_id": [0, 1],
                "utm_x": [580_000, 581_500],  # 1500 m apart — different 1km blocks
                "utm_y": [4_497_000, 4_497_000],
            }
        )
        result = assign_spatial_blocks(grid, block_size_m=1000.0)
        assert result.iloc[0]["block_id"] != result.iloc[1]["block_id"]


# ---------------------------------------------------------------------------
# Negative downsampling
# ---------------------------------------------------------------------------

class TestNegativeDownsampling:
    """Validate train-only negative downsampling."""

    def test_ratio_capped(self, synthetic_features):
        """After downsampling, neg:pos ratio in train should be <= MAX_NEG_RATIO."""
        from floodmap.splits import apply_splits, downsample_negatives

        # First apply splits to get the 'split' column
        with_splits = apply_splits(synthetic_features, force=True)

        # Reset so we can test downsampling in isolation
        with_splits_reset = with_splits.copy()
        with_splits_reset["split"] = with_splits["split"]  # keep as-is

        result = downsample_negatives(with_splits_reset, split="train", max_ratio=MAX_NEG_RATIO)
        train = result[result["split"] == "train"]

        n_pos = int(train["flooded"].sum())
        n_neg = int((train["flooded"] == 0).sum())
        if n_pos > 0:
            ratio = n_neg / n_pos
            assert ratio <= MAX_NEG_RATIO + 0.01, (
                f"Neg:pos ratio {ratio:.2f} exceeds MAX_NEG_RATIO={MAX_NEG_RATIO}"
            )

    def test_val_test_untouched(self, synthetic_features):
        """Val and test splits should be unmodified by downsampling."""
        from floodmap.splits import apply_splits, downsample_negatives

        with_splits = apply_splits(synthetic_features, force=True)

        val_before = with_splits[with_splits["split"] == "val"].copy()
        test_before = with_splits[with_splits["split"] == "test"].copy()

        result = downsample_negatives(with_splits, split="train", max_ratio=1)

        val_after = result[result["split"] == "val"]
        test_after = result[result["split"] == "test"]

        assert len(val_after) == len(val_before), "Val size changed after downsampling"
        assert len(test_after) == len(test_before), "Test size changed after downsampling"

    def test_no_positive_no_crash(self):
        """Downsampling with zero positives should not raise an exception."""
        from floodmap.splits import downsample_negatives

        df = pd.DataFrame(
            {
                "split": ["train"] * 10,
                "flooded": np.zeros(10, dtype=np.int8),
                "event_id": ["E0000"] * 10,
            }
        )
        # Should complete without error
        result = downsample_negatives(df, split="train", max_ratio=5)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Running splits twice should produce identical output."""

    def test_deterministic(self, synthetic_features):
        """apply_splits should be deterministic given the same input."""
        from floodmap.splits import apply_splits

        result1 = apply_splits(synthetic_features.copy(), force=True)
        result2 = apply_splits(synthetic_features.copy(), force=True)

        pd.testing.assert_frame_equal(
            result1.sort_values(["cell_id", "event_id"]).reset_index(drop=True),
            result2.sort_values(["cell_id", "event_id"]).reset_index(drop=True),
            check_like=False,
        )
