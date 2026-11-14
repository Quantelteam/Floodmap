"""
conftest.py — keep the test suite from touching real pipeline outputs.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_pipeline_outputs(tmp_path, monkeypatch):
    """
    Redirect every cached/output path used by code under test to a temp dir.

    ``apply_splits(..., force=True)`` writes ``data/processed/features.parquet``;
    without this, running pytest overwrites the real dataset with synthetic rows.
    """
    import floodmap.splits as splits

    monkeypatch.setattr(splits, "_FEATURES_PATH", tmp_path / "features.parquet")
