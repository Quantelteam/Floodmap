"""
test_landcover.py — WorldCover tile selection.
"""

from __future__ import annotations

from floodmap.config import DEFAULT_BBOX


def test_nyc_bbox_uses_the_single_n39w075_tile():
    from floodmap.landcover import worldcover_tiles_for_bbox

    assert worldcover_tiles_for_bbox(DEFAULT_BBOX) == ["N39W075"]


def test_tiles_are_named_by_south_west_corner_multiple_of_3():
    from floodmap.landcover import worldcover_tiles_for_bbox

    # lat 40.6 -> floor to 39 (not 40), lon -74.0 -> floor to -75 (not -74)
    assert worldcover_tiles_for_bbox((40.6, -74.0, 40.7, -73.9)) == ["N39W075"]


def test_bbox_spanning_tile_edges_returns_all_tiles():
    from floodmap.landcover import worldcover_tiles_for_bbox

    assert worldcover_tiles_for_bbox((-1, -1, 1, 1)) == ["N00E000", "N00W003", "S03E000", "S03W003"]
