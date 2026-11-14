# NYC Flood Prediction Data Pipeline

Builds `data/processed/features.parquet` — one row per (100 m grid cell, storm event) with flood labels, terrain features, rain features, land cover features, and train/val/test splits.

## Study area

Default sub-area: southern Brooklyn/Queens (SW corner 40.635°N 74.025°W → NE corner 40.680°N 73.960°W), approximately 25 km².

---

## Getting started (new contributor)

### 1. Clone the repo

```bash
git clone https://github.com/<your-org>/floodmap.git
cd floodmap
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# Optional: richdem + pysheds for terrain analysis (often no Windows wheels).
# Without them the pipeline uses numpy finite differences and a simple D8 flow accumulation.
pip install -r requirements-optional.txt
```

Verify the setup with `python -m pytest -q` (no network needed).

### 4. (Optional) Get a free Socrata API token

Without a token the 311 download still works but is rate-limited.
Register at https://data.cityofnewyork.us/profile/app_tokens, then `cp .env.example .env` and fill in:

```
NYC_APP_TOKEN=your_token_here
```

### 5. Run the pipeline

```bash
# Full pipeline — downloads everything and builds features.parquet (~30–90 min first run)
python make_dataset.py --step all

# Or run steps individually
python make_dataset.py --step reports
python make_dataset.py --step dem
python make_dataset.py --step rain
python make_dataset.py --step landcover
python make_dataset.py --step grid
python make_dataset.py --step features
python make_dataset.py --step splits
python make_dataset.py --step stormwater
```

The pipeline is **idempotent** — re-running skips already-downloaded files. Use `--force` to re-download everything.

### 6. Skip the download — use the shared processed file

If a teammate has already run the pipeline, grab the pre-built file from Google Drive (link in the team chat) and place it at:

```
data/processed/features.parquet
```

Then you can go straight to modelling without downloading anything.

---

## Data sources and download sizes

| Source | Size | Download time |
|---|---|---|
| NYC 311 Reports (Socrata API) | ~50 MB | 1–3 min |
| NOAA LiDAR DEM (NOAA S3, anonymous) | ~1 GB per tile (2 tiles for the default bbox, ~2.2 GB) | 5–20 min |
| Rain — GHCN-Daily (NOAA NCEI) + ISD hourly (AWS Open Data mirror `noaa-global-hourly-pds`) | ~90 MB | 1–2 min |
| ESA WorldCover 10 m land cover | a few MB — only the bbox window of the 3°×3° tile is read remotely (HTTP range requests) | seconds |
| NYC Stormwater Flood Maps (NYC Open Data file `9i7c-xyvv`, zip of geodatabases) | ~34 MB zip, deleted after extracting the 2 scenarios (~40 MB) | <1 min |

> **LiDAR is the big one.** The tile files carry no coordinates in their names, so the pipeline reads each tile's header and downloads only tiles that overlap your bounding box — not the whole 31-tile / ~20 GB set.
> Raw files land in `data/raw/` which is gitignored. They are never committed to the repo.
>
> **Rain station:** both daily (GHCN `USW00094728`) and hourly (ISD `72505394728`) data come from NY City Central Park. The public ISD archive lags by months; events newer than its last record get `rain_peak_1h_mm` estimated from their total rainfall (a warning is logged).
>
> **Stormwater scenarios:** *moderate* = 2.13 in/hr with current sea levels; *extreme* = 3.66 in/hr with **2080 sea-level rise** (the only extreme scenario NYC publishes).

---

## What NOT to commit

`.gitignore` already handles this, but for clarity:

| Path | Why excluded |
|---|---|
| `data/raw/` | Large downloads (GBs of TIF/CSV files) |
| `.env` | Contains your API token |
| `.venv/` | Virtual environment — each contributor makes their own |

`data/processed/features.parquet` and `flood_map_comparison.parquet` **are** committed (<1 MB for the default bbox), so collaborators can start modelling without running the pipeline. If you enlarge the bbox and the file grows past ~50 MB, add it to `.gitignore` and share it via Google Drive / Git LFS instead. `data/processed/features_raw.parquet` (intermediate) is ignored.

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the test suite on every push and pull request.

---

## Output schema (`data/processed/features.parquet`)

| Column | Type | Description |
|---|---|---|
| `cell_id` | int64 | Grid cell identifier |
| `lat` | float64 | Cell centroid latitude (WGS-84) |
| `lon` | float64 | Cell centroid longitude (WGS-84) |
| `event_id` | str | Storm event identifier (E0000, E0001, …) |
| `elev` | float64 | Mean elevation (m) |
| `slope` | float64 | Slope (degrees) |
| `curvature` | float64 | Planform curvature (1/m) |
| `twi` | float64 | Topographic Wetness Index |
| `flow_accum` | float64 | Flow accumulation (cells) |
| `depression_depth` | float64 | Pit/depression depth (m) |
| `dist_water` | float64 | Distance to nearest water pixel (m) |
| `impervious_frac` | float64 | Fraction of impervious surface (0–1) |
| `landcover_built` | float64 | Fraction built-up land (0–1) |
| `landcover_lowveg` | float64 | Fraction low vegetation (0–1) |
| `landcover_tree` | float64 | Fraction tree cover (0–1) |
| `landcover_water` | float64 | Fraction water (0–1) |
| `rain_total_mm` | float64 | Total event rainfall (mm) |
| `rain_peak_1h_mm` | float64 | Peak 1-hour rainfall (mm); estimated from `rain_total_mm` for events after the ISD archive ends |
| `rain_prev_48h_mm` | float64 | Antecedent 48 h rainfall (mm) |
| `flooded` | int8 | Label: 1 = flooded, 0 = not flooded |
| `split` | str | `train` / `val` / `test` |
| `block_id` | int64 | 1 km² spatial block for cross-validation |
| `in_moderate_flood_zone` | bool | Inside NYC official moderate flood zone |
| `in_extreme_flood_zone` | bool | Inside NYC official extreme flood zone |

---

## Pipeline steps explained

| Step | What it does | Output |
|---|---|---|
| `reports` | Downloads NYC 311 flood complaints via Socrata API (paginated), filters to 4 flood descriptor types, deduplicates to one report per (100 m cell, day) | `data/raw/reports_311.parquet` |
| `dem` | Downloads the NOAA LiDAR tiles overlapping the bbox (anonymous S3), mosaics them onto a 5 m UTM 18N grid aligned to the 100 m prediction grid (feet → metres), computes elevation/slope/curvature/TWI/flow accumulation/depression depth, aggregates to 100 m | `data/raw/dem_5m.tif`, `data/raw/terrain_100m.tif` |
| `rain` | Downloads GHCN-Daily CSV for Central Park + ISD hourly, identifies storm events (rainy-day clusters with ≥3 flood reports and ≥10 mm rain) | `data/raw/rain_daily.parquet`, `data/raw/events.parquet` |
| `landcover` | Reads the bbox window of the ESA WorldCover 10 m tile (N39W075 for NYC) straight from S3, reprojects to UTM, computes per-cell fractions for built-up / low-veg / tree / water and distance to water | `data/raw/worldcover_clipped.tif`, `data/raw/landcover_100m.tif`, `data/raw/dist_water_100m.tif` |
| `grid` | Generates 100 m × 100 m grid in UTM 18N, assigns 1 km spatial block IDs | `data/raw/grid.parquet`, `data/raw/grid.gpkg` |
| `features` | Cross-joins grid cells × storm events, samples rasters at centroids, assigns flood labels (reports are snapped to grid cells from their coordinates; 1 if any report in the cell during event ±6 h), imputes NaNs | `data/processed/features_raw.parquet` |
| `splits` | Chronological 70/15/15 event split, downsamples train negatives (max 20:1 ratio), validates no event leakage | `data/processed/features.parquet` |
| `stormwater` | Downloads the NYC Stormwater Flood Maps zip, extracts the moderate/extreme geodatabases, flags cells whose centroid is inside a flood polygon, computes precision/recall vs. flood labels | updates `features.parquet`, `data/processed/flood_map_comparison.parquet` |

---

## CLI reference

```bash
python make_dataset.py --step {all,reports,dem,rain,landcover,grid,features,splits,stormwater}
                       [--bbox LAT_MIN LON_MIN LAT_MAX LON_MAX]
                       [--force]
                       [--log-level {DEBUG,INFO,WARNING}]
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests use small synthetic inputs (no network). They never touch `data/raw/`.

---

## File structure

```
make_dataset.py          Main orchestration script
floodmap/
  __init__.py
  config.py              Constants, paths, API configs
  reports.py             NYC 311 download & cleaning
  dem.py                 LiDAR DEM download & terrain features
  rain.py                Rain download & storm event identification
  landcover.py           ESA WorldCover download & fractions
  grid.py                100 m grid generation
  features.py            Feature table join
  splits.py              Train/val/test splits
  stormwater.py          NYC Stormwater Flood Map
  utils.py               Shared helpers
tests/
  test_reports.py
  test_dem.py
  test_rain.py
  test_features.py
  test_splits.py
  test_landcover.py
  test_stormwater.py
data/
  raw/                   Downloaded files — gitignored, re-created by pipeline
  processed/             Output parquet files — committed if <100 MB
requirements.txt           Core dependencies
requirements-optional.txt  richdem / pysheds (optional)
.env.example             Template for .env (token); copy to .env
.github/workflows/       CI: runs pytest
.env                     API tokens — gitignored, never commit
```

---

## Troubleshooting

**Download fails** — the pipeline stops with an error rather than substituting fake data. Re-run the step (`--force` to discard caches). The NOAA S3 bucket is public but can be slow.

**Synthetic data (tests / offline demos only)** — set `FLOODMAP_ALLOW_SYNTHETIC=1` to let the DEM step fall back to a synthetic terrain when no LiDAR is available. Never use the resulting dataset for modelling; delete `data/raw/dem_5m.tif`, `terrain_100m.tif` and `data/processed/*.parquet` before a real run.

**Stormwater step prints `organizePolygons() ... more than 100 parts`** — harmless GDAL warning; the layers are a few giant MultiPolygons. They are split into single polygons after reading, so the step takes ~1 min.

**Few positive labels in val/test** — 311 reports are sparse (~1 % of cell-event rows are flooded, ~0.3 % in val/test, which are not downsampled). Prefer PR-AUC / recall at fixed precision over accuracy, and treat val/test metrics as noisy. About 77 % of reports are "Sewer Backup", which is not always surface flooding; edit `FLOOD_DESCRIPTORS` in `floodmap/config.py` to change that.

**Rate limited on 311 API** — add `NYC_APP_TOKEN` to your `.env`.

**Out of memory on DEM step** — reduce `DEM_RES_M` in `floodmap/config.py` from 5 to 10, or shrink the bounding box with `--bbox`.
