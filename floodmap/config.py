"""
config.py — All constants, paths, bounding boxes, and API key environment variable names.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Study area
# ---------------------------------------------------------------------------
# Default sub-area: southern Brooklyn/Queens (high flood-report density).
# (lat_min, lon_min, lat_max, lon_max)
DEFAULT_BBOX = (40.635, -74.025, 40.680, -73.960)

# ---------------------------------------------------------------------------
# Grid / raster resolutions
# ---------------------------------------------------------------------------
GRID_RES_M = 100   # prediction grid resolution in metres
DEM_RES_M = 5      # terrain feature computation resolution

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------
CRS_WGS84 = "EPSG:4326"
CRS_NYC = "EPSG:2263"      # NY State Plane (US feet)
CRS_METRIC = "EPSG:32618"  # UTM Zone 18 N (metres) — used for all metric work

# ---------------------------------------------------------------------------
# 311 flood descriptors
# ---------------------------------------------------------------------------
FLOOD_DESCRIPTORS = [
    "Street Flooding (SJ)",
    "Catch Basin Clogged/Flooding (Use Comments) (SC1)",
    "Sewer Backup (Use Comments) (SA)",
    "Manhole Overflow (Use Comments) (MC)",
]

# ---------------------------------------------------------------------------
# Rain / event thresholds
# ---------------------------------------------------------------------------
RAIN_STATION_ID = "USW00094728"   # GHCN station: NY City Central Park
RAIN_STATION_ALT = "USW00094789"  # Fallback: JFK Airport
# Same Central Park station in the ISD hourly archive (USAF 725053 + WBAN 94728),
# so hourly peaks are consistent with the daily totals.
ISD_STATION_ID = "72505394728"
MIN_REPORTS_FOR_EVENT = 3         # min flood reports in bbox to call it a storm event
MIN_RAIN_MM = 10.0                # min daily rain (mm) to open a storm event
EVENT_GAP_DAYS = 1                # consecutive rainy days within this gap are merged

# ---------------------------------------------------------------------------
# Flood label window (hours around event start/end)
# ---------------------------------------------------------------------------
FLOOD_LABEL_BUFFER_H = 6

# ---------------------------------------------------------------------------
# Class-balance warnings
# ---------------------------------------------------------------------------
MIN_FLOOD_FRAC = 0.005   # warn if flooded fraction falls below 0.5 %
MAX_FLOOD_FRAC = 0.20    # warn if flooded fraction exceeds 20 %
MAX_NEG_RATIO = 20       # negative:positive cap for train downsampling

# ---------------------------------------------------------------------------
# Train / val / test chronological fractions
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
LOG_FILE = Path("data/pipeline.log")

# ---------------------------------------------------------------------------
# 311 Socrata API
# ---------------------------------------------------------------------------
SOCRATA_DOMAIN = "data.cityofnewyork.us"
SOCRATA_DATASET = "erm2-nwe9"
# Set NYC_APP_TOKEN env var to avoid throttling (optional but recommended)
SOCRATA_APP_TOKEN: str = os.environ.get("NYC_APP_TOKEN", "")

# ---------------------------------------------------------------------------
# NOAA LiDAR S3
# ---------------------------------------------------------------------------
NOAA_LIDAR_BUCKET = "noaa-nos-coastal-lidar-pds"
NOAA_LIDAR_PREFIX = "dem/NYC_topobathy_hydroEnforced_DEM_2017_9310/"

# ---------------------------------------------------------------------------
# NOAA GHCN-Daily
# ---------------------------------------------------------------------------
GHCN_BASE_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/"
)
# The ncei.noaa.gov /access/ path now returns an HTML page instead of CSV; the same
# files are served from the AWS Open Data mirror.
ISD_BASE_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/"

# ---------------------------------------------------------------------------
# ESA WorldCover
# ---------------------------------------------------------------------------
# WorldCover v200 (2021) is distributed as 3 deg x 3 deg Cloud-Optimised GeoTIFFs
# named after their SOUTH-WEST corner, which is always a multiple of 3 degrees
# (NYC is in N39W075 -- there is no N40W075 tile).  We read only the bbox window
# straight from S3 via /vsicurl/, so the ~100 MB tile is never fully downloaded.
WORLDCOVER_TILE_DEG = 3
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)

# WorldCover class values
LC_TREE = 10
LC_SHRUB = 20
LC_GRASSLAND = 30
LC_CROPLAND = 40
LC_BUILT = 50
LC_BARE = 60
LC_SNOW = 70
LC_WATER = 80
LC_WETLAND = 90
LC_MANGROVE = 95
LC_MOSS = 100

# ---------------------------------------------------------------------------
# NYC Stormwater Flood Map (Open Data)
# ---------------------------------------------------------------------------
# 9i7c-xyvv is a Socrata *file* asset (a zip of ESRI file geodatabases), not a
# geospatial dataset, so the /api/geospatial export endpoint returns HTTP 400.
# The blob id is resolved at run time from the metadata endpoint so the pipeline
# keeps working if NYC re-uploads the file.
STORMWATER_DATASET_ID = "9i7c-xyvv"
STORMWATER_META_URL = f"https://data.cityofnewyork.us/api/views/{STORMWATER_DATASET_ID}.json"
STORMWATER_FILE_URL = (
    "https://data.cityofnewyork.us/api/views/{dataset}/files/{blob}"
    "?download=true&filename={filename}"
)
# Scenario geodatabases used for the comparison (substring match on the .gdb name).
STORMWATER_MODERATE_KEY = "Moderate_Flood_2_13_inches_per_hr_with_Current_Sea_Levels"
STORMWATER_EXTREME_KEY = "Extreme_Flood_3_66_inches_per_hr_with_2080_Sea_Level_Rise"

# ---------------------------------------------------------------------------
# Synthetic fallbacks (tests / offline demos only)
# ---------------------------------------------------------------------------
# Silent synthetic data previously masked real download failures and produced a
# dataset of fake terrain.  They are now OFF unless explicitly requested.
ALLOW_SYNTHETIC = os.getenv("FLOODMAP_ALLOW_SYNTHETIC", "0") == "1"
