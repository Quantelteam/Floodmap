"""
simulator.py - interactive flood simulation map for FloodMap.

Trains a gradient-boosting model on data/processed/features.parquet in which
rain can only push risk UP (monotonic constraints), scores every 100 m cell for
a sweep of rain amounts, and writes one self-contained web page:

    reports/simulator.html   dark map of the study area, a rain slider,
                             play button, "wet ground" and "2050 climate" switches

Usage
-----
    python simulator.py            # build the page and open it in the browser
    python simulator.py --no-open  # build only

Extra dependencies beyond requirements.txt: scikit-learn.
The colours are a relative risk index of a street-flooding report, not an
official forecast.
"""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
STATIC = [
    "elev", "slope", "curvature", "twi", "flow_accum", "depression_depth",
    "dist_water", "impervious_frac", "landcover_lowveg", "landcover_tree",
    "landcover_water",
]
RAIN = ["rain_total_mm", "rain_peak_1h_mm", "rain_prev_48h_mm"]
FEATURES = STATIC + RAIN

RAIN_LEVELS = list(range(10, 205, 5))   # event rainfall, mm
WET_GROUND_MM = 8.0                     # rain in the 48 h before the storm when "wet ground" is on
CLIMATE_FACTOR = 1.2                    # 2050 scenario: storms ~20% heavier
N_CLASSES = 6                           # 0 = minimal ... 5 = severe
RISK_EDGES = [0.007, 0.016, 0.035, 0.07, 0.15]  # probability cut points between the 6 classes

# Same neighbourhood labels as the figures in baselines.py
try:
    from baselines import LANDMARKS
except Exception:  # pragma: no cover - keep the simulator standalone
    LANDMARKS = {
        "Upper\nNew York\nBay": (40.6610, -74.0190),
        "Green-Wood\nCemetery": (40.6525, -73.9915),
        "Prospect\nPark": (40.6640, -73.9690),
        "Sunset Park": (40.6440, -74.0100),
        "Red Hook": (40.6765, -74.0100),
        "Park Slope": (40.6745, -73.9800),
        "Kensington": (40.6395, -73.9730),
    }
PARK_THRESHOLD = 0.6   # tree + low vegetation share, same rule as baselines.py

# Known NYC storms for reference marks on the slider (approximate Central Park totals)
REFERENCE_STORMS = [
    {"mm": 40, "label": "Typical storm"},
    {"mm": 90, "label": "Sep 29, 2023"},
    {"mm": 180, "label": "Hurricane Ida 2021"},
]


def peak_from_total(total: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Estimate peak 1-hour rain from event total using the storms in the data."""
    ev = df.drop_duplicates("event_id")
    a, b = np.polyfit(ev["rain_total_mm"], ev["rain_peak_1h_mm"], 1)
    return np.clip(a * total + b, 1.0, None)


def train_model(df: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingClassifier

    # +1 = more of this feature can only raise risk; 0 = free
    mono = [0] * len(STATIC) + [1, 1, 1]
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, monotonic_cst=mono, random_state=SEED,
    )
    model.fit(df[FEATURES], df["flooded"])
    return model


def simulate(df: pd.DataFrame, model) -> dict:
    cells = df.drop_duplicates("cell_id").set_index("cell_id").sort_index()
    X = cells[STATIC].copy()
    levels = np.array(RAIN_LEVELS, dtype=float)
    peaks = peak_from_total(levels, df)

    probs = {}  # (wet, level_idx) -> array of probabilities
    for wet in (0, 1):
        for i, (tot, pk) in enumerate(zip(levels, peaks)):
            Xs = X.copy()
            Xs["rain_total_mm"] = tot
            Xs["rain_peak_1h_mm"] = pk
            Xs["rain_prev_48h_mm"] = WET_GROUND_MM if wet else 0.0
            probs[(wet, i)] = model.predict_proba(Xs[FEATURES])[:, 1]

    # one fixed, roughly logarithmic colour scale for ALL scenarios, so the map
    # visibly heats up with rain (model probability of a flooding report per cell)
    edges = np.array(RISK_EDGES)

    water = (cells["landcover_water"] > 0.5).to_numpy()
    park = ((cells["landcover_tree"] + cells["landcover_lowveg"]) > PARK_THRESHOLD).to_numpy() & ~water
    scen = {}
    for (wet, i), p in probs.items():
        cls = np.searchsorted(edges, p, side="right").astype(int)
        cls[park] = -2
        cls[water] = -1
        scen[f"{wet}_{RAIN_LEVELS[i]}"] = "".join(
            "w" if c == -1 else "p" if c == -2 else str(c) for c in cls)

    for z in ("in_moderate_flood_zone", "in_extreme_flood_zone"):
        if z not in cells.columns:
            cells[z] = False
    cell_info = [
        [round(float(r.lat), 6), round(float(r.lon), 6), round(float(r.elev), 1),
         round(float(r.impervious_frac) * 100), int(bool(r.in_moderate_flood_zone)),
         int(bool(r.in_extreme_flood_zone))]
        for r in cells.itertuples()
    ]

    # reported floods in the historical data, for the "real reports" layer
    hist = df[df["flooded"] == 1].groupby("cell_id").size()
    reports = [[round(float(cells.at[c, "lat"]), 6), round(float(cells.at[c, "lon"]), 6), int(n)]
               for c, n in hist.items() if c in cells.index]

    return {
        "levels": RAIN_LEVELS,
        "peaks": [round(float(p), 1) for p in peaks],
        "scenarios": scen,
        "cells": cell_info,
        "reports": reports,
        "climate": CLIMATE_FACTOR,
        "wet_mm": WET_GROUND_MM,
        "refs": REFERENCE_STORMS,
        "n_storms": int(df["event_id"].nunique()),
        "n_cells": int(len(cells)),
        "n_land": int((~water & ~park).sum()),
        "landmarks": [[k, v[0], v[1]] for k, v in LANDMARKS.items()],
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FloodMap Simulator</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{
  --bg:#070b14; --panel:rgba(13,19,33,.86); --line:rgba(120,160,255,.14);
  --text:#e6ecff; --muted:#8b97b8; --accent:#4cc9f0; --accent2:#7b5cff;
  --c0:#1b2a4a; --c1:#2e6f9e; --c2:#f9c74f; --c3:#f8961e; --c4:#f3722c; --c5:#e5383b;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif;overflow:hidden}
#map{position:absolute;inset:0;background:var(--bg)}
.leaflet-container{background:var(--bg);font-family:Inter,sans-serif}
.leaflet-control-attribution{background:rgba(7,11,20,.7)!important;color:#5b6787!important}
.leaflet-control-attribution a{color:#7c8bb3!important}
.leaflet-control-zoom a{background:var(--panel)!important;color:var(--text)!important;border-color:var(--line)!important}

canvas#rain{position:absolute;inset:0;pointer-events:none;z-index:450;transition:opacity .4s}

.panel{position:absolute;z-index:500;background:var(--panel);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border:1px solid var(--line);border-radius:18px;box-shadow:0 20px 60px rgba(0,0,0,.45)}

#head{top:18px;left:18px;padding:16px 20px;max-width:360px}
#head .brand{display:flex;align-items:center;gap:10px}
#head .logo{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:grid;place-items:center;font-size:18px;box-shadow:0 0 24px rgba(76,201,240,.45)}
#head h1{font-size:19px;font-weight:800;letter-spacing:-.02em}
#head h1 span{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
#head p{color:var(--muted);font-size:12.5px;margin-top:8px;line-height:1.5}

#stats{top:18px;right:18px;padding:16px 18px;width:250px}
.stat{display:flex;justify-content:space-between;align-items:baseline;padding:7px 0;border-bottom:1px solid var(--line)}
.stat:last-child{border:none}
.stat .k{color:var(--muted);font-size:12px}
.stat .v{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:18px;transition:color .3s}
.stat .v.hot{color:var(--c5)}
.level{margin-top:10px;padding:10px 12px;border-radius:12px;font-weight:700;font-size:13px;text-align:center;
  letter-spacing:.06em;text-transform:uppercase;transition:all .35s}

#legend{bottom:200px;right:18px;padding:14px 16px;width:250px}
#legend h3,#ctrl h3{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;font-weight:600}
.lg{display:flex;align-items:center;gap:10px;font-size:12.5px;margin:5px 0}
.sw{width:16px;height:16px;border-radius:5px}
.toggle-row{display:flex;align-items:center;justify-content:space-between;margin-top:8px;font-size:12.5px;color:var(--text)}

#ctrl{left:50%;transform:translateX(-50%);bottom:22px;padding:18px 24px 16px;width:min(760px,calc(100% - 36px))}
.row{display:flex;align-items:center;gap:18px}
#rainval{font-family:'JetBrains Mono',monospace;font-size:38px;font-weight:700;min-width:150px;line-height:1}
#rainval small{font-size:15px;color:var(--muted);font-weight:500}
#rainsub{font-size:12px;color:var(--muted);margin-top:4px}
.sliderwrap{flex:1;position:relative;padding-top:4px}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:10px;border-radius:99px;outline:none;cursor:pointer;
  background:linear-gradient(90deg,#1e3a5f 0%,var(--accent) 35%,var(--c3) 70%,var(--c5) 100%)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:26px;height:26px;border-radius:50%;background:#fff;
  border:4px solid var(--accent2);box-shadow:0 0 0 6px rgba(123,92,255,.25),0 4px 14px rgba(0,0,0,.5);cursor:grab}
input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;background:#fff;border:4px solid var(--accent2);
  box-shadow:0 0 0 6px rgba(123,92,255,.25);cursor:grab}
.ticks{position:relative;height:30px;margin-top:6px}
.tick{position:absolute;transform:translateX(-50%);font-size:10.5px;color:var(--muted);text-align:center;white-space:nowrap;cursor:pointer}
.tick::before{content:"";display:block;width:2px;height:7px;background:var(--muted);margin:0 auto 3px;opacity:.6}
.tick:hover{color:var(--text)}
.btns{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap}
button.b{background:rgba(255,255,255,.05);color:var(--text);border:1px solid var(--line);padding:9px 15px;border-radius:11px;
  font:600 13px Inter,sans-serif;cursor:pointer;display:flex;align-items:center;gap:8px;transition:all .2s}
button.b:hover{background:rgba(255,255,255,.1);border-color:rgba(120,160,255,.35)}
button.b.on{background:linear-gradient(135deg,rgba(76,201,240,.25),rgba(123,92,255,.3));border-color:var(--accent)}
button.b.play{background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;color:#06101f}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
button.on .dot{background:var(--accent);box-shadow:0 0 8px var(--accent)}

.leaflet-tooltip.tt{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;
  box-shadow:0 10px 30px rgba(0,0,0,.5);font:12px Inter,sans-serif;padding:8px 10px}
.leaflet-tooltip.tt::before{display:none}
.tt b{font-family:'JetBrains Mono',monospace}

.lm{color:#e6ecff;font:italic 600 12.5px Inter,sans-serif;text-align:center;white-space:pre;line-height:1.15;
  text-shadow:0 0 3px #070b14,0 0 6px #070b14,0 0 10px #070b14;pointer-events:none;transform:translate(-50%,-50%);letter-spacing:.01em}
.lm.water{color:#8fd3ff}
.lm.green{color:#a7e08a}
#sumbtn{margin-top:12px}
#summary{top:50%;left:50%;transform:translate(-50%,-50%);width:min(640px,calc(100% - 36px));max-height:calc(100% - 60px);
  overflow:auto;padding:0;display:none;z-index:900;background:rgba(5,8,15,.96)}
#summary.show{display:block;animation:pop .25s ease}
@keyframes pop{from{opacity:0;transform:translate(-50%,-48%)}to{opacity:1;transform:translate(-50%,-50%)}}
.term-bar{display:flex;align-items:center;gap:7px;padding:12px 14px;border-bottom:1px solid var(--line)}
.term-bar i{width:11px;height:11px;border-radius:50%;display:block}
.term-bar span{margin-left:10px;color:var(--muted);font:500 12px 'JetBrains Mono',monospace;flex:1}
.term-bar button{background:none;border:none;color:var(--muted);font-size:18px;cursor:pointer}
#sumtxt{font:500 13px/1.6 'JetBrains Mono',monospace;color:#cfe3ff;padding:16px 20px 20px;white-space:pre;overflow-x:auto}
#sumtxt .h{color:#4cc9f0;font-weight:700}
#sumtxt .g{color:#7ee787}
#sumtxt .y{color:#f9c74f}
#sumtxt .r{color:#ff7b72}
#sumtxt .m{color:#6e7b99}
@media (max-width:820px){
  #stats,#legend{display:none}
  #head{max-width:calc(100% - 36px)}
  #head p{display:none}
  #rainval{font-size:28px;min-width:110px}
}
</style>
</head>
<body>
<div id="map"></div>
<canvas id="rain"></canvas>

<div id="head" class="panel">
  <div class="brand"><div class="logo"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#06101f" stroke-width="2.4" stroke-linecap="round"><path d="M2 8c2.5-2 4.5-2 7 0s4.5 2 7 0 4.5-2 6 0"/><path d="M2 13c2.5-2 4.5-2 7 0s4.5 2 7 0 4.5-2 6 0"/><path d="M2 18c2.5-2 4.5-2 7 0s4.5 2 7 0 4.5-2 6 0"/></svg></div><h1>Flood<span>Map</span> Simulator</h1></div>
  <p>Brooklyn, NYC · 100 m cells · Machine-learning model trained on <b id="nst"></b> real storms and NYC 311 flooding reports.
  Move the slider to see where streets flood as the rain gets heavier.</p>
  <button class="b" id="sumbtn">▤ Final pipeline summary</button>
</div>

<div id="stats" class="panel">
  <div class="stat"><span class="k">High-risk cells</span><span class="v" id="sHigh">0</span></div>
  <div class="stat"><span class="k">Area at high risk</span><span class="v" id="sArea">0</span></div>
  <div class="stat"><span class="k">Severe cells</span><span class="v" id="sSev">0</span></div>
  <div class="stat"><span class="k">Peak 1-hour rain</span><span class="v" id="sPeak">0</span></div>
  <div class="level" id="lvl">—</div>
</div>

<div id="legend" class="panel">
  <h3>Flood risk</h3>
  <div class="lg"><div class="sw" style="background:var(--c5)"></div>Severe</div>
  <div class="lg"><div class="sw" style="background:var(--c4)"></div>High</div>
  <div class="lg"><div class="sw" style="background:var(--c3)"></div>Elevated</div>
  <div class="lg"><div class="sw" style="background:var(--c2)"></div>Moderate</div>
  <div class="lg"><div class="sw" style="background:var(--c1)"></div>Low</div>
  <div class="lg"><div class="sw" style="background:#5fa04e"></div>Park / trees</div>
  <div class="lg"><div class="sw" style="background:#2b5f8a"></div>Water</div>
  <div class="toggle-row" style="margin-top:12px">
    <button class="b" id="bRep" style="width:100%;justify-content:center"><span class="dot"></span>Real 311 reports</button>
  </div>
  <div class="toggle-row">
    <button class="b" id="bZone" style="width:100%;justify-content:center"><span class="dot"></span>Official NYC flood zone</button>
  </div>
</div>

<div id="summary" class="panel">
  <div class="term-bar"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i>
    <span>python make_dataset.py --step all</span><button id="sumclose" aria-label="Close">×</button></div>
  <div id="sumtxt"></div>
</div>

<div id="ctrl" class="panel">
  <h3>Storm rainfall</h3>
  <div class="row">
    <div>
      <div id="rainval">0<small> mm</small></div>
      <div id="rainsub">&nbsp;</div>
    </div>
    <div class="sliderwrap">
      <input type="range" id="slider" min="0" max="1" step="1" value="0">
      <div class="ticks" id="ticks"></div>
    </div>
  </div>
  <div class="btns">
    <button class="b play" id="bPlay">▶ Play storm</button>
    <button class="b" id="bWet"><span class="dot"></span>Wet ground (rained before)</button>
    <button class="b" id="bClim"><span class="dot"></span>Climate 2050 (+20% rain)</button>
    <button class="b on" id="bFx"><span class="dot"></span>Rain effect</button>
  </div>
</div>

<script>
const D = __DATA__;
const COLORS = ['#1b2a4a','#2e6f9e','#f9c74f','#f8961e','#f3722c','#e5383b'];
const FILL   = [0.10, 0.28, 0.55, 0.68, 0.78, 0.88];
const NAMES  = ['Minimal','Low','Moderate','Elevated','High','Severe'];
document.getElementById('nst').textContent = D.n_storms;

const map = L.map('map', {zoomControl:false, preferCanvas:true, zoomSnap:0.25, maxZoom:16}).setView([40.6575,-73.9925], 14);
L.control.zoom({position:'bottomleft'}).addTo(map);
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/';
L.tileLayer(ESRI + 'World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
  attribution:'Tiles &copy; Esri', maxZoom:16
}).addTo(map);

// 100 m cells
const dLat = 0.000449, dLon = 0.000593;
const renderer = L.canvas({padding:0.3});
const rects = D.cells.map((c, i) => {
  const r = L.rectangle([[c[0]-dLat, c[1]-dLon],[c[0]+dLat, c[1]+dLon]],
    {renderer, stroke:false, fillOpacity:0, fillColor:COLORS[0]}).addTo(map);
  r.bindTooltip(() => tip(i), {className:'tt', sticky:true});
  return r;
});
const bounds = L.latLngBounds(D.cells.map(c => [c[0], c[1]]));
const fit = () => map.fitBounds(bounds, innerWidth > 820
  ? {paddingTopLeft:[300,20], paddingBottomRight:[280,190]} : {paddingTopLeft:[10,90], paddingBottomRight:[10,230]});
fit();

// labels on top
L.tileLayer(ESRI + 'World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
  {maxZoom:16, pane:'shadowPane'}).addTo(map);

// layers
const repLayer = L.layerGroup(D.reports.map(r => L.circleMarker([r[0], r[1]],
  {renderer, radius:2.5 + Math.min(r[2], 8), color:'#fff', weight:1, fillColor:'#4cc9f0', fillOpacity:.75})));
const zoneLayer = L.layerGroup(D.cells.filter(c => c[4]).map(c =>
  L.rectangle([[c[0]-dLat, c[1]-dLon],[c[0]+dLat, c[1]+dLon]],
  {renderer, color:'#7b5cff', weight:1.2, fill:false, opacity:.9})));

// neighbourhood labels (same as the report figures)
const labelPane = map.createPane('labels'); labelPane.style.zIndex = 640; labelPane.style.pointerEvents = 'none';
D.landmarks.forEach(([name, lat, lon]) => {
  const cls = /Bay/.test(name) ? 'lm water' : /Park\n|Cemetery/.test(name) ? 'lm green' : 'lm';
  L.marker([lat, lon], {pane:'labels', interactive:false,
    icon: L.divIcon({className:'', html:`<div class="${cls}">${name}</div>`, iconSize:[0,0]})}).addTo(map);
});

// final pipeline summary (terminal style)
(function(){
  const s = D.summary, eq = '='.repeat(60), pad = (t, n) => String(t).padStart(n);
  const L1 = [];
  L1.push(`<span class="m">${eq}</span>`, `<span class="h">  FINAL PIPELINE SUMMARY</span>`, `<span class="m">${eq}</span>`);
  L1.push(`  Output file   : ${s.file}`);
  L1.push(`  File size     : ${s.size_mb} MB`);
  L1.push(`  Total rows    : <span class="y">${s.rows.toLocaleString('en-US')}</span>`);
  L1.push(`  Unique cells  : <span class="y">${s.cells}</span>  <span class="m">(${s.water} water, ${s.park} park)</span>`);
  L1.push(`  Unique events : <span class="y">${s.events}</span>  <span class="m">(${s.first} → ${s.last})</span>`);
  L1.push(`  Flooded rows  : <span class="r">${s.flooded.toLocaleString('en-US')}</span> (${s.flooded_pct.toFixed(3)}%)`);
  L1.push(`  NaN values    : <span class="g">${s.nan}</span>`);
  s.splits.forEach(([n, rows, fl, pct]) =>
    L1.push(`  ${n.padEnd(13)} : ${pad(rows.toLocaleString('en-US'), 8)} rows, <span class="r">${pad(fl, 6)}</span> flooded (${pct.toFixed(3)}%)`));
  if (s.models.length){
    L1.push(`<span class="m">${eq}</span>`, `<span class="h">  TEST: reports found in riskiest 10% of cells</span>`);
    s.models.forEach(([n, w, v, lo, hi]) =>
      L1.push(`  ${(n + (w ? ' / ' + w : '')).padEnd(36)}: <span class="${v >= 30 ? 'g' : v >= 15 ? 'y' : 'r'}">${pad(v, 3)}%</span>  <span class="m">(${lo}–${hi}%)</span>`));
  }
  L1.push(`<span class="m">${eq}</span>`, `  Simulator     : ${s.levels} rain levels × 2 ground conditions`, `<span class="m">${eq}</span>`);
  document.getElementById('sumtxt').innerHTML = L1.join('\n');
  const box = document.getElementById('summary');
  document.getElementById('sumbtn').onclick = () => box.classList.toggle('show');
  document.getElementById('sumclose').onclick = () => box.classList.remove('show');
  document.addEventListener('keydown', e => { if (e.code === 'Escape') box.classList.remove('show'); });
})();

let state = {idx:0, wet:0, clim:false, cur:null};

function effectiveMM(){ return D.levels[state.idx] * (state.clim ? D.climate : 1); }
function scenarioKey(){
  const mm = effectiveMM();
  let best = 0;
  D.levels.forEach((l, i) => { if (Math.abs(l - mm) < Math.abs(D.levels[best] - mm)) best = i; });
  return {key: state.wet + '_' + D.levels[best], li: best};
}
function tip(i){
  const c = D.cells[i], s = state.cur ? state.cur[i] : '0';
  const name = s === 'w' ? 'Water' : s === 'p' ? 'Park / trees' : NAMES[+s] + ' risk';
  const col = s === 'w' ? '#4cc9f0' : s === 'p' ? '#a7e08a' : COLORS[+s];
  return `<div style="font-weight:700;color:${col};margin-bottom:4px">${name}</div>
    Elevation <b>${c[2]} m</b><br>Paved surface <b>${c[3]}%</b>
    ${c[4] ? '<br><span style="color:#a594ff">In official NYC flood zone</span>' : ''}`;
}

function render(){
  const {key, li} = scenarioKey();
  const s = D.scenarios[key];
  state.cur = s;
  let high = 0, sev = 0;
  for (let i = 0; i < s.length; i++){
    const ch = s[i];
    if (ch === 'w'){ rects[i].setStyle({fillColor:'#2b5f8a', fillOpacity:.22}); continue; }
    if (ch === 'p'){ rects[i].setStyle({fillColor:'#5fa04e', fillOpacity:.38}); continue; }
    const k = +ch;
    if (k >= 4) high++;
    if (k === 5) sev++;
    rects[i].setStyle({fillColor:COLORS[k], fillOpacity:FILL[k]});
  }
  const mm = effectiveMM();
  document.getElementById('rainval').innerHTML = Math.round(mm) + '<small> mm</small>';
  document.getElementById('rainsub').textContent =
    (state.clim ? `${D.levels[state.idx]} mm today → +20% in 2050` : 'total rain in one storm') +
    (state.wet ? ' · wet ground' : '');
  animateNum('sHigh', high, v => v.toLocaleString());
  animateNum('sArea', high * 0.01, v => v.toFixed(2) + ' km²');
  animateNum('sSev', sev, v => v.toLocaleString());
  document.getElementById('sPeak').textContent = D.peaks[li] + ' mm/h';
  document.getElementById('sHigh').classList.toggle('hot', high > 150);
  const lvl = document.getElementById('lvl');
  const share = high / D.n_land;
  const [txt, bg, fg] = share < .02 ? ['Low danger','rgba(46,111,158,.25)','#8fd3ff']
                      : share < .08 ? ['Watch','rgba(249,199,79,.18)','#f9c74f']
                      : share < .2  ? ['Warning','rgba(248,150,30,.22)','#f8961e']
                      :               ['Flood emergency','rgba(229,56,59,.28)','#ff6b6e'];
  lvl.textContent = txt; lvl.style.background = bg; lvl.style.color = fg;
  rainIntensity = Math.min(1, mm / 180);
}

const animTimers = {};
function animateNum(id, target, fmt){
  const el = document.getElementById(id);
  const from = +(el.dataset.v || 0);
  cancelAnimationFrame(animTimers[id]);
  const t0 = performance.now();
  const step = t => {
    const p = Math.min(1, (t - t0) / 350), e = 1 - Math.pow(1 - p, 3);
    const v = from + (target - from) * e;
    el.textContent = fmt(Number.isInteger(target) ? Math.round(v) : v);
    if (p < 1) animTimers[id] = requestAnimationFrame(step); else el.dataset.v = target;
  };
  animTimers[id] = requestAnimationFrame(step);
}

// slider
const slider = document.getElementById('slider');
slider.max = D.levels.length - 1;
state.idx = D.levels.indexOf(40);
slider.value = state.idx;
slider.addEventListener('input', () => { state.idx = +slider.value; render(); });

const ticks = document.getElementById('ticks');
D.refs.forEach(r => {
  const i = D.levels.indexOf(r.mm);
  if (i < 0) return;
  const t = document.createElement('div');
  t.className = 'tick';
  t.style.left = (i / (D.levels.length - 1) * 100) + '%';
  t.innerHTML = r.label + '<br>' + r.mm + ' mm';
  t.onclick = () => { stopPlay(); state.idx = i; slider.value = i; render(); };
  ticks.appendChild(t);
});

// buttons
function tog(id, fn){ const b = document.getElementById(id); b.onclick = () => { b.classList.toggle('on'); fn(b.classList.contains('on')); }; }
tog('bWet', on => { state.wet = on ? 1 : 0; render(); });
tog('bClim', on => { state.clim = on; render(); });
tog('bRep', on => on ? repLayer.addTo(map) : map.removeLayer(repLayer));
tog('bZone', on => on ? zoneLayer.addTo(map) : map.removeLayer(zoneLayer));
tog('bFx', on => { rainOn = on; cvs.style.opacity = on ? 1 : 0; });

let playT = null;
const bPlay = document.getElementById('bPlay');
function stopPlay(){ clearInterval(playT); playT = null; bPlay.textContent = '▶ Play storm'; }
bPlay.onclick = () => {
  if (playT){ stopPlay(); return; }
  if (state.idx >= D.levels.length - 1) state.idx = 0;
  bPlay.textContent = '❚❚ Pause';
  playT = setInterval(() => {
    state.idx++; slider.value = state.idx; render();
    if (state.idx >= D.levels.length - 1) stopPlay();
  }, 260);
};
document.addEventListener('keydown', e => {
  if (e.code === 'Space'){ e.preventDefault(); bPlay.click(); }
  if (e.code === 'ArrowRight'){ stopPlay(); state.idx = Math.min(D.levels.length-1, state.idx+1); slider.value = state.idx; render(); }
  if (e.code === 'ArrowLeft'){ stopPlay(); state.idx = Math.max(0, state.idx-1); slider.value = state.idx; render(); }
});

// rain animation
const cvs = document.getElementById('rain'), ctx = cvs.getContext('2d');
let rainIntensity = 0, rainOn = true, drops = [];
function resize(){ cvs.width = innerWidth * devicePixelRatio; cvs.height = innerHeight * devicePixelRatio; }
addEventListener('resize', resize); resize();
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
function frame(){
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  if (rainOn && !reduce){
    const want = Math.floor(rainIntensity * 420);
    while (drops.length < want) drops.push(newDrop(true));
    if (drops.length > want) drops.length = want;
    ctx.strokeStyle = 'rgba(170,200,255,0.35)';
    ctx.lineWidth = 1.2 * devicePixelRatio;
    ctx.beginPath();
    for (const d of drops){
      ctx.moveTo(d.x, d.y); ctx.lineTo(d.x - d.l * .18, d.y + d.l);
      d.y += d.v; d.x -= d.v * .18;
      if (d.y > cvs.height) Object.assign(d, newDrop(false));
    }
    ctx.stroke();
  }
  requestAnimationFrame(frame);
}
function newDrop(any){
  const s = devicePixelRatio;
  return {x: Math.random() * cvs.width * 1.2, y: any ? Math.random() * cvs.height : -20 * s,
          l: (10 + Math.random() * 18) * s, v: (9 + Math.random() * 9) * s};
}
frame();
render();
</script>
</body>
</html>
"""


def pipeline_summary(df: pd.DataFrame, features_path: Path, data: dict) -> dict:
    """Numbers for the FINAL PIPELINE SUMMARY block (same layout as make_dataset.py)."""
    fl = df["flooded"]
    out = {
        "file": str(features_path),
        "size_mb": round(features_path.stat().st_size / 1e6, 2),
        "rows": int(len(df)),
        "cells": int(df["cell_id"].nunique()),
        "events": int(df["event_id"].nunique()),
        "flooded": int(fl.sum()),
        "flooded_pct": round(float(fl.mean() * 100), 3),
        "nan": int(df.isna().sum().sum()),
        "first": str(pd.to_datetime(df["start_date"]).min().date()),
        "last": str(pd.to_datetime(df["start_date"]).max().date()),
        "splits": [],
        "water": sum(c == "w" for c in data["scenarios"]["0_10"]),
        "park": sum(c == "p" for c in data["scenarios"]["0_10"]),
        "levels": len(RAIN_LEVELS),
        "models": [],
    }
    for name in ("train", "val", "test"):
        g = df[df["split"] == name]
        out["splits"].append([name, int(len(g)), int(g["flooded"].sum()),
                              round(float(g["flooded"].mean() * 100), 3), int(g["event_id"].nunique())])
    metrics = Path("reports/metrics.csv")
    if metrics.exists():   # written by baselines.py
        m = pd.read_csv(metrics)
        m = m[(m["split"] == "test") & (m["metric"] == "recall_top10")]
        label = {"rule": "Rule baseline", "logreg": "Logistic regression", "rf": "Random forest",
                 "gbm": "Gradient boosting", "hist": "Report history"}
        where = {"both": "", "storm": "same area", "block": "unseen blocks"}
        for r in m.itertuples():
            out["models"].append([label.get(r.model, r.model), where.get(r.setting, r.setting),
                                  round(r.value * 100), round(r.ci_low * 100), round(r.ci_high * 100)])
    return out


def print_summary(s: dict) -> None:
    line = "=" * 60
    print(f"\n{line}\n  FINAL PIPELINE SUMMARY\n{line}")
    print(f"  Output file   : {s['file']}")
    print(f"  File size     : {s['size_mb']} MB")
    print(f"  Total rows    : {s['rows']:,}")
    print(f"  Unique cells  : {s['cells']}  ({s['water']} water, {s['park']} park)")
    print(f"  Unique events : {s['events']}  ({s['first']} -> {s['last']})")
    print(f"  Flooded rows  : {s['flooded']:,} ({s['flooded_pct']:.3f}%)")
    print(f"  NaN values    : {s['nan']}")
    for name, rows, fl, pct, ev in s["splits"]:
        print(f"  {name:<13} : {rows:>8,} rows, {fl:>6} flooded ({pct:.3f}%)")
    if s["models"]:
        print(f"{line}\n  TEST: share of reports in riskiest 10% of cells")
        for name, where, v, lo, hi in s["models"]:
            print(f"  {(name + (' / ' + where if where else '')):<36}: {v:>3}%  ({lo}-{hi}%)")
    print(f"{line}\n  Simulator     : {s['levels']} rain levels x 2 ground conditions\n{line}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--out", default="reports/simulator.html")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(args.features)
    print(f"Loaded {len(df):,} rows, {df['cell_id'].nunique():,} cells, {df['event_id'].nunique()} storms")
    print("Training monotonic gradient-boosting model ...", flush=True)
    model = train_model(df)
    print(f"Simulating {len(RAIN_LEVELS)} rain levels x 2 ground conditions ...", flush=True)
    data = simulate(df, model)
    data["summary"] = pipeline_summary(df, Path(args.features), data)
    print_summary(data["summary"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(f"Done in {time.time() - t0:.1f}s -> {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
