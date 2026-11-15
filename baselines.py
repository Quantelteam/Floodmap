"""
baselines.py - baseline models, evaluation and figures for FloodMap.

Reads data/processed/features.parquet (built by make_dataset.py) and writes
everything the documentation needs into reports/:

    reports/metrics.csv            every metric, with a 95% interval
    reports/comparisons.csv        how often each model beats the rule baseline
    reports/summary.json           dataset counts used in the write-up
    reports/test_predictions.csv   one row per (cell, test storm) with every score
    reports/val_predictions.csv    the same for the validation storms
    reports/per_storm.csv          recall in the top 10% for each test storm
    reports/by_area.csv            recall in the top 10% by area / reporting density
    reports/failure_cases.csv      the worst misses and the most persistent false alarms
    reports/feature_importance.csv permutation importance of the tree model
    reports/figures/*.png          the figures

Usage
-----
    python baselines.py                 # fit, evaluate, draw (about 3-5 minutes)
    python baselines.py --plots-only    # redraw from the saved predictions

Two evaluation settings are reported, because they answer different questions:

    "same area"      train on the training storms, test on the newest storms.
                     Every grid cell appears in both, so a model can score well
                     simply by remembering which places get reported.
    "unseen blocks"  the same storm split, plus 1 km blocks held out in 6 folds:
                     each test row is scored by a model that never saw its block.
                     This is the "new storm in a place it has not seen" number.

Extra dependencies beyond requirements.txt: scikit-learn, matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_FOLDS = 6          # spatial folds over the 1 km blocks
TOP_FRAC = 0.10      # "a planner can inspect the riskiest 10% of cells"
N_BOOT = 2000        # bootstrap resamples (whole storms are resampled)

STATIC = [
    "elev", "slope", "curvature", "twi", "flow_accum", "depression_depth",
    "dist_water", "impervious_frac", "landcover_lowveg", "landcover_tree",
    "landcover_water",
]
RAIN = ["rain_total_mm", "rain_peak_1h_mm", "rain_prev_48h_mm"]
# Deliberately NOT used as model inputs:
#   lat, lon, cell_id, block_id   -> would let a model memorise locations
#   landcover_built               -> identical to impervious_frac in this dataset
#   in_*_flood_zone               -> official map, kept for comparison only
FEATURES = STATIC + RAIN

MODELS = {            # column prefix -> display name
    "rule": "Rule baseline",
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "gbm": "Gradient boosting",
    "hist": "Report history (reference)",
}
SETTINGS = {"storm": "New storms, same area", "block": "New storms, unseen 1 km blocks"}
COLORS = {"rule": "#777777", "logreg": "#4477AA", "rf": "#EE6677", "gbm": "#AA3377",
          "hist": "#CCBB44"}
SPLIT_COLORS = {"train": "#4477AA", "val": "#CCBB44", "test": "#EE6677"}

LANDMARKS = {         # approximate label positions (lat, lon); skipped if outside the grid
    "Upper\nNew York\nBay": (40.6610, -74.0190),
    "Green-Wood\nCemetery": (40.6525, -73.9915),
    "Prospect\nPark": (40.6640, -73.9690),
    "Sunset Park": (40.6440, -74.0100),
    "Red Hook": (40.6765, -74.0100),
    "Park Slope": (40.6745, -73.9800),
    "Kensington": (40.6395, -73.9730),
}


# --------------------------------------------------------------------------- data
def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in FEATURES + ["cell_id", "event_id", "flooded", "split", "block_id",
                                      "lat", "lon", "start_date"] if c not in df.columns]
    if missing:
        raise ValueError(f"features file is missing columns: {missing}")
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per grid cell with its static values, sorted by cell_id."""
    keep = ["cell_id", "lat", "lon", "block_id"] + STATIC
    keep += [c for c in ("in_moderate_flood_zone", "in_extreme_flood_zone") if c in df.columns]
    return df[keep].drop_duplicates("cell_id").set_index("cell_id").sort_index()


def rule_score(cells: pd.DataFrame) -> pd.Series:
    """Hand-made score, no training: low ground, lots of inflow, wet, paved, in a dip."""
    pct = lambda s: s.rank(pct=True)
    return (pct(-cells["elev"]) + pct(cells["flow_accum"]) + pct(cells["twi"])
            + pct(cells["impervious_frac"]) + pct(cells["depression_depth"])) / 5.0


def block_folds(cells: pd.DataFrame) -> dict[int, int]:
    blocks = np.sort(cells["block_id"].unique())
    perm = np.random.RandomState(SEED).permutation(blocks)
    return {int(b): k for k, part in enumerate(np.array_split(perm, N_FOLDS)) for b in part}


# --------------------------------------------------------------------------- models
def make_model(name: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "rf":
        return RandomForestClassifier(n_estimators=1000, min_samples_leaf=5, max_features="sqrt",
                                      random_state=SEED, n_jobs=-1)
    if name == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              random_state=SEED)
    raise KeyError(name)


def fit_predict(df: pd.DataFrame, out: Path) -> None:
    from sklearn.inspection import permutation_importance

    cells = cell_table(df)
    fold_of = block_folds(cells)
    train = df[df["split"] == "train"]
    ev = df[df["split"].isin(["val", "test"])].copy()
    ev["fold"] = ev["block_id"].map(fold_of)

    ev["rule"] = ev["cell_id"].map(rule_score(cells)).values
    # reference only: how many training-period reports did this cell get?
    hist = train.groupby("cell_id")["flooded"].sum().reindex(cells.index).fillna(0)
    jitter = pd.Series(np.random.RandomState(SEED).uniform(0, 0.5, len(cells)), index=cells.index)
    ev["hist"] = ev["cell_id"].map(hist + jitter).values      # jitter only breaks ties
    ev["train_reports_cell"] = ev["cell_id"].map(hist).astype(int).values

    fitted = {}
    for name in ("logreg", "rf", "gbm"):
        print(f"fitting {name} ...", flush=True)
        model = make_model(name).fit(train[FEATURES], train["flooded"])
        fitted[name] = model
        ev[f"{name}_storm"] = model.predict_proba(ev[FEATURES])[:, 1]
        col = np.zeros(len(ev))
        for k in range(N_FOLDS):
            tr_k = train[train["block_id"].map(fold_of) != k]
            m = (ev["fold"] == k).values
            col[m] = make_model(name).fit(tr_k[FEATURES], tr_k["flooded"]).predict_proba(
                ev.loc[m, FEATURES])[:, 1]
        ev[f"{name}_block"] = col

    val = ev[ev["split"] == "val"]
    print("permutation importance ...", flush=True)
    imp = permutation_importance(fitted["rf"], val[FEATURES], val["flooded"],
                                 scoring="average_precision", n_repeats=5, random_state=SEED)
    pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean,
                  "std": imp.importances_std}).sort_values("importance", ascending=False).to_csv(
        out / "feature_importance.csv", index=False)

    keep = (["cell_id", "event_id", "start_date", "lat", "lon", "block_id", "fold", "flooded",
             "train_reports_cell", "rule", "hist"]
            + [f"{m}_{s}" for m in ("logreg", "rf", "gbm") for s in ("storm", "block")])
    for split in ("val", "test"):
        ev.loc[ev["split"] == split, keep].to_csv(out / f"{split}_predictions.csv", index=False,
                                                    float_format="%.6g")


# --------------------------------------------------------------------------- metrics
def add_ranks(pred: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """rank_<col> = position of the cell inside its storm, 1 = riskiest."""
    pred = pred.sort_values(["event_id", "cell_id"]).reset_index(drop=True)
    for c in cols:
        pred[f"rank_{c}"] = pred.groupby("event_id")[c].rank(method="first", ascending=False)
    pred["n_cells"] = pred.groupby("event_id")["cell_id"].transform("size")
    return pred


def flagged(pred: pd.DataFrame, col: str, frac: float = TOP_FRAC) -> np.ndarray:
    return (pred[f"rank_{col}"] <= np.ceil(frac * pred["n_cells"])).values


class Scorer:
    """PR-AUC and ROC-AUC with per-row weights, fast enough to bootstrap."""

    def __init__(self, y: np.ndarray, s: np.ndarray):
        order = np.argsort(-s, kind="stable")
        self.order = order
        ss = s[order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ss) != 0) + 1]
        self.y = y[order].astype(float)

    def __call__(self, w: np.ndarray | None = None) -> tuple[float, float]:
        w = np.ones(len(self.y)) if w is None else w[self.order]
        tp_g = np.add.reduceat(w * self.y, self.starts)
        n_g = np.add.reduceat(w, self.starts)
        fp_g = n_g - tp_g
        tp, fp = np.cumsum(tp_g), np.cumsum(fp_g)
        p, n = tp[-1], fp[-1]
        if p == 0 or n == 0:
            return np.nan, np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        ap = float(np.sum(prec * tp_g) / p)
        auc = float(np.sum(tp_g * (n - fp + 0.5 * fp_g)) / (p * n))
        return ap, auc


def evaluate(pred: pd.DataFrame, cols: list[str], seed: int = SEED):
    """Point estimates + storm-bootstrap intervals for every score column."""
    y = pred["flooded"].values
    storms, storm_idx = np.unique(pred["event_id"].values, return_inverse=True)
    rng = np.random.RandomState(seed)
    weights = rng.multinomial(len(storms), np.ones(len(storms)) / len(storms), size=N_BOOT)
    pos_e = np.bincount(storm_idx, weights=y, minlength=len(storms))

    rows, boots = [], {}
    for c in cols:
        sc = Scorer(y, pred[c].values)
        ap, auc = sc()
        hit_e = np.bincount(storm_idx, weights=y * flagged(pred, c), minlength=len(storms))
        rec = hit_e.sum() / pos_e.sum()
        b = np.empty((N_BOOT, 3))
        for i, w_e in enumerate(weights):
            b[i, :2] = sc(w_e[storm_idx].astype(float))
            b[i, 2] = (w_e * hit_e).sum() / max((w_e * pos_e).sum(), 1e-9)
        boots[c] = b
        for j, (metric, value) in enumerate([("pr_auc", ap), ("roc_auc", auc), ("recall_top10", rec)]):
            lo, hi = np.nanpercentile(b[:, j], [2.5, 97.5])
            rows.append(dict(score=c, metric=metric, value=value, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows), boots


def gain_curve(pred: pd.DataFrame, col: str):
    """x = share of cells inspected in each storm, y = share of all reports found."""
    pos = pred[pred["flooded"] == 1]
    x = np.sort((pos[f"rank_{col}"] / pos["n_cells"]).values)
    return np.r_[0, x, 1], np.r_[0, np.arange(1, len(x) + 1) / len(x), 1]


def pr_roc_points(y: np.ndarray, s: np.ndarray):
    from sklearn.metrics import precision_recall_curve, roc_curve
    p, r, _ = precision_recall_curve(y, s)
    f, t, _ = roc_curve(y, s)
    return (r, p), (f, t)


def split_col(model: str, setting: str) -> str:
    return model if model in ("rule", "hist") else f"{model}_{setting}"


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
        "legend.frameon": False, "figure.facecolor": "white", "axes.grid": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    })


class Grid:
    """Maps a per-cell value onto the 100 m grid for drawing."""

    def __init__(self, cells: pd.DataFrame):
        self.cells = cells
        lon = cells["lon"].values
        wrap = np.flatnonzero(np.diff(lon) < 0)
        self.ncols = int(wrap[0] + 1) if len(wrap) else len(cells)
        self.nrows = len(cells) // self.ncols
        self.extent = [0, self.ncols * 0.1, 0, self.nrows * 0.1]      # km
        self.water = self.arr(cells["landcover_water"]) > 0.5
        self.park = (self.arr(cells["landcover_tree"] + cells["landcover_lowveg"]) > 0.6) & ~self.water

    def arr(self, per_cell) -> np.ndarray:
        v = pd.Series(per_cell, index=self.cells.index) if not isinstance(per_cell, pd.Series) else per_cell
        return v.reindex(self.cells.index).values.astype(float).reshape(self.nrows, self.ncols)

    def xy(self, lat: float, lon: float):
        d = (self.cells["lat"] - lat) ** 2 + ((self.cells["lon"] - lon) * 0.76) ** 2
        i = int(np.argmin(d.values))
        if d.values[i] > 0.002 ** 2:
            return None
        return (i % self.ncols + 0.5) * 0.1, (i // self.ncols + 0.5) * 0.1

    def base(self, ax, title: str = "", labels: bool = False, parks: bool = True):
        from matplotlib.colors import ListedColormap
        ax.set_facecolor("#f3f1ec")
        if parks:
            ax.imshow(np.where(self.park, 1.0, np.nan), origin="lower", extent=self.extent,
                      cmap=ListedColormap(["#dfe8d6"]), interpolation="nearest", zorder=1)
        ax.imshow(np.where(self.water, 1.0, np.nan), origin="lower", extent=self.extent,
                  cmap=ListedColormap(["#c9dbe6"]), interpolation="nearest", zorder=4)
        ax.set_xlim(self.extent[:2]); ax.set_ylim(self.extent[2:])
        ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#999999"); s.set_linewidth(0.8)
        x1 = self.extent[1]                                      # scale bar + north arrow, under the map
        ax.plot([x1 - 1, x1], [-0.13, -0.13], color="#333333", lw=2, solid_capstyle="butt", clip_on=False, zorder=9)
        ax.text(x1 - 1.08, -0.13, "1 km", ha="right", va="center", fontsize=8, color="#333333", clip_on=False)
        ax.annotate("", xy=(x1 - 1.75, -0.02), xytext=(x1 - 1.75, -0.27), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
        ax.text(x1 - 1.84, -0.14, "N", ha="right", va="center", fontsize=8, color="#333333", clip_on=False)
        if labels:
            import matplotlib.patheffects as pe
            for name, (lat, lon) in LANDMARKS.items():
                p = self.xy(lat, lon)
                if p:
                    ax.text(*p, name, ha="center", va="center", fontsize=7.5, style="italic", color="#444444",
                            zorder=10, path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.8)])
        if title:
            ax.set_title(title)

    def layer(self, ax, arr, cmap, vmin=None, vmax=None, norm=None, alpha=1.0, zorder=3):
        a = np.where(self.water, np.nan, arr)
        return ax.imshow(a, origin="lower", extent=self.extent, cmap=cmap, vmin=vmin, vmax=vmax,
                         norm=norm, interpolation="nearest", alpha=alpha, zorder=zorder)


def make_figures(df: pd.DataFrame, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    _style()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cells = cell_table(df)
    grid = Grid(cells)
    fold_of = block_folds(cells)

    score_cols = ["rule", "hist"] + [f"{m}_{s}" for m in ("logreg", "rf", "gbm") for s in ("storm", "block")]
    test = add_ranks(pd.read_csv(out / "test_predictions.csv", parse_dates=["start_date"]), score_cols)
    val = add_ranks(pd.read_csv(out / "val_predictions.csv", parse_dates=["start_date"]), score_cols)

    # ---------------------------------------------------------------- metrics tables
    m_test, boots = evaluate(test, score_cols)
    m_val, _ = evaluate(val, score_cols)
    metrics = pd.concat([m_test.assign(split="test"), m_val.assign(split="val")], ignore_index=True)
    metrics["model"] = metrics["score"].str.replace("_storm|_block", "", regex=True)
    metrics["setting"] = np.where(metrics["score"].str.endswith("_block"), "block",
                                  np.where(metrics["score"].str.endswith("_storm"), "storm", "both"))
    metrics[["split", "setting", "model", "metric", "value", "ci_low", "ci_high"]].to_csv(
        out / "metrics.csv", index=False, float_format="%.5f")

    # tree model = the one with the better validation PR-AUC in the "same area" setting
    v = m_val[m_val["metric"] == "pr_auc"].set_index("score")["value"]
    tree = "rf" if v["rf_storm"] >= v["gbm_storm"] else "gbm"
    shown = ["rule", "logreg", tree]
    get = lambda col, metric, tab=m_test: tab[(tab["score"] == col) & (tab["metric"] == metric)].iloc[0]

    comp = []
    for setting in SETTINGS:
        for mdl in ("logreg", "rf", "gbm", "hist"):
            if mdl == "hist" and setting == "block":
                continue
            c = split_col(mdl, setting)
            for j, metric in enumerate(["pr_auc", "roc_auc", "recall_top10"]):
                d = boots[c][:, j] - boots["rule"][:, j]
                comp.append(dict(setting=setting, model=mdl, metric=metric, vs="rule",
                                 diff=get(c, metric)["value"] - get("rule", metric)["value"],
                                 diff_ci_low=np.nanpercentile(d, 2.5), diff_ci_high=np.nanpercentile(d, 97.5),
                                 share_of_resamples_better=float(np.nanmean(d > 0))))
    pd.DataFrame(comp).to_csv(out / "comparisons.csv", index=False, float_format="%.5f")

    prevalence = test["flooded"].mean()
    n_pos = int(test["flooded"].sum())
    n_storms_test = test["event_id"].nunique()

    # ---------------------------------------------------------------- 01 study area + reports
    pos = df[df["flooded"] == 1]
    per_cell = pos.groupby("cell_id").size().reindex(cells.index).fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    grid.base(ax, labels=True)
    bounds = [0.5, 1.5, 2.5, 5.5, 100]
    cmap = ListedColormap(["#fdcc8a", "#fc8d59", "#e34a33", "#7f0000"])
    grid.layer(ax, np.where(grid.arr(per_cell) > 0, grid.arr(per_cell), np.nan), cmap, norm=BoundaryNorm(bounds, 4))
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cmap.colors, ["1 storm", "2 storms", "3-5 storms", "6+ storms"])]
              + [Patch(color="#c9dbe6", label="water"), Patch(color="#dfe8d6", label="park / trees")],
              loc="upper left", bbox_to_anchor=(1.01, 1), title="Storms with a flood\nreport in the cell", alignment="left")
    n_any = int((per_cell > 0).sum())
    top5 = per_cell.sort_values(ascending=False).head(int(round(0.05 * len(cells)))).sum() / per_cell.sum()
    ax.set_title(f"Flood reports cluster in a few places: {n_any:,} of {len(cells):,} cells ever had one")
    fig.text(0.125, 0.05, va="top", s=f"{len(pos):,} flooded (cell, storm) pairs over {df['event_id'].nunique()} storms, "
             f"{df['start_date'].min():%b %Y} to {df['start_date'].max():%b %Y}. "
             f"The busiest 5% of cells hold {top5:.0%} of them.", fontsize=8.5, color="#444444")
    fig.savefig(fig_dir / "01_study_area_reports.png"); plt.close(fig)

    # ---------------------------------------------------------------- 02 elevation + flow accumulation
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    grid.base(axes[0], "Elevation (m)", parks=False)
    im = grid.layer(axes[0], grid.arr(cells["elev"]), "terrain", vmin=-5, vmax=cells["elev"].max())
    fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)
    grid.base(axes[1], "Flow accumulation (upstream cells, log scale)", parks=False)
    im = grid.layer(axes[1], grid.arr(cells["flow_accum"]), "Blues", norm=LogNorm(vmin=2, vmax=cells["flow_accum"].max()))
    fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02)
    fig.savefig(fig_dir / "02_elevation_flow.png"); plt.close(fig)

    # ---------------------------------------------------------------- 03 storm timeline
    ev = (df.groupby("event_id").agg(split=("split", "first"), start=("start_date", "first"),
                                     rain=("rain_total_mm", "first"), reports=("flooded", "sum"))
          .sort_values("start"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw=dict(hspace=0.12))
    for s, g in ev.groupby("split"):
        axes[0].bar(g["start"], g["rain"], width=9, color=SPLIT_COLORS[s], label=f"{s} ({len(g)} storms)")
        axes[1].bar(g["start"], g["reports"], width=9, color=SPLIT_COLORS[s])
    for a in axes:
        for s in ("val", "test"):
            a.axvline(ev[ev["split"] == s]["start"].min() - pd.Timedelta(days=5), color="#555555", lw=0.8, ls="--")
    axes[0].set_ylabel("Storm rain (mm)"); axes[1].set_ylabel("Cells with a report")
    axes[0].set_title(f"{len(ev)} storms, split by date: oldest for training, newest for testing")
    axes[0].legend(ncol=3, loc="upper right",
                   handles=[Patch(color=SPLIT_COLORS[s], label=f"{s} ({(ev['split'] == s).sum()} storms)") for s in ("train", "val", "test")])
    fig.savefig(fig_dir / "03_storm_timeline.png"); plt.close(fig)

    # ---------------------------------------------------------------- 04 class balance
    bal = df.groupby("split")["flooded"].agg(["mean", "sum", "size"]).reindex(["train", "val", "test"])
    full_train = (df[df["split"] == "train"]["event_id"].nunique() * len(cells))
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    labels = [f"train as shipped\n(negatives cut to 20:1)", "train before\ndownsampling", "validation", "test"]
    vals = [bal.loc["train", "mean"], bal.loc["train", "sum"] / full_train, bal.loc["val", "mean"], bal.loc["test", "mean"]]
    notes = [f"{int(bal.loc['train', 'sum']):,} of {int(bal.loc['train', 'size']):,} rows",
             f"{int(bal.loc['train', 'sum']):,} of {full_train:,} rows",
             f"{int(bal.loc['val', 'sum']):,} of {int(bal.loc['val', 'size']):,} rows",
             f"{int(bal.loc['test', 'sum']):,} of {int(bal.loc['test', 'size']):,} rows"]
    cols = [SPLIT_COLORS["train"], "#a9c0dc", SPLIT_COLORS["val"], SPLIT_COLORS["test"]]
    ax.barh(labels[::-1], [100 * x for x in vals[::-1]], color=cols[::-1], height=0.62)
    for i, (x, n) in enumerate(zip(vals[::-1], notes[::-1])):
        ax.text(100 * x + 0.06, i, f"{100 * x:.2f}%  ({n})", va="center", fontsize=9)
    ax.set_xlim(0, 7.2); ax.set_xlabel("Share of rows with a flood report (%)"); ax.grid(axis="y", visible=False)
    ax.set_title("Floods are rare: about 3 rows in 1,000 at the real rate")
    fig.savefig(fig_dir / "04_class_balance.png"); plt.close(fig)

    # ---------------------------------------------------------------- 05 split diagram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.08))
    ax = axes[0]
    ax.grid(False); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    for s in ("train", "val", "test"):
        g = ev[ev["split"] == s]
        ax.barh(0, (g["start"].max() - g["start"].min()).days + 20, left=g["start"].min() - pd.Timedelta(days=10),
                color=SPLIT_COLORS[s], height=0.5, alpha=0.9)
        mid = g["start"].min() + (g["start"].max() - g["start"].min()) / 2
        ax.text(mid, 0, f"{s}\n{len(g)} storms", ha="center", va="center", color="white" if s != "val" else "#222222",
                fontsize=9.5, fontweight="bold")
        ax.text(mid, -0.42, f"{g['start'].min():%b %Y} to\n{g['start'].max():%b %Y}", ha="center", va="top", fontsize=8.5, color="#444444")
    ax.plot(ev["start"], np.full(len(ev), 0.36), "|", color="#333333", ms=7)
    ax.text(ev["start"].min(), 0.48, "each tick = one storm", fontsize=8.5, color="#444444")
    ax.set_ylim(-0.95, 0.75); ax.set_title("1. By storm: a storm is never divided across sets")
    ax = axes[1]
    grid.base(ax, "2. By place: 1 km blocks held out in 6 folds", parks=False)
    fold_arr = grid.arr(cells["block_id"].map(fold_of))
    fold_cmap = ListedColormap(["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"])
    grid.layer(ax, fold_arr, fold_cmap, vmin=-0.5, vmax=5.5, alpha=0.75)
    blk = grid.arr(cells["block_id"])
    for c in np.flatnonzero(np.diff(blk[0]) != 0):
        ax.axvline((c + 1) * 0.1, color="white", lw=0.8, zorder=5)
    for r in np.flatnonzero(np.diff(blk[:, 0]) != 0):
        ax.axhline((r + 1) * 0.1, color="white", lw=0.8, zorder=5)
    ax.legend(handles=[Patch(color=fold_cmap.colors[k], alpha=0.75, label=f"fold {k + 1}") for k in range(N_FOLDS)],
              loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.text(0.56, -0.04, "A test cell is scored by a model that never saw any cell of its block.", fontsize=8.5, color="#444444")
    fig.savefig(fig_dir / "05_split_diagram.png"); plt.close(fig)

    # ---------------------------------------------------------------- 06 baseline comparison
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), gridspec_kw=dict(wspace=0.32))
    for ax, (metric, label) in zip(axes, [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("recall_top10", "Recall in the top 10% of cells")]):
        w = 0.26
        for i, mdl in enumerate(shown):
            for j, setting in enumerate(SETTINGS):
                r = get(split_col(mdl, setting), metric)
                x = j + (i - 1) * w
                ax.bar(x, r["value"], width=w * 0.92, color=COLORS[mdl], label=MODELS[mdl] if j == 0 else None)
                ax.errorbar(x, r["value"], yerr=[[r["value"] - r["ci_low"]], [r["ci_high"] - r["value"]]],
                            color="#333333", lw=1, capsize=2)
                ax.text(x, r["ci_high"], {"pr_auc": "{:.4f}", "roc_auc": "{:.3f}", "recall_top10": "{:.0%}"}[metric].format(r["value"]),
                        ha="center", va="bottom", fontsize=7.5)
        ref = {"pr_auc": prevalence, "roc_auc": 0.5, "recall_top10": TOP_FRAC}[metric]
        ax.axhline(ref, color="#222222", ls=":", lw=1)
        ax.text(1.01, ref, "no skill", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, color="#222222")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["same area", "unseen blocks"])
        ax.set_title(label); ax.grid(axis="x", visible=False)
        if metric == "roc_auc":
            ax.set_ylim(0.4, None)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.115, 1.06), ncol=3, fontsize=9)
    fig.suptitle(f"Test set: {n_storms_test} newest storms, {n_pos} flooded rows of {len(test):,}. Whiskers show the 95% range when whole storms are resampled.",
                 x=0.125, ha="left", fontsize=9, color="#444444", y=1.12)
    fig.savefig(fig_dir / "06_baseline_comparison.png"); plt.close(fig)

    # ---------------------------------------------------------------- 07 / 08 / 09 curves
    y = test["flooded"].values

    def curve_fig(kind, fname, title):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True, gridspec_kw=dict(wspace=0.08))
        for ax, (setting, sname) in zip(axes, SETTINGS.items()):
            for mdl in shown:
                c = split_col(mdl, setting)
                if kind == "pr":
                    (r, p), _ = pr_roc_points(y, test[c].values)
                    ax.step(r, p, where="post", color=COLORS[mdl], lw=1.6, label=f"{MODELS[mdl]} (PR-AUC {get(c, 'pr_auc')['value']:.4f})")
                elif kind == "roc":
                    _, (f, t) = pr_roc_points(y, test[c].values)
                    ax.plot(f, t, color=COLORS[mdl], lw=1.6, label=f"{MODELS[mdl]} (AUC {get(c, 'roc_auc')['value']:.3f})")
                else:
                    gx, gy = gain_curve(test, c)
                    ax.step(gx, gy, where="post", color=COLORS[mdl], lw=1.6,
                            label=f"{MODELS[mdl]} ({get(c, 'recall_top10')['value']:.0%} at 10%)")
            if kind == "pr":
                ax.axhline(prevalence, color="#222222", ls=":", lw=1, label=f"always 'no flood' ({prevalence:.2%})")
                ax.set_yscale("log"); ax.set_ylim(prevalence / 2.5, 0.5); ax.set_xlabel("Recall")
            elif kind == "roc":
                ax.plot([0, 1], [0, 1], color="#222222", ls=":", lw=1, label="no skill")
                ax.set_xlabel("False positive rate")
            else:
                ax.plot([0, 1], [0, 1], color="#222222", ls=":", lw=1, label="no skill")
                ax.axvline(TOP_FRAC, color="#999999", lw=0.8, ls="--")
                ax.set_xlabel("Share of cells inspected in each storm (riskiest first)")
            ax.set_title(sname); ax.legend(fontsize=8.2, loc="upper right" if kind == "pr" else "lower right")
        axes[0].set_ylabel({"pr": "Precision (log scale)", "roc": "True positive rate", "gain": "Share of reported floods found"}[kind])
        fig.suptitle(title, x=0.125, ha="left", fontsize=9, color="#444444", y=1.0)
        fig.savefig(fig_dir / fname); plt.close(fig)

    sub = f"Test set: {n_storms_test} newest storms, {n_pos} flooded rows of {len(test):,}"
    curve_fig("pr", "07_pr_curve.png", sub)
    curve_fig("roc", "08_roc_curve.png", sub)
    curve_fig("gain", "09_gain_curve.png", sub)

    # ---------------------------------------------------------------- 10 confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw=dict(wspace=0.35))
    conf = {}
    for ax, (setting, sname) in zip(axes, SETTINGS.items()):
        f = flagged(test, split_col(tree, setting))
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
        conf[setting] = dict(tp=tp, fp=fp, fn=fn, tn=tn)
        mat = np.array([[tp, fn], [fp, tn]])
        ax.imshow([[1, 0.35], [0.35, 0.05]], cmap="Blues", vmin=0, vmax=1.6)
        for (i, j), n in np.ndenumerate(mat):
            ax.text(j, i, f"{n:,}", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["flagged\n(top 10%)", "not flagged"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["flood\nreported", "no report"])
        ax.grid(False); ax.set_title(sname, fontsize=10)
        ax.set_xlabel(f"precision {tp / (tp + fp):.1%}, recall {tp / (tp + fn):.0%}", fontsize=9)
    fig.suptitle(f"{MODELS[tree]}, threshold = the riskiest 10% of cells in each test storm", x=0.06, ha="left", fontsize=9, color="#444444", y=1.04)
    fig.savefig(fig_dir / "10_confusion_matrix.png"); plt.close(fig)

    # ---------------------------------------------------------------- risk surface (unseen blocks)
    bcol = split_col(tree, "block")
    risk = (1 - (test.groupby("cell_id")[f"rank_{bcol}"].mean() - 1) / (len(cells) - 1)) * 100   # 100 = riskiest
    risk_pct = risk.rank(pct=True) * 100
    test_reports = test.groupby("cell_id")["flooded"].sum()

    # ---------------------------------------------------------------- 11 official map overlap
    zone = {}
    for z in ("moderate", "extreme"):
        col = f"in_{z}_flood_zone"
        if col in cells:
            inz = cells[col].astype(bool)
            share = float(inz.mean())
            k = int(round(share * len(cells)))
            model_top = risk.sort_values(ascending=False).head(k).index
            zone[z] = dict(cell_share=share,
                           reports_in_zone=float(test_reports[inz[inz].index].sum() / test_reports.sum()),
                           reports_in_model_same_budget=float(test_reports[model_top].sum() / test_reports.sum()),
                           overlap_of_model_with_zone=float(inz[model_top].mean()) if k else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw=dict(wspace=0.05))
    grid.base(axes[0], "FloodMap: riskiest 10% of cells", labels=True)
    grid.layer(axes[0], np.where(grid.arr(risk_pct) >= 90, 1.0, np.nan), ListedColormap(["#d7301f"]))
    grid.base(axes[1], "Official NYC Stormwater Flood Map", labels=True)
    if "in_extreme_flood_zone" in cells:
        grid.layer(axes[1], np.where(grid.arr(cells["in_extreme_flood_zone"].astype(float)) > 0, 1.0, np.nan), ListedColormap(["#6baed6"]))
        grid.layer(axes[1], np.where(grid.arr(cells["in_moderate_flood_zone"].astype(float)) > 0, 1.0, np.nan), ListedColormap(["#08519c"]), zorder=3.5)
    for ax in axes:
        rp = test_reports[test_reports > 0]
        pts = [((c % grid.ncols + 0.5) * 0.1, (c // grid.ncols + 0.5) * 0.1) for c in rp.index]
        ax.scatter(*zip(*pts), s=9, facecolor="none", edgecolor="#111111", lw=0.6, zorder=8)
    axes[0].legend(handles=[Patch(color="#d7301f", label="riskiest 10% of cells"),
                            plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#111111", ms=4, label="cell with a test-storm report")],
                   loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=8.5)
    axes[1].legend(handles=[Patch(color="#08519c", label="moderate scenario (2.13 in/hr)"), Patch(color="#6baed6", label="extreme scenario (3.66 in/hr)")],
                   loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=8.5)
    fig.savefig(fig_dir / "11_official_map_comparison.png"); plt.close(fig)

    # ---------------------------------------------------------------- 12 feature importance
    imp = pd.read_csv(out / "feature_importance.csv").sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color=[("#4477AA" if f in RAIN else "#EE6677") for f in imp["feature"]],
            error_kw=dict(lw=1, capsize=2, ecolor="#333333"))
    ax.axvline(0, color="#333333", lw=0.8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in validation PR-AUC when the feature is shuffled")
    ax.set_title(f"What the {MODELS[tree].lower()} relies on")
    ax.legend(handles=[Patch(color="#EE6677", label="place (terrain, land cover)"), Patch(color="#4477AA", label="storm (rain)")], loc="lower right")
    fig.savefig(fig_dir / "12_feature_importance.png"); plt.close(fig)

    # ---------------------------------------------------------------- 13 error map
    f = flagged(test, bcol)
    t2 = test.assign(flag=f)
    hits = t2[(t2["flag"]) & (t2["flooded"] == 1)].groupby("cell_id").size()
    miss = t2[(~t2["flag"]) & (t2["flooded"] == 1)].groupby("cell_id").size()
    flag_share = t2.groupby("cell_id")["flag"].mean()
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    grid.base(ax, labels=True)
    grid.layer(ax, np.where(grid.arr(flag_share) > 0.5, 1.0, np.nan), ListedColormap(["#f4c7a1"]))
    to_xy = lambda idx: list(zip(*[((c % grid.ncols + 0.5) * 0.1, (c // grid.ncols + 0.5) * 0.1) for c in idx]))
    if len(miss):
        ax.scatter(*to_xy(miss.index), s=16 + 10 * miss.values, marker="x", color="#111111", lw=1.1, zorder=8)
    if len(hits):
        ax.scatter(*to_xy(hits.index), s=16 + 10 * hits.values, marker="o", color="#1b7837", edgecolor="white", lw=0.5, zorder=9)
    n_fp = int((f & (y == 0)).sum())
    ax.legend(handles=[Patch(color="#f4c7a1", label=f"flagged in most storms\n({n_fp:,} flagged rows had no report)"),
                       plt.Line2D([], [], marker="o", ls="", color="#1b7837", ms=6, label=f"report inside the flagged 10% ({int(hits.sum())})"),
                       plt.Line2D([], [], marker="x", ls="", color="#111111", ms=6, label=f"report missed ({int(miss.sum())})")],
              loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"Where the model is wrong: {int(miss.sum())} of {n_pos} test reports fall outside the flagged cells")
    fig.text(0.125, 0.05, va="top", s=f"{MODELS[tree]}, unseen blocks, riskiest 10% of cells flagged in each of the {n_storms_test} test storms.", fontsize=8.5, color="#444444")
    fig.savefig(fig_dir / "13_error_map.png"); plt.close(fig)

    # ---------------------------------------------------------------- 14 by area / reporting density
    train = df[df["split"] == "train"]
    blk_reports = train.groupby("block_id")["flooded"].sum().reindex(np.sort(cells["block_id"].unique())).fillna(0)
    terc = pd.qcut(blk_reports.rank(method="first"), 3, labels=["low", "medium", "high"])
    t2["density"] = t2["block_id"].map(terc)
    half_r, half_c = grid.nrows / 2, grid.ncols / 2
    t2["quadrant"] = np.where(t2["cell_id"] // grid.ncols >= half_r, "north", "south") + "-" + np.where(t2["cell_id"] % grid.ncols >= half_c, "east", "west")
    t2["flag_rule"] = flagged(test, "rule")
    rows = []
    for group, order in (("density", ["low", "medium", "high"]), ("quadrant", ["north-west", "north-east", "south-west", "south-east"])):
        for gname in order:
            g = t2[t2[group] == gname]
            p = g[g["flooded"] == 1]
            rows.append(dict(grouping=group, group=gname, cells=g["cell_id"].nunique(), test_reports=len(p),
                             recall_tree=p["flag"].mean() if len(p) else np.nan,
                             recall_rule=p["flag_rule"].mean() if len(p) else np.nan,
                             share_of_cells_flagged_tree=g["flag"].mean()))
    by_area = pd.DataFrame(rows)
    by_area.to_csv(out / "by_area.csv", index=False, float_format="%.4f")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True, gridspec_kw=dict(wspace=0.08, width_ratios=[3, 4]))
    for ax, (group, title) in zip(axes, [("density", "By past reporting in the 1 km block"), ("quadrant", "By part of the study area")]):
        g = by_area[by_area["grouping"] == group].reset_index(drop=True)
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["recall_rule"], 0.38, color=COLORS["rule"], label=MODELS["rule"])
        ax.bar(x + 0.2, g["recall_tree"], 0.38, color=COLORS[tree], label=f"{MODELS[tree]} (unseen blocks)")
        ax.axhline(TOP_FRAC, color="#222222", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({r} reports)" for n, r in zip(g["group"], g["test_reports"])])
        ax.set_title(title, fontsize=10); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Recall in the top 10% of cells")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].text(1.01, TOP_FRAC, "no skill", transform=axes[1].get_yaxis_transform(), va="center", ha="left", fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.115, 1.08), ncol=2, fontsize=9)
    fig.savefig(fig_dir / "14_performance_by_area.png"); plt.close(fig)

    # ---------------------------------------------------------------- 15 per storm
    ps = (t2.groupby("event_id").apply(lambda g: pd.Series(dict(
        date=g["start_date"].iloc[0], reports=int(g["flooded"].sum()),
        found_tree=int((g["flag"] & (g["flooded"] == 1)).sum()),
        found_rule=int((g["flag_rule"] & (g["flooded"] == 1)).sum()))), include_groups=False).sort_values("date"))
    rain = df.drop_duplicates("event_id").set_index("event_id")[["rain_total_mm", "rain_peak_1h_mm"]]
    ps = ps.join(rain)
    ps["recall_tree"] = ps["found_tree"] / ps["reports"]; ps["recall_rule"] = ps["found_rule"] / ps["reports"]
    ps.to_csv(out / "per_storm.csv", float_format="%.4f")
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    x = np.arange(len(ps))
    ax.bar(x, ps["reports"], color="#dddddd", label="reports not found")
    ax.bar(x, ps["found_tree"], color=COLORS[tree], label=f"found in the flagged 10% ({MODELS[tree].lower()}, unseen blocks)")
    for i, r in enumerate(ps.itertuples()):
        ax.text(i, r.reports + 0.4, f"{r.found_tree}/{r.reports}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{d.day} {d:%b}\n{d:%Y}\n{mm:.0f} mm" for d, mm in zip(ps["date"], ps["rain_total_mm"])], fontsize=8)
    ax.set_ylabel("Cells with a flood report"); ax.grid(axis="x", visible=False); ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Per test storm: most storms have under 10 reports, so each score is noisy")
    fig.savefig(fig_dir / "15_performance_by_storm.png"); plt.close(fig)

    # ---------------------------------------------------------------- 16 calibration
    n_tr_pos = int(train["flooded"].sum()); n_tr_neg_kept = int((train["flooded"] == 0).sum())
    beta = n_tr_neg_kept / (train["event_id"].nunique() * len(cells) - n_tr_pos)     # share of negatives kept
    p_raw = test[bcol].values
    p_adj = p_raw * beta / (p_raw * beta + 1 - p_raw)
    bins = pd.qcut(pd.Series(p_adj).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame(dict(p=p_adj, y=y, b=bins)).groupby("b").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"), k=("y", "sum"))
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    lim = [1e-4, 0.1]
    ax.plot(lim, lim, color="#222222", ls=":", lw=1, label="perfectly calibrated")
    ax.plot(cal["pred"], cal["obs"].clip(lower=lim[0]), "o-", color=COLORS[tree], label=f"{MODELS[tree]} (unseen blocks)")
    for r in cal.itertuples():
        ax.annotate(f"{int(r.k)}", (r.pred, max(r.obs, lim[0])), textcoords="offset points", xytext=(5, -9), fontsize=7.5, color="#444444")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Predicted chance of a report (corrected for 20:1 downsampling)")
    ax.set_ylabel("Observed share of rows with a report"); ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Calibration by tenths of the test rows")
    fig.text(0.13, -0.02, "Numbers = reports in each tenth. A bin with 0 reports is drawn on the bottom edge.", fontsize=8, color="#444444")
    fig.savefig(fig_dir / "16_calibration.png"); plt.close(fig)

    # ---------------------------------------------------------------- 17 risk map
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    grid.base(ax, labels=True, parks=False)
    bounds = [0, 50, 75, 90, 95, 100]
    rcmap = ListedColormap(["#fff7ec", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"])
    grid.layer(ax, grid.arr(risk_pct), rcmap, norm=BoundaryNorm(bounds, 5))
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(rcmap.colors[::-1], ["riskiest 5%", "next 5%", "next 15%", "next 25%", "lower half"])]
              + [Patch(color="#c9dbe6", label="water")], loc="upper left", bbox_to_anchor=(1.01, 1), title="Relative risk", alignment="left")
    ax.set_title("FloodMap relative risk of a street-flooding report, 100 m cells")
    fig.text(0.125, 0.05, va="top", s=f"{MODELS[tree]}, each cell scored by a model that never saw its 1 km block; average rank over the {n_storms_test} test storms.\n"
             "A ranking of cells against each other, not a probability and not an official flood forecast.", fontsize=8.5, color="#444444")
    fig.savefig(fig_dir / "17_risk_map.png"); plt.close(fig)

    # ---------------------------------------------------------------- 18 pipeline diagram
    fig, ax = plt.subplots(figsize=(11, 4.7)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.75)

    def box(x, yy, w, h, text, fc="#ffffff", bold=False):
        ax.add_patch(FancyBboxPatch((x, yy), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#555555", lw=1))
        ax.text(x + w / 2, yy + h / 2, text, ha="center", va="center", fontsize=8.6, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color="#555555", lw=1))

    srcs = [("NYC 311 flood reports", 3.6), ("Rain: Central Park\n(daily + hourly)", 2.7), ("LiDAR elevation, 2017", 1.8), ("ESA WorldCover\nland cover, 10 m", 0.9), ("NYC Stormwater\nFlood Maps", 0.0)]
    steps = [("clean, snap to cells,\n1 report per cell per day", 3.6), ("storms: rain >= 10 mm\nand >= 3 reports", 2.7), ("5 m terrain features,\naveraged to 100 m", 1.8), ("built / tree / water share,\ndistance to water", 0.9), ("flag cells inside\nofficial zones", 0.0)]
    for (t, yy), (t2_, _) in zip(srcs, steps):
        box(0.1, yy + 0.05, 2.2, 0.7, t, fc="#eef3f8")
        box(2.9, yy + 0.05, 2.5, 0.7, t2_)
        arrow(2.3, yy + 0.4, 2.9, yy + 0.4)
    box(6.1, 1.55, 2.0, 1.5, f"features.parquet\n\n{len(cells):,} cells x {df['event_id'].nunique()} storms\nlabel + split", fc="#fdebd3", bold=False)
    for yy in (3.6, 2.7, 1.8, 0.9):
        arrow(5.4, yy + 0.4, 6.1, 2.3 + (yy - 2.25) * 0.25)
    box(8.8, 2.75, 2.1, 0.9, "3 baselines\nrule, logistic, tree")
    box(8.8, 1.45, 2.1, 0.9, "evaluation on the\nnewest storms")
    box(8.8, 0.05, 2.1, 0.7, "comparison only,\nnever a model input", fc="#f3f3f3")
    arrow(8.1, 2.6, 8.8, 3.1); arrow(9.85, 2.75, 9.85, 2.35); arrow(5.4, 0.4, 8.8, 0.4); arrow(9.85, 0.75, 9.85, 1.45)
    ax.text(0.1, 4.72, "make_dataset.py", fontsize=9, fontweight="bold", va="top")
    ax.text(8.8, 4.0, "baselines.py", fontsize=9, fontweight="bold", va="top")
    fig.savefig(fig_dir / "18_pipeline.png"); plt.close(fig)

    # ---------------------------------------------------------------- failure cases
    pos_t = t2[t2["flooded"] == 1].copy()
    pos_t["risk_percentile_in_storm"] = (1 - (pos_t[f"rank_{bcol}"] - 1) / (pos_t["n_cells"] - 1)) * 100
    static_cols = ["elev", "slope", "flow_accum", "twi", "impervious_frac", "landcover_tree", "dist_water"]
    worst = pos_t.sort_values("risk_percentile_in_storm").head(8).merge(cells[static_cols], left_on="cell_id", right_index=True)
    worst = worst.assign(kind="missed report")[["kind", "cell_id", "lat", "lon", "event_id", "start_date", "risk_percentile_in_storm", "train_reports_cell"] + static_cols]
    never = per_cell[per_cell == 0].index
    fa = (pd.DataFrame(dict(risk_percentile=risk_pct)).loc[lambda d: d.index.isin(never)].sort_values("risk_percentile", ascending=False).head(8)
          .join(cells[["lat", "lon"] + static_cols]).reset_index().assign(kind="flagged, never reported in any storm"))
    pd.concat([worst, fa], ignore_index=True).to_csv(out / "failure_cases.csv", index=False, float_format="%.5g")

    # ---------------------------------------------------------------- summary
    def row(model, setting):
        c = split_col(model, setting)
        return {m: dict(value=float(get(c, m)["value"]), ci=[float(get(c, m)["ci_low"]), float(get(c, m)["ci_high"])])
                for m in ("pr_auc", "roc_auc", "recall_top10")}

    summary = dict(
        cells=len(cells), grid=[grid.nrows, grid.ncols], area_km2=len(cells) * 0.01, blocks=int(cells["block_id"].nunique()),
        water_cells=int(grid.water.sum()), storms=int(df["event_id"].nunique()),
        first_storm=str(df["start_date"].min().date()), last_storm=str(df["start_date"].max().date()),
        rows_total=len(df), flooded_rows=int(df["flooded"].sum()),
        rows_before_downsampling=int(df["event_id"].nunique() * len(cells)),
        splits={s: dict(rows=int(len(g)), flooded=int(g["flooded"].sum()), storms=int(g["event_id"].nunique()),
                        first=str(g["start_date"].min().date()), last=str(g["start_date"].max().date()))
                for s, g in df.groupby("split")},
        cells_ever_reported=n_any, share_reports_in_busiest_5pct_cells=float(top5),
        test_prevalence=float(prevalence), tree_model=tree,
        validation_pr_auc_same_area=dict(rf=float(v["rf_storm"]), gbm=float(v["gbm_storm"])),
        test={s: {m: row(m, s) for m in (["rule", "logreg", "rf", "gbm"] + (["hist"] if s == "storm" else []))} for s in SETTINGS},
        confusion_top10=conf, official_map=zone,
        share_test_reports_in_cells_with_training_report=float(pos_t["train_reports_cell"].gt(0).mean()),
        cells_with_training_report=int((train.groupby("cell_id")["flooded"].sum() > 0).sum()),
        downsampling_negative_share_kept=float(beta),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--plots-only", action="store_true", help="reuse saved predictions, only redraw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_features(Path(args.features))
    if not args.plots_only or not (out / "test_predictions.csv").exists():
        fit_predict(df, out)
    s = make_figures(df, out)
    print(f"\nTree model chosen on validation PR-AUC: {MODELS[s['tree_model']]}")
    for setting, name in SETTINGS.items():
        print(f"\n{name}")
        for m, r in s["test"][setting].items():
            print(f"  {MODELS[m]:28s} PR-AUC {r['pr_auc']['value']:.4f}   ROC-AUC {r['roc_auc']['value']:.3f}   "
                  f"recall@10% {r['recall_top10']['value']:.3f}")
    print(f"\nEverything is in {out}/")


if __name__ == "__main__":
    main()
