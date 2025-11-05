"""
Batch metrics over PID vs RL across multiple subsets (csv_1..csv_5, figure_8).

- Each subset contains multiple TRIAL folders, each with log_flight1.csv.
- Converts PATS/ENU -> NED (planes are in NED).
- Uses only rows where drone_state_str == "ns_chasing_insect".
- Metrics computed only where bestinsect (evader) is INBOUNDS (n·(x-p0) >= margin).
- Aggregates per log, saves tidy CSV, and (if data exist) plots comparisons.

TLDR: This file is the same as the metric_inbounds_computation.py but then for a series of logs

Outputs
-------
analysis_out/metrics_all.csv
analysis_out/plots/*.png
"""

# TODO lots of repeat code, which is shared with metric_inbounds_computation.py unify if time permits

from __future__ import annotations
import os, glob, json, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ROOTS: Dict[str, str] = {
    "PID": "/Users/merlijnbroekers/Desktop/rl_acc_trials/PID",
    "RL": "/Users/merlijnbroekers/Desktop/rl_acc_trials/RL",
}
SUBSETS: List[str] = [
    "csv_1",
    "csv_2",
    "csv_3",
    "csv_4",
    "csv_5",
    "figure_8",
    "virtual_insect",
]
# SUBSETS: List[str] = ["csv_1", "csv_2", "csv_3", "csv_4", "csv_5", "figure_8"]
# SUBSETS: List[str] = ["virtual_insect"]

# Each TRIAL folder under a subset contains this file containing the PATS flight log
LOG_FILENAME: str = "log_flight1.csv"
DELIMS_TRY: List[str] = [";"]

# Positions in CSV are PATS/ENU -> convert, because planes are NED.
# Made modular to also be able to take PATS planes:
ENABLE_PATS_TO_NED: bool = True

# Only use rows where the drone is chasing:
REQUIRE_CHASING: bool = True
CHASING_STATE_VALUE: str = "ns_chasing_insect"

# Planes
PLANES_FILE: Optional[str] = None  # None -> DEFAULT_PLANES (below)
MARGIN_M: float = 0.0

# Metric knobs
NEAR_MISS_THRESHOLD_M: float = 0.30
INTERCEPT_THRESHOLD_M: Optional[float] = 0.15  # None -> NEAR_MISS_THRESHOLD_M
CLUSTER_RADIUS_M: float = 0.15
DEBOUNCE_S: float = 0.05

# Output
OUTPUT_DIR: str = "figures/pats"
SAVE_CSV_PATH: str = os.path.join("analysis_out/pats/metrics_all.csv")
PLOTS_DIR: str = os.path.join(OUTPUT_DIR, "plots")

# Columns (case/space-insensitive matching)
CAND_DRONE_POS = [["posX_drone"], ["posY_drone"], ["posZ_drone"]]
# Evader = bestinsect ONLY --> can change if you want to evaluate against filterd postiions
CAND_EVADER_POS = [["posX_bestinsect"], ["posY_bestinsect"], ["posZ_bestinsect"]]
CAND_TIME_ELAPSED = ["elapsed", "time", "timestamp"]
CAND_TIME_DT = ["dt", "delta_t", "delta_ms"]
CAND_DRONE_STATE = ["drone_state_str"]

# Display mappings for nicer labels
DISPLAY_CONTROLLER = {"PID": "PATS", "RL": "RL"}

DISPLAY_SUBSET = {
    "csv_1": "Replay Moth 1",
    "csv_2": "Replay Moth 2",
    "csv_3": "Replay Moth 3",
    "csv_4": "Replay Moth 4",
    "csv_5": "Replay Moth 5",
    "figure_8": "Figure 8",
    "virtual_insect": "Virtual Insect",
}


def disp_controller(c: str) -> str:
    return DISPLAY_CONTROLLER.get(c, c)


def disp_subset(s: str) -> str:
    return DISPLAY_SUBSET.get(s, s)


# Inline planes (NED)
DEFAULT_PLANES: List[Dict] = [
    {
        "n": [0.5961325493620492, 0.7253743710122877, 0.3441772878468769],
        "p0": [0.0, 0.0, 0.0],
    },
    {
        "n": [0.5961325493620492, -0.7253743710122877, 0.3441772878468769],
        "p0": [0.0, 0.0, 0.0],
    },
    {"n": [0.8571673007021123, 0.0, -0.5150380749100543], "p0": [0.0, 0.0, 0.0]},
    {"n": [-0.017452406437283352, 0.0, 0.9998476951563913], "p0": [0.0, 0.0, 0.0]},
    {"n": [0.8660254037844387, 0.0, 0.5], "p0": [0.47631397208144133, 0.0, 0.275]},
    {"n": [-0.8660254037844387, -0.0, -0.5], "p0": [3.464101615137755, 0.0, 2.0]},
    {"n": [0.0, 0.0, -1.0], "p0": [0.0, 0.0, 2.0]},
    {"n": [0.0, 0.0, 1.0], "p0": [0.0, 0.0, -0.1]},
    {"n": [-1.0, 0.0, 0.0], "p0": [2.25, 0.0, 0.0]},
    {"n": [1.0, 0.0, 0.0], "p0": [0.55, 0.0, 0.0]},
]


# helpers: transforms & parsing
def pats_to_ned(arr: np.ndarray) -> np.ndarray:
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    return np.stack([-z, -x, -y], axis=-1)


def _normalize_name(s: str) -> str:
    return s.strip().replace(" ", "").lower()


def pick_column(df: pd.DataFrame, names) -> Optional[str]:
    norm_map = {_normalize_name(c): c for c in df.columns}
    for name in names:
        if isinstance(name, (list, tuple)):
            for n in name:
                k = _normalize_name(n)
                if k in norm_map:
                    return norm_map[k]
        else:
            k = _normalize_name(name)
            if k in norm_map:
                return norm_map[k]
    return None


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def check_point_against_planes(
    point_ned: np.ndarray, planes: List[Dict], margin: float = 0.0
) -> bool:
    for pl in planes:
        n_hat = normalize(np.asarray(pl["n"], dtype=float))
        p0 = np.asarray(pl["p0"], dtype=float)
        s = float(np.dot(n_hat, point_ned - p0))
        if s < margin:
            return False
        # else continue
    return True


def load_planes(path: Optional[str]) -> List[Dict]:
    if path is None:
        return DEFAULT_PLANES
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "PLANES" in data:
        return data["PLANES"]
    if isinstance(data, list):
        return data
    raise ValueError(
        "Unsupported planes file. Provide list of {n,p0} or dict with key 'PLANES'."
    )


# stats & events
def robust_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "Q1": np.nan,
            "Q3": np.nan,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "Q1": float(np.percentile(arr, 25)),
        "Q3": float(np.percentile(arr, 75)),
    }


def detect_interceptions(
    times: np.ndarray,
    distances: np.ndarray,
    threshold: float,
    min_separation_s: float = 0.05,
) -> List[int]:
    hits, last_hit_t, above_prev = [], -1e9, True
    for i in range(len(distances)):
        d, t = distances[i], times[i]
        above = not np.isfinite(d) or (d > threshold)
        if above_prev and not above:
            if (t - last_hit_t) >= min_separation_s:
                hits.append(i)
                last_hit_t = t
        above_prev = above
    return hits


# CSV loader with robust delimiter handling
def read_csv_robust(path: str) -> pd.DataFrame:
    last_err = None
    for sep in DELIMS_TRY:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", skipinitialspace=True)
            if df.shape[1] > 1:  # looks sane
                return df
        except Exception as e:
            last_err = e
            continue
    # fall back: try default pandas parsing (comma) once
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        pass
    raise RuntimeError(
        f"Failed to parse CSV with any delimiter: {path}. Last error: {last_err}"
    )


# per-log pipeline
def load_log(
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    df = read_csv_robust(csv_path)

    pos_d_cols = [pick_column(df, x) for x in CAND_DRONE_POS]
    pos_e_cols = [pick_column(df, x) for x in CAND_EVADER_POS]
    state_col = pick_column(df, CAND_DRONE_STATE)
    if not all(pos_d_cols) or not all(pos_e_cols) or state_col is None:
        raise RuntimeError(
            f"Missing columns in {csv_path}: drone={pos_d_cols}, bestinsect={pos_e_cols}, state={state_col}"
        )

    # numeric positions
    for c in pos_d_cols + pos_e_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # time
    elapsed_col = pick_column(df, CAND_TIME_ELAPSED)
    dt_col = pick_column(df, CAND_TIME_DT)
    if elapsed_col:
        df[elapsed_col] = pd.to_numeric(df[elapsed_col], errors="coerce")
        times = df[elapsed_col].to_numpy(dtype=float)
        if np.isfinite(times[0]):
            times = times - times[0]
    else:
        if not dt_col:
            times = np.arange(len(df), dtype=float) * 0.02
        else:
            df[dt_col] = pd.to_numeric(df[dt_col], errors="coerce")
            dt = df[dt_col].to_numpy(dtype=float)
            med = np.nanmedian(dt) if np.isfinite(dt).any() else 20.0
            if med > 10.0:
                dt = dt / 1000.0
            dt = np.nan_to_num(dt, nan=(med / (1000.0 if med > 10.0 else 1.0)))
            times = np.cumsum(dt)
            times = times - times[0]

    # chasing mask (preserve alignment)
    state_vals = df[state_col].astype(str).str.strip().fillna("")
    chasing_mask_full = state_vals.eq(CHASING_STATE_VALUE)

    # positions (PATS/ENU)
    pos_d_enu = df.loc[:, pos_d_cols].to_numpy(dtype=float)
    pos_e_enu = df.loc[:, pos_e_cols].to_numpy(dtype=float)

    # numeric validity
    numeric_valid = (
        np.isfinite(pos_d_enu).all(axis=1)
        & np.isfinite(pos_e_enu).all(axis=1)
        & np.isfinite(times)
    )

    # diagnostics
    diag = {
        "n_rows": int(len(df)),
        "n_numeric_valid": int(np.count_nonzero(numeric_valid)),
        "n_chasing_rows": int(np.count_nonzero(chasing_mask_full.to_numpy())),
    }

    times = times[numeric_valid]
    pos_d_enu = pos_d_enu[numeric_valid]
    pos_e_enu = pos_e_enu[numeric_valid]
    chasing_mask = chasing_mask_full.to_numpy()[numeric_valid]

    # transform to NED
    if ENABLE_PATS_TO_NED:
        pos_d_ned = pats_to_ned(pos_d_enu)
        pos_e_ned = pats_to_ned(pos_e_enu)
    else:
        pos_d_ned, pos_e_ned = pos_d_enu, pos_e_enu

    return times, pos_d_ned, pos_e_ned, chasing_mask, diag


def compute_metrics_for_log(
    times: np.ndarray,
    pursuer_pos_ned: np.ndarray,
    evader_pos_ned: np.ndarray,
    chasing_mask: np.ndarray,
    planes: List[Dict],
) -> Dict:
    near_thr = NEAR_MISS_THRESHOLD_M
    intercept_thr = (
        INTERCEPT_THRESHOLD_M if INTERCEPT_THRESHOLD_M is not None else near_thr
    )

    base_mask = chasing_mask if REQUIRE_CHASING else np.ones_like(chasing_mask, bool)
    inbounds_mask = np.array(
        [
            check_point_against_planes(evader_pos_ned[i], planes, margin=MARGIN_M)
            for i in range(len(evader_pos_ned))
        ],
        dtype=bool,
    )
    use_mask = base_mask & inbounds_mask

    t = times[use_mask]
    pdn = pursuer_pos_ned[use_mask]
    pen = evader_pos_ned[use_mask]

    diag = {
        "n_inbounds_rows": int(np.count_nonzero(inbounds_mask)),
        "n_used_rows": int(np.count_nonzero(use_mask)),
    }

    if t.size == 0:
        return {
            **diag,
            "closest_distance": float("inf"),
            "time_in_near_miss": 0.0,
            "mean_distance": np.nan,
            "std_distance": np.nan,
            "total_interceptions": 0,
            "num_clusters": 0,
            "first_interception_time": float("inf"),
            "num_samples_used": 0,
            "used_ratio_pct": 0.0,
        }

    d = np.linalg.norm(pdn - pen, axis=1)

    closest_distance = float(np.min(d))
    time_in_near_miss = 100.0 * (np.count_nonzero(d < near_thr) / len(d))
    mean_distance = float(np.mean(d))
    std_distance = float(np.std(d))

    hit_idx = detect_interceptions(
        t, d, threshold=intercept_thr, min_separation_s=DEBOUNCE_S
    )
    total_interceptions = len(hit_idx)
    first_interception_time = (
        float(t[hit_idx[0]]) if total_interceptions > 0 else float("inf")
    )

    if total_interceptions > 0:
        hit_positions = pen[hit_idx]
        clustering = DBSCAN(eps=CLUSTER_RADIUS_M, min_samples=1).fit(hit_positions)
        num_clusters = int(len(set(clustering.labels_)))
    else:
        num_clusters = 0

    return {
        **diag,
        "closest_distance": closest_distance,
        "time_in_near_miss": float(time_in_near_miss),
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "total_interceptions": total_interceptions,
        "num_clusters": num_clusters,
        "first_interception_time": first_interception_time,
        "num_samples_used": int(len(t)),
        "used_ratio_pct": float(100.0 * len(t) / max(1, len(times))),
    }


# batch runner
@dataclass
class LogResult:
    controller: str
    subset: str
    trial_name: str
    filepath: str
    metrics: Dict


def find_csvs(root: str, subset: str) -> List[Tuple[str, str]]:
    """
    Return list of (trial_name, filepath) under <root>/<subset>.
    Expects each immediate subfolder to be a trial folder containing LOG_FILENAME.
    Falls back to a recursive search if nothing is found.
    """
    base = os.path.join(root, subset)
    results: List[Tuple[str, str]] = []
    if not os.path.isdir(base):
        print(f"[WARN] Missing subset folder: {base}")
        return results

    # Preferred: one-level trial folders
    entries = sorted(os.listdir(base))
    for entry in entries:
        trial_path = os.path.join(base, entry)
        if not os.path.isdir(trial_path):
            continue
        candidate = os.path.join(trial_path, LOG_FILENAME)
        if os.path.isfile(candidate):
            results.append((entry, candidate))

    if results:
        print(
            f"[INFO] Found {len(results)} trials in {base}: "
            + ", ".join([t for t, _ in results])
        )
        return results

    # Fallback: recursive glob
    pattern = os.path.join(base, "**", LOG_FILENAME)
    globs = sorted(glob.glob(pattern, recursive=True))
    for fp in globs:
        trial_name = os.path.basename(os.path.dirname(fp))
        results.append((trial_name, fp))
    if results:
        print(f"[INFO] (fallback) Found {len(results)} trial files in {base}")
    else:
        print(f"[WARN] No {LOG_FILENAME} found in {base} (or below).")
    return results


def run_batch() -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    planes = load_planes(PLANES_FILE)

    rows = []
    for controller, root in ROOTS.items():
        if not os.path.isdir(root):
            print(f"[WARN] Controller root missing: {controller} -> {root}")
            continue
        for subset in SUBSETS:
            pairs = find_csvs(root, subset)
            for trial_name, fp in pairs:
                try:
                    times, pos_d, pos_e, chasing_mask, diag0 = load_log(fp)
                    m = compute_metrics_for_log(
                        times, pos_d, pos_e, chasing_mask, planes
                    )
                    diag = {
                        **diag0,
                        **{k: m[k] for k in ["n_inbounds_rows", "n_used_rows"]},
                    }
                except Exception as e:
                    print(f"[ERROR] {controller}/{subset}/{trial_name}: {fp} -> {e}")
                    continue

                row = {
                    "controller": controller,
                    "subset": subset,
                    "trial": trial_name,
                    "filepath": fp,
                    **m,
                    **{f"diag_{k}": v for k, v in diag.items()},
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SAVE_CSV_PATH, index=False)
    print(f"[OK] Wrote per-log metrics: {SAVE_CSV_PATH} (rows={len(df)})")
    return df


# visualization helpers
def _finite(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def plot_box(df: pd.DataFrame, metric: str, filename: str, ylabel: str):
    # Guard: need at least one finite value
    vals_all = _finite(df[metric]).dropna()
    if vals_all.empty:
        print(f"[SKIP] No finite data for plot {metric}")
        return
    plt.figure()
    data, labels = [], []
    for ctrl in sorted(df.controller.unique()):
        vals = _finite(df.loc[df.controller == ctrl, metric]).dropna()
        if len(vals) == 0:
            continue
        data.append(vals.values)
        labels.append(ctrl)
    plt.boxplot(data, tick_labels=labels, showmeans=False)
    plt.ylabel(ylabel)
    plt.title(metric)
    out = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Saved {out}")


def plot_box_by_subset(df: pd.DataFrame, metric: str, filename: str, ylabel: str):
    vals_all = _finite(df[metric]).dropna()
    if vals_all.empty:
        print(f"[SKIP] No finite data for plot {metric} by subset")
        return
    plt.figure(figsize=(10, 5))
    subsets = SUBSETS
    positions = np.arange(len(subsets))
    width = 0.35
    ctrls = sorted(df.controller.unique())
    for i, ctrl in enumerate(ctrls):
        centers = positions + (i - (len(ctrls) - 1) / 2) * width
        data = []
        for s in subsets:
            vals = _finite(
                df.loc[(df.controller == ctrl) & (df.subset == s), metric]
            ).dropna()
            data.append(vals.values)
        plt.boxplot(
            data,
            positions=centers,
            widths=width * 0.8,
            patch_artist=False,
            tick_labels=[""] * len(subsets),
            showmeans=False,
        )
    plt.xticks(positions, subsets, rotation=0)
    plt.ylabel(ylabel)
    plt.title(f"{metric} by subset")
    out = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Saved {out}")


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def plot_success_rate(df: pd.DataFrame, filename: str):
    if df.empty:
        print(f"[SKIP] No data for success-rate plot")
        return
    ctrls = sorted(df.controller.unique())
    means, lows, highs = [], [], []
    for ctrl in ctrls:
        mask = df.controller == ctrl
        k = int((mask & (df.total_interceptions > 0)).sum())
        n = int(mask.sum())
        p, lo, hi = wilson_ci(k, n)
        means.append(p)
        lows.append(p - lo)
        highs.append(hi - p)
    x = np.arange(len(ctrls))
    plt.figure()
    plt.bar(x, means, yerr=[lows, highs], capsize=4)
    plt.xticks(x, ctrls)
    plt.ylim(0, 1)
    plt.ylabel("Success rate")
    plt.title("Interception success (Wilson 95% CI)")
    out = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Saved {out}")


# pretty print + success-rate summaries
def _fmt_pct(x):
    return "nan" if not np.isfinite(x) else f"{100.0*x:5.1f}%"


def print_controller_medians(df: pd.DataFrame, metrics: list[str]):
    print("\n=== Per-controller summary (median [IQR]) ===")
    for metric in metrics:
        print(f"\n{metric}")
        print("-" * max(12, len(metric)))
        line_fmt = "{:<10s}  n={:<3d}  median={:<8.3f}  IQR=[{:<8.3f}, {:<8.3f}]"
        for ctrl, grp in df.groupby("controller", dropna=True):
            vals = _finite(pd.to_numeric(grp[metric], errors="coerce")).dropna()
            if len(vals) == 0:
                print(f"{ctrl:<10s}  n=0   median=nan      IQR=[nan, nan]")
                continue
            med = float(np.median(vals))
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            print(line_fmt.format(ctrl, len(vals), med, q1, q3))


def print_success_tables(df_rates: pd.DataFrame):
    # Overall per controller
    print("\n=== Interception success rate (any interception in a log) ===")
    print("Overall (Wilson 95% CI)")
    print("------------------------")
    print(
        "{:<10s}  {:>4s}/{:<4s}  {:>7s}  {:>17s}".format(
            "Controller", "k", "N", "Rate", "95% CI"
        )
    )
    for ctrl, g in (
        df_rates[df_rates["level"] == "overall"]
        .sort_values("controller")
        .groupby("controller")
    ):
        r = g.iloc[0]
        print(
            "{:<10s}  {:>4d}/{:<4d}  {:>7s}  {:>17s}".format(
                ctrl, int(r.k), int(r.n), _fmt_pct(r.rate), _fmt_ci(r.ci_low, r.ci_high)
            )
        )

    # Per-subset table
    print("\nBy subset")
    print("---------")
    subsets_sorted = sorted(
        df_rates[df_rates.level == "subset"]["subset"].unique(),
        key=lambda s: (s not in SUBSETS, SUBSETS.index(s) if s in SUBSETS else 999),
    )
    ctrls_sorted = sorted(df_rates["controller"].unique())
    header = "Subset      " + "  ".join([f"{c:^24s}" for c in ctrls_sorted])
    print(header)
    print("-" * len(header))
    for s in subsets_sorted:
        row = [f"{s:<11s}"]
        for c in ctrls_sorted:
            r = df_rates[
                (df_rates.level == "subset")
                & (df_rates.subset == s)
                & (df_rates.controller == c)
            ]
            if r.empty:
                row.append(f"{'-':>24s}")
            else:
                rr = r.iloc[0]
                cell = f"{int(rr.k)}/{int(rr.n)}  {_fmt_pct(rr.rate)}  {_fmt_ci(rr.ci_low, rr.ci_high)}"
                row.append(f"{cell:>24s}")
        print("  ".join(row))


# === Paired Hodges–Lehmann effect (RL − PID) over trajectories ===
# We use per-trajectory (subset) medians to respect heterogeneity,
# and we ignore subsets where either controller has no finite value.

METRIC_HL = "mean_distance"  # change to any metric you want


def _finite_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def paired_hl_with_ci(
    diffs: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int | None = None
):
    """Median of paired differences + bootstrap CI (resample trajectories with replacement)."""
    diffs = np.asarray(diffs, dtype=float)
    hl = float(np.median(diffs))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    m = len(diffs)
    for b in range(n_boot):
        sample = diffs[rng.integers(0, m, size=m)]
        boots[b] = np.median(sample)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return hl, float(lo), float(hi)


def _fmt_ci(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "[nan, nan]"
    return f"[{100.0*lo:5.1f}%, {100.0*hi:5.1f}%]"


def summarize_success_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy dataframe with success rates overall and by subset."""
    rows = []
    for ctrl, g_ctrl in df.groupby("controller", dropna=True):
        # overall
        k = int((g_ctrl["total_interceptions"] > 0).sum())
        n = int(len(g_ctrl))
        p, lo, hi = wilson_ci(k, n)
        rows.append(
            {
                "level": "overall",
                "subset": "(all)",
                "controller": ctrl,
                "k": k,
                "n": n,
                "rate": p,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
        # by subset
        for subset, g_sub in g_ctrl.groupby("subset", dropna=True):
            ks = int((g_sub["total_interceptions"] > 0).sum())
            ns = int(len(g_sub))
            ps, los, his = wilson_ci(ks, ns)
            rows.append(
                {
                    "level": "subset",
                    "subset": subset,
                    "controller": ctrl,
                    "k": ks,
                    "n": ns,
                    "rate": ps,
                    "ci_low": los,
                    "ci_high": his,
                }
            )
    return pd.DataFrame(rows)


# NEW: boxplots with individual datapoints & subset-specific markers
def _finite_vals_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df.copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return out.dropna(subset=[metric])


# Nice distinct colors per subset (trajectory)
# Tweak if you prefer a different palette
_SUBSET_ORDER = [
    "csv_1",
    "csv_2",
    "csv_3",
    "csv_4",
    "csv_5",
    "figure_8",
    "virtual_insect",
]
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#0000ff"]
SUBSET_COLORS = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(_SUBSET_ORDER)}


def _subset_offsets(subsets: list[str], half_span: float = 0.35) -> dict[str, float]:
    """
    Evenly space subsets across [-half_span, +half_span] to avoid overlap.
    Returns dict{subset -> offset}.
    """
    n = len(subsets)
    if n == 1:
        return {subsets[0]: 0.0}
    xs = np.linspace(-half_span, +half_span, n)
    return {s: float(x) for s, x in zip(subsets, xs)}


def plot_pid_rl_box_with_points(
    df: pd.DataFrame, metric: str, filename: str, ylabel: str
):
    """
    Two controller groups (PATS/PID vs RL), with points colored by THREE evader groups:
      - Replay Moths: csv_1..csv_5 (pooled to one legend/color)
      - Figure 8:     figure_8
      - Virtual Insect: virtual_insect
    Subsets still receive small deterministic offsets for readability, but share their group color.
    Legend shows only the three groups.
    """
    dff = _finite_vals_df(df, metric)
    if dff.empty:
        print(f"[SKIP] No finite data for {metric} in plot_pid_rl_box_with_points")
        return

    # Map subsets -> 3 groups
    group_map = {
        "csv_1": "Replay Moths",
        "csv_2": "Replay Moths",
        "csv_3": "Replay Moths",
        "csv_4": "Replay Moths",
        "csv_5": "Replay Moths",
        "figure_8": "Figure 8",
        "virtual_insect": "Virtual Insect",
    }
    # Colors per 3-group legend
    GROUP_COLORS = {
        "Replay Moths": "#1f77b4",
        "Figure 8": "#ff7f0e",
        "Virtual Insect": "#2ca02c",
    }

    # Only use known subsets, attach group label
    dff = dff.copy()
    dff["evader_group"] = dff["subset"].map(group_map)
    dff = dff[~dff["evader_group"].isna()]
    if dff.empty:
        print(f"[SKIP] No finite data (mapped to groups) for {metric}")
        return

    # Keep subset offsets for visual separation of points (deterministic by subset)
    present_subsets = [s for s in _SUBSET_ORDER if s in dff["subset"].unique()]
    if not present_subsets:
        present_subsets = sorted(dff["subset"].astype(str).unique())
    offsets = _subset_offsets(present_subsets, half_span=0.2)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    order = ["PID", "RL"]
    centers = [1.0, 2.0]  # x positions for the two boxes

    # Wider boxes (per controller)
    data = [dff.loc[dff.controller == c, metric].values for c in order]
    bp = ax.boxplot(
        data,
        positions=centers,
        widths=0.42,
        tick_labels=[disp_controller(c) for c in order],
        showmeans=False,
        meanline=False,
        whis=1.5,
        flierprops=dict(marker=""),
    )
    for med in bp["medians"]:
        med.set_linewidth(2.0)
        med.set_color("#333333")

    # Overlay points: deterministic subset offset + tiny jitter, colored by 3-group
    rng = np.random.default_rng(42)
    jitter = 0.05
    seen_groups = {}  # for 3-entry legend

    for x_center, ctrl in zip(centers, order):
        g_ctrl = dff[dff.controller == ctrl]
        if g_ctrl.empty:
            continue
        for subset, g_sub in g_ctrl.groupby("subset", dropna=False):
            s = subset if isinstance(subset, str) else str(subset)
            grp = group_map.get(s, None)
            if grp is None:
                continue
            color = GROUP_COLORS.get(grp, "#7f7f7f")
            x0 = x_center + offsets.get(s, 0.0)
            x_vals = x0 + rng.normal(0.0, jitter, size=len(g_sub))
            sc = ax.scatter(
                x_vals,
                g_sub[metric].values,
                s=56,
                alpha=0.95,
                edgecolor="white",
                linewidths=0.8,
                color=color,
                label=grp,
                zorder=3,
            )
            # Keep one handle per group name for the (3-item) legend
            if grp not in seen_groups:
                seen_groups[grp] = sc

    # Legend: only 3 pooled groups
    handles = [
        seen_groups[g]
        for g in ["Replay Moths", "Figure 8", "Virtual Insect"]
        if g in seen_groups
    ]
    labels = [h.get_label() for h in handles]
    if handles:
        ax.legend(
            handles, labels, title="Evader Group", loc="upper right", frameon=True
        )

    ax.set_xlim(0.3, 2.7)
    ax.set_ylabel("First Interception Time (s)")
    ax.set_title("First Interception Time Depending on Control Method")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    out = os.path.join(PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {out}")


def plot_by_subset_box_with_points(
    df: pd.DataFrame, metric: str, filename: str, ylabel: str
):
    dff = _finite_vals_df(df, metric)
    if dff.empty:
        print(f"[SKIP] No finite data for {metric} in plot_by_subset_box_with_points")
        return

    subsets = [s for s in _SUBSET_ORDER if s in dff["subset"].unique()]
    if not subsets:
        print(f"[SKIP] No known subsets present for {metric}")
        return

    n = len(subsets)
    figw = max(10, 1.25 * n)
    fig, ax = plt.subplots(figsize=(figw, 5.2))
    group_width = 0.7  # distance between PID and RL within a subset bin
    box_width = 0.5  # wider boxes

    x_base = np.arange(n)
    pid_positions = x_base - group_width / 2
    rl_positions = x_base + group_width / 2

    # Draw wider boxes
    def _box(values: np.ndarray, x: float):
        if len(values) == 0:
            return
        bp = ax.boxplot(
            values,
            positions=[x],
            widths=box_width,
            patch_artist=False,
            showmeans=False,
            flierprops=dict(marker=""),
            whis=1.5,
        )
        for med in bp["medians"]:
            med.set_color("#333333")
            med.set_linewidth(2.0)

    for idx, s in enumerate(subsets):
        pid_vals = dff.loc[(dff.subset == s) & (dff.controller == "PID"), metric].values
        rl_vals = dff.loc[(dff.subset == s) & (dff.controller == "RL"), metric].values
        _box(pid_vals, pid_positions[idx])
        _box(rl_vals, rl_positions[idx])

    # Points: no overlap by using small jitter around each (PID/RL) position; color by subset
    rng = np.random.default_rng(123)
    jitter = 0.02
    for idx, s in enumerate(subsets):
        color = SUBSET_COLORS.get(s, "#7f7f7f")
        for ctrl, x_center in [("PID", pid_positions[idx]), ("RL", rl_positions[idx])]:
            pts = dff.loc[(dff.subset == s) & (dff.controller == ctrl), metric].values
            if len(pts) == 0:
                continue
            x_vals = x_center + rng.normal(0.0, jitter, size=len(pts))
            ax.scatter(
                x_vals,
                pts,
                s=56,
                alpha=0.95,
                edgecolor="white",
                linewidths=0.8,
                color=color,
                zorder=3,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(subsets, rotation=0)
    ax.set_xlim(-0.8, n - 0.2)  # breathing room for wider boxes
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} by trajectory")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Trajectory legend (color only)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=SUBSET_COLORS.get(s, "#7f7f7f"),
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=s,
        )
        for s in subsets
    ]
    ax.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        title="Trajectory",
        loc="upper right",
        frameon=True,
    )

    out = os.path.join(PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {out}")


# main
def main():
    df = run_batch()

    df_hits = df[df["total_interceptions"] > 0].copy()

    # Boxplot PID vs RL + individual datapoints (subset markers)
    plot_pid_rl_box_with_points(
        df_hits,
        metric="time_in_near_miss",
        filename="time_in_near_miss.png",
        ylabel="seconds",
    )

    plot_pid_rl_box_with_points(
        df_hits,
        metric="first_interception_time",
        filename="first_interception_time.png",
        ylabel="seconds",
    )

    # By-subset: paired boxes per trajectory + individual datapoints
    plot_by_subset_box_with_points(
        df,
        metric="first_interception_time",
        filename="first_interception_time_by_subset_box_points.png",
        ylabel="seconds",
    )

    # If nothing usable was found, exit gracefully
    if df.empty:
        print(
            "[WARN] No rows in results. Verify ROOTS paths and that each trial folder contains",
            LOG_FILENAME,
        )
        return

    # Diagnostics: why rows might be empty
    print("\n=== Diagnostics (sums over rows) ===")
    if "diag_n_rows" in df.columns:
        diag_cols = [c for c in df.columns if c.startswith("diag_")]
        print(df[["controller", "subset", "trial"] + diag_cols])

    # Plots (only if we have at least something finite)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_box(df, "closest_distance", "closest_distance_box.png", "meters")
    plot_box(
        df,
        "first_interception_time",
        "first_interception_time_box.png",
        "seconds (finite only)",
    )
    plot_box(df, "time_in_near_miss", "near_miss_percent_box.png", "%")
    plot_success_rate(df, "success_rate_bars.png")
    plot_box_by_subset(
        df, "closest_distance", "closest_distance_by_subset.png", "meters"
    )
    plot_box_by_subset(
        df,
        "first_interception_time",
        "first_interception_time_by_subset.png",
        "seconds (finite only)",
    )
    plot_box_by_subset(df, "time_in_near_miss", "near_miss_percent_by_subset.png", "%")

    print(f"\nAll done. Results in: {OUTPUT_DIR}")

    # 1) Per-subset medians per controller (finite-only)
    df_metric = df.copy()
    df_metric[METRIC_HL] = _finite_series(df_metric[METRIC_HL])

    traj_medians = (
        df_metric.groupby(["subset", "controller"], dropna=True)[METRIC_HL]
        .median()  # NaN if no finite values in that group
        .unstack("controller")  # columns -> controllers (PID, RL)
    )

    # Keep only subsets where BOTH controllers have a finite median
    paired = traj_medians.dropna(subset=["PID", "RL"], how="any").copy()
    paired["diff_RL_minus_PID"] = paired["RL"] - paired["PID"]

    # 2) HL + bootstrap CI (if we have at least 2 paired trajectories)
    diffs = paired["diff_RL_minus_PID"].to_numpy()
    if diffs.size == 0:
        print(
            f"[HL] No paired trajectories with finite {METRIC_HL}. Skipping HL summary."
        )
    else:
        hl, lo, hi = paired_hl_with_ci(diffs, n_boot=5000, alpha=0.05, seed=42)
        print(f"\n=== Paired Hodges–Lehmann effect on {METRIC_HL} (RL − PID) ===")
        print(f"Trajectories used: {len(diffs)} / {len(traj_medians)}")
        print(f"HL (median paired diff): {hl:.3f}")
        print(f"95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
        # Optional: show the per-trajectory table
        print("\nPer-trajectory medians and paired differences:")
        print(paired[["PID", "RL", "diff_RL_minus_PID"]].round(3))

        # Save a compact CSV
        out_csv = os.path.join(OUTPUT_DIR, f"paired_effect_{METRIC_HL}.csv")
        paired.to_csv(out_csv, index=True)
        print(f"[OK] Wrote {out_csv}")

        # Clear, aligned medians for a few core metrics
    print_controller_medians(
        df,
        metrics=[
            "closest_distance",
            "first_interception_time",
            "time_in_near_miss",
            "mean_distance",
        ],
    )

    # Success rates (overall + per subset) with Wilson 95% CIs
    df_rates = summarize_success_rates(df)
    print_success_tables(df_rates)

    # Save a tidy CSV with these summary rows
    summary_csv = os.path.join(OUTPUT_DIR, "summary_stats.csv")
    df_rates.to_csv(summary_csv, index=False)
    print(f"\n[OK] Wrote summary stats: {summary_csv}")


if __name__ == "__main__":
    main()
