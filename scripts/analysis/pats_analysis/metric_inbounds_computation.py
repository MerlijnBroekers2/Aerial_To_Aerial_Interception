from __future__ import annotations
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# ----------------------------------------------------------------------------
# CONFIG — EDIT THESE CONSTANTS
# ----------------------------------------------------------------------------
CSV_PATH: str = (
    "/Users/merlijnbroekers/Desktop/rl_acc_trials/PID/figure_8/20251001_181125/log_flight1.csv"
)
DELIM: str = ";"  # semicolon-delimited logs

# Planes are defined in NED; CSV is PATS/ENU → convert:
ENABLE_PATS_TO_NED: bool = True

# Only use rows where the drone is chasing:
REQUIRE_CHASING: bool = True
CHASING_STATE_VALUE: str = "ns_chasing_insect"  # exact match in drone_state_str

# Metrics knobs
NEAR_MISS_THRESHOLD_M: float = 0.30
INTERCEPT_THRESHOLD_M: Optional[float] = 0.15  # None → NEAR_MISS_THRESHOLD_M
CLUSTER_RADIUS_M: float = 0.15
DEBOUNCE_S: float = 0.05

SAVE_JSON_PATH: Optional[str] = None  # e.g. "/tmp/metrics.json" or None

# Column candidates (case/space-insensitive)
CAND_DRONE_POS = [["posX_drone"], ["posY_drone"], ["posZ_drone"]]
CAND_TARGET_POS = [
    ["posX_bestinsect"],
    ["posY_bestinsect"],
    ["posZ_bestinsect"],
]
CAND_TIME_ELAPSED = ["elapsed", "time", "timestamp"]
CAND_TIME_DT = ["dt", "delta_t", "delta_ms"]
CAND_DRONE_STATE = ["drone_state_str"]

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


# frame transform
def pats_to_ned(arr: np.ndarray) -> np.ndarray:
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    return np.stack([-z, -x, -y], axis=-1)


# utilities
def _normalize_name(s: str) -> str:
    return s.strip().replace(" ", "").lower()


def pick_column(df: pd.DataFrame, names) -> Optional[str]:
    norm_map = {_normalize_name(c): c for c in df.columns}
    for name in names:
        if isinstance(name, (list, tuple)):
            for n in name:
                key = _normalize_name(n)
                if key in norm_map:
                    return norm_map[key]
        else:
            key = _normalize_name(name)
            if key in norm_map:
                return norm_map[key]
    return None


# planes / inbounds
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
        "Unsupported planes file format. Use a list of {n,p0} or a dict with key 'PLANES'}."
    )


# stats / interception detection
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
    hits = []
    last_hit_t = -1e9
    above_prev = True
    for i in range(len(distances)):
        d = distances[i]
        t = times[i]
        above = not np.isfinite(d) or (d > threshold)
        if above_prev and not above:
            if (t - last_hit_t) >= min_separation_s:
                hits.append(i)
                last_hit_t = t
        above_prev = above
    return hits


# load CSV and build masks
def load_log(
    csv_path: str, delim: str = ";"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      times: (N,)
      pos_d_ned: (N,3)
      pos_t_ned: (N,3)
      chasing_mask: (N,) bool  — True where drone_state_str == CHASING_STATE_VALUE
    """
    df = pd.read_csv(csv_path, sep=delim, engine="python", skipinitialspace=True)

    # Columns
    pos_d_cols = [pick_column(df, x) for x in CAND_DRONE_POS]
    pos_t_cols = [pick_column(df, x) for x in CAND_TARGET_POS]
    state_col = pick_column(df, CAND_DRONE_STATE)
    if not all(pos_d_cols) or not all(pos_t_cols) or state_col is None:
        raise RuntimeError(
            f"Missing required columns: drone_pos={pos_d_cols}, target_pos={pos_t_cols}, drone_state={state_col}"
        )

    # Numeric conversions for positions/time
    for c in pos_d_cols + pos_t_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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

    state_vals = df[state_col].astype(str).str.strip().fillna("")
    chasing_mask_full = state_vals.eq(CHASING_STATE_VALUE)

    pos_d_enu = df.loc[:, pos_d_cols].to_numpy(dtype=float)
    pos_t_enu = df.loc[:, pos_t_cols].to_numpy(dtype=float)

    numeric_valid = (
        np.isfinite(pos_d_enu).all(axis=1)
        & np.isfinite(pos_t_enu).all(axis=1)
        & np.isfinite(times)
    )

    times = times[numeric_valid]
    pos_d_enu = pos_d_enu[numeric_valid]
    pos_t_enu = pos_t_enu[numeric_valid]
    chasing_mask = chasing_mask_full.to_numpy()[numeric_valid]

    if ENABLE_PATS_TO_NED:
        pos_d_ned = pats_to_ned(pos_d_enu)
        pos_t_ned = pats_to_ned(pos_t_enu)
    else:
        pos_d_ned = pos_d_enu
        pos_t_ned = pos_t_enu

    return times, pos_d_ned, pos_t_ned, chasing_mask


def compute_metrics_for_inbounds_and_chasing(
    times: np.ndarray,
    pursuer_pos_ned: np.ndarray,
    evader_pos_ned: np.ndarray,
    chasing_mask: np.ndarray,
    planes: List[Dict],
    near_miss_threshold: float = 0.30,
    intercept_threshold: Optional[float] = None,
    cluster_radius: float = 0.15,
    margin: float = 0.0,
    debounce_s: float = 0.05,
) -> Dict:
    if intercept_threshold is None:
        intercept_threshold = near_miss_threshold

    base_mask = (
        chasing_mask if REQUIRE_CHASING else np.ones_like(chasing_mask, dtype=bool)
    )

    inbounds_mask = np.array(
        [
            check_point_against_planes(evader_pos_ned[i], planes, margin=margin)
            for i in range(len(evader_pos_ned))
        ],
        dtype=bool,
    )

    use_mask = base_mask & inbounds_mask

    t_use = times[use_mask]
    p_use = pursuer_pos_ned[use_mask]
    e_use = evader_pos_ned[use_mask]

    if t_use.size == 0:
        return {
            "closest_distance": float("inf"),
            "time_in_near_miss": 0.0,
            "mean_distance": np.nan,
            "std_distance": np.nan,
            "robust_distance_stats": robust_stats(np.array([])),
            "total_interceptions": 0,
            "num_clusters": 0,
            "first_interception_time": float("inf"),
            "num_samples_used": 0,
            "total_samples": int(len(times)),
            "used_ratio_pct": 0.0,
            "filters": {
                "require_chasing": REQUIRE_CHASING,
                "chasing_state_value": CHASING_STATE_VALUE,
                "evader_inbounds": True,
            },
            "notes": "No samples satisfied chasing AND inbounds filters.",
        }

    d = np.linalg.norm(p_use - e_use, axis=1)

    closest_distance = float(np.min(d))
    time_in_near_miss = 100.0 * (np.count_nonzero(d < near_miss_threshold) / len(d))
    mean_distance = float(np.mean(d))
    std_distance = float(np.std(d))
    robust = robust_stats(d)

    hit_idx = detect_interceptions(
        t_use, d, threshold=intercept_threshold, min_separation_s=debounce_s
    )
    total_interceptions = len(hit_idx)
    first_interception_time = (
        float(t_use[hit_idx[0]]) if total_interceptions > 0 else float("inf")
    )

    if total_interceptions > 0:
        hit_positions = e_use[hit_idx]  # NED
        clustering = DBSCAN(eps=cluster_radius, min_samples=1).fit(hit_positions)
        num_clusters = int(len(set(clustering.labels_)))
    else:
        num_clusters = 0

    return {
        "closest_distance": closest_distance,
        "time_in_near_miss": float(time_in_near_miss),
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "robust_distance_stats": robust,
        "total_interceptions": total_interceptions,
        "num_clusters": num_clusters,
        "first_interception_time": first_interception_time,
        "num_samples_used": int(len(t_use)),
        "total_samples": int(len(times)),
        "used_ratio_pct": float(100.0 * len(t_use) / max(1, len(times))),
        "near_miss_threshold": float(near_miss_threshold),
        "intercept_threshold": float(intercept_threshold),
        "cluster_radius": float(cluster_radius),
        "margin": float(margin),
        "filters": {
            "require_chasing": REQUIRE_CHASING,
            "chasing_state_value": CHASING_STATE_VALUE,
            "evader_inbounds": True,
        },
        "frame_note": "Positions converted PATS/ENU → NED before plane checks",
    }


# main
def main():
    planes = load_planes(path=None)  # loads default planes
    times, pursuer_pos_ned, evader_pos_ned, chasing_mask = load_log(
        CSV_PATH, delim=DELIM
    )

    metrics = compute_metrics_for_inbounds_and_chasing(
        times=times,
        pursuer_pos_ned=pursuer_pos_ned,
        evader_pos_ned=evader_pos_ned,
        chasing_mask=chasing_mask,
        planes=planes,
        near_miss_threshold=NEAR_MISS_THRESHOLD_M,
        intercept_threshold=INTERCEPT_THRESHOLD_M,
        cluster_radius=CLUSTER_RADIUS_M,
        margin=None,  # No margins
        debounce_s=DEBOUNCE_S,
    )

    print(json.dumps(metrics, indent=2))
    if SAVE_JSON_PATH:
        with open(SAVE_JSON_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[OK] Metrics written to {SAVE_JSON_PATH}")


if __name__ == "__main__":
    main()
