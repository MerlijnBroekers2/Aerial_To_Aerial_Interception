# split_real_trials_by_spoof_positions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


# ========================= Config =========================
@dataclass(frozen=True)
class SegmentationConfig:
    # Position matching tolerance to mark a row as "spoof" (meters)
    tol_m: float = 0.05

    # Minimum spoof hold to be considered a real spoof segment
    # (seconds; avoids false positives if the target briefly passes near a spoof point)
    min_spoof_duration_s: float = 1.0

    # Optional guard after spoof ends before starting the trial (seconds)
    post_spoof_guard_s: float = 0.0

    # Optional: require spoof segments to be nearly stationary (target speed below this)
    # Set to None to disable. If enabled, segments whose median target speed exceeds this
    # are discarded from spoof detection.
    max_spoof_speed_mps: float | None = 0.05

    # Metric thresholds (align with your pipeline)
    intercept_radius: float = 0.15
    near_miss_threshold: float = 0.30


# ========================= IO / Prep =========================
def _prep_real_df(real_csv: str) -> pd.DataFrame:
    df = pd.read_csv(real_csv, low_memory=False)
    need = [
        "time",
        "pos_x",
        "pos_y",
        "pos_z",
        "pos_target_x",
        "pos_target_y",
        "pos_target_z",
    ]
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{real_csv} missing required columns: {missing}")
    df[need] = df[need].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
    # Ensure time is monotonic
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _prep_spoof_df(spoof_csv: str) -> np.ndarray:
    """
    Returns an array S of shape (M,3) with spoof positions (x_n, y_e, z_d).
    """
    df = pd.read_csv(spoof_csv)
    need = ["trial", "x_n", "y_e", "z_d"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{spoof_csv} missing required columns: {missing}")
    df = df.copy()
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce").astype("Int64")
    df["x_n"] = pd.to_numeric(df["x_n"], errors="coerce")
    df["y_e"] = pd.to_numeric(df["y_e"], errors="coerce")
    df["z_d"] = pd.to_numeric(df["z_d"], errors="coerce")
    df = df.dropna(subset=["x_n", "y_e", "z_d"]).sort_values("trial")
    return df[["x_n", "y_e", "z_d"]].to_numpy(dtype=float)


# ========================= Spoof mask & segments =========================
def _infer_dt(time_vec: np.ndarray) -> float:
    if time_vec.size < 2:
        return 0.01
    diffs = np.diff(time_vec)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else 0.01


def _target_speed(time_vec: np.ndarray, pos_tgt: np.ndarray) -> np.ndarray:
    """Finite-difference speed of target (m/s) with matching length (prepend 0)."""
    if len(time_vec) < 2:
        return np.zeros(len(time_vec))
    dt = np.diff(time_vec)
    dpos = np.linalg.norm(np.diff(pos_tgt, axis=0), axis=1)
    spd = np.zeros(len(time_vec))
    good = dt > 0
    spd[1:][good] = dpos[good] / dt[good]
    return spd


def _build_spoof_mask(
    pos_target: np.ndarray, spoof_xyz: np.ndarray, tol_m: float
) -> np.ndarray:
    """
    pos_target: (N,3), spoof_xyz: (M,3)
    Returns boolean mask (N,) True if distance to any spoof point <= tol_m
    """
    # Broadcast distances to all spoof points, then reduce by min
    diff = pos_target[:, None, :] - spoof_xyz[None, :, :]
    d = np.linalg.norm(diff, axis=2)  # (N, M)
    return (d <= float(tol_m)).any(axis=1)


def _clean_spoof_runs(
    is_spoof: np.ndarray,
    time_vec: np.ndarray,
    pos_target: np.ndarray,
    min_spoof_duration_s: float,
    max_spoof_speed_mps: float | None,
) -> np.ndarray:
    """
    Remove too-short or too-fast spoof runs.
    """
    is_spoof = is_spoof.astype(bool).copy()
    N = len(is_spoof)
    if N == 0:
        return is_spoof

    dt_med = _infer_dt(time_vec)
    min_len = int(np.ceil(min_spoof_duration_s / max(dt_med, 1e-6)))

    # Label contiguous spoof runs
    i = 0
    while i < N:
        if not is_spoof[i]:
            i += 1
            continue
        j = i + 1
        while j < N and is_spoof[j]:
            j += 1
        # run is [i, j)
        run_len = j - i

        # Duration filter
        too_short = run_len < max(1, min_len)

        # Speed filter (optional)
        too_fast = False
        if (not too_short) and (max_spoof_speed_mps is not None):
            spd = _target_speed(time_vec[i:j], pos_target[i:j])
            if np.isfinite(spd).any():
                med_spd = float(np.nanmedian(spd))
                too_fast = med_spd > float(max_spoof_speed_mps)

        if too_short or too_fast:
            is_spoof[i:j] = False

        i = j
    return is_spoof


def _find_spoof_segments(is_spoof: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx) for spoof segments; end_idx is exclusive.
    """
    segments: List[Tuple[int, int]] = []
    N = len(is_spoof)
    i = 0
    while i < N:
        if not is_spoof[i]:
            i += 1
            continue
        j = i + 1
        while j < N and is_spoof[j]:
            j += 1
        segments.append((i, j))
        i = j
    return segments


# ========================= Metrics =========================
def _compute_metrics_on_window(
    df: pd.DataFrame,
    near_miss_threshold: float,
    intercept_radius: float,
) -> dict:
    df = df.copy()
    # re-zero time for this window
    t0 = float(df["time"].iloc[0])
    df["time"] = df["time"] - t0

    pursuer_pos = df[["pos_x", "pos_y", "pos_z"]].to_numpy()
    evader_pos = df[["pos_target_x", "pos_target_y", "pos_target_z"]].to_numpy()

    if len(df) == 0:
        distances = np.array([])
    else:
        distances = np.linalg.norm(pursuer_pos - evader_pos, axis=1)

    if distances.size == 0:
        return {
            "first_interception_time": float("inf"),
            "closest_distance": float("inf"),
            "time_in_near_miss": 0.0,
            "total_interceptions": 0,
            "num_clusters": 0,
            "mean_distance": float("nan"),
            "std_distance": float("nan"),
            "interception_flag": 0,
        }

    closest_distance = float(np.min(distances))
    time_in_near_miss = (
        100.0 * float((distances < near_miss_threshold).sum()) / float(len(distances))
    )

    hits_idx = np.where(distances <= intercept_radius)[0]
    total_interceptions = int(hits_idx.size)
    if total_interceptions > 0:
        interception_positions = evader_pos[hits_idx]
        clustering = DBSCAN(eps=intercept_radius, min_samples=1).fit(
            interception_positions
        )
        num_clusters = int(len(set(clustering.labels_)))
        first_interception_time = float(df["time"].to_numpy()[hits_idx[0]])
        interception_flag = 1
    else:
        num_clusters = 0
        first_interception_time = float("inf")
        interception_flag = 0

    return {
        "first_interception_time": first_interception_time,
        "closest_distance": closest_distance,
        "time_in_near_miss": float(time_in_near_miss),
        "total_interceptions": total_interceptions,
        "num_clusters": num_clusters,
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "interception_flag": interception_flag,
    }


def split_real_trials(
    real_csv: str,
    spoof_csv: str,
    cfg: SegmentationConfig = SegmentationConfig(),
    save_metrics_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Segment a real-life log into trials purely by matching pos_target to spoof positions.

    Trial definition:
      - Detect spoof segments where ||(pos_target) - (spoof_xyz)|| <= tol_m.
      - Clean segments by duration (>= min_spoof_duration_s) and optional low-speed check.
      - Trial k starts at the first sample AFTER spoof segment k ends (+ optional guard)
        and ends at the FIRST sample of spoof segment k+1 (exclusive).
      - The last trial ends at the end of the log.

    Returns a DataFrame with one row per trial and the usual metrics.
    """
    df = _prep_real_df(real_csv)
    S = _prep_spoof_df(spoof_csv)  # (M, 3) spoof points
    T = df[["pos_target_x", "pos_target_y", "pos_target_z"]].to_numpy()
    t = df["time"].to_numpy()
    dt_med = _infer_dt(t)
    guard_n = int(np.round(cfg.post_spoof_guard_s / max(dt_med, 1e-6)))

    # 1) Raw spoof mask (distance to any spoof point <= tol)
    is_spoof_raw = _build_spoof_mask(T, S, cfg.tol_m)
    # 2) Clean segments
    is_spoof = _clean_spoof_runs(
        is_spoof_raw,
        time_vec=t,
        pos_target=T,
        min_spoof_duration_s=cfg.min_spoof_duration_s,
        max_spoof_speed_mps=cfg.max_spoof_speed_mps,
    )
    # 3) Segment indices
    segments = _find_spoof_segments(is_spoof)

    if not segments:
        # No spoof segments found → treat the whole file as a single trial.
        m = _compute_metrics_on_window(
            df, cfg.near_miss_threshold, cfg.intercept_radius
        )
        out = pd.DataFrame(
            [{**m, "trial": 1, "i_start": 0, "i_end_excl": len(df), "n_rows": len(df)}]
        )
        if save_metrics_csv:
            pd.DataFrame(out).to_csv(save_metrics_csv, index=False)
        return out

    # Trials are intervals between spoof segments
    rows: List[dict] = []
    for k, (i0, i1) in enumerate(segments):
        # trial k starts after this spoof
        start_idx = i1 + guard_n
        # until next spoof start (or end of file)
        if k + 1 < len(segments):
            end_idx = segments[k + 1][0]
        else:
            end_idx = len(df)

        # clamp + skip empty windows
        start_idx = min(max(start_idx, 0), len(df))
        end_idx = min(max(end_idx, 0), len(df))
        if end_idx - start_idx < 2:
            rows.append(
                {
                    "trial": k + 1,
                    "i_start": start_idx,
                    "i_end_excl": end_idx,
                    "n_rows": int(max(0, end_idx - start_idx)),
                    "first_interception_time": float("inf"),
                    "closest_distance": float("inf"),
                    "time_in_near_miss": 0.0,
                    "total_interceptions": 0,
                    "num_clusters": 0,
                    "mean_distance": float("nan"),
                    "std_distance": float("nan"),
                    "interception_flag": 0,
                }
            )
            continue

        df_win = df.iloc[start_idx:end_idx].copy()
        m = _compute_metrics_on_window(
            df_win,
            near_miss_threshold=cfg.near_miss_threshold,
            intercept_radius=cfg.intercept_radius,
        )
        rows.append(
            {
                "trial": k + 1,
                "i_start": int(start_idx),
                "i_end_excl": int(end_idx),
                "n_rows": int(len(df_win)),
                **m,
            }
        )

    out = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    if save_metrics_csv:
        out.to_csv(save_metrics_csv, index=False)
    return out


# -------------- Optional helper: visualize segmentation (quick verification) ----------
def preview_segmentation(
    real_csv: str, spoof_csv: str, cfg: SegmentationConfig = SegmentationConfig()
):
    """
    Quick plot: distance-to-nearest-spoof over time, with spoof windows and trial windows.
    """
    import matplotlib.pyplot as plt

    df = _prep_real_df(real_csv)
    S = _prep_spoof_df(spoof_csv)
    T = df[["pos_target_x", "pos_target_y", "pos_target_z"]].to_numpy()
    t = df["time"].to_numpy()
    dt_med = _infer_dt(t)
    guard_n = int(np.round(cfg.post_spoof_guard_s / max(dt_med, 1e-6)))

    # Dist to nearest spoof
    diff = T[:, None, :] - S[None, :, :]
    d = np.linalg.norm(diff, axis=2).min(axis=1)

    is_spoof_raw = d <= cfg.tol_m
    is_spoof = _clean_spoof_runs(
        is_spoof_raw, t, T, cfg.min_spoof_duration_s, cfg.max_spoof_speed_mps
    )
    segments = _find_spoof_segments(is_spoof)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, d, label="dist(target, nearest spoof)")
    ax.axhline(cfg.tol_m, color="r", linestyle="--", label="tol")

    for k, (i0, i1) in enumerate(segments):
        ax.axvspan(
            t[i0],
            t[i1 - 1] if i1 > i0 else t[i0],
            color="green",
            alpha=0.2,
            label="spoof" if k == 0 else None,
        )
        # trial window (after guard) until next spoof start
        start_idx = min(i1 + guard_n, len(t) - 1)
        end_idx = segments[k + 1][0] if k + 1 < len(segments) else len(t) - 1
        if end_idx > start_idx:
            ax.axvspan(
                t[start_idx],
                t[end_idx - 1],
                color="blue",
                alpha=0.1,
                label="trial" if k == 0 else None,
            )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance to spoof (m)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example (edit paths or import in your notebook):
    REAL = "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_ctbr_logs/dr_00/part_2/20250924-121552.csv"
    SPOOF = "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_ctbr_logs/dr_00/part_2/spoof_starts_20250924_102131.csv"
    cfg = SegmentationConfig(
        tol_m=0.05,
        min_spoof_duration_s=1.0,
        post_spoof_guard_s=0.0,
        max_spoof_speed_mps=0.05,
    )
    df_trials = split_real_trials(REAL, SPOOF, cfg, save_metrics_csv=None)
    print(df_trials["first_interception_time"])
    # preview_segmentation(REAL, SPOOF, cfg)
