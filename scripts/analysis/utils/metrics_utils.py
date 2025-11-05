import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable
from sklearn.cluster import DBSCAN

from scripts.analysis.utils.stats_utils import bootstrap_ci_mean


def mean_action_variation(history, key: str = "action", norm: str = "l2") -> float:
    acts = [h.get(key, None) for h in history]
    acts = [np.asarray(a).ravel() for a in acts if a is not None]
    if len(acts) < 2:
        return np.nan
    A = np.vstack(acts)
    diffs = A[1:] - A[:-1]
    if norm == "l2":
        step_var = np.linalg.norm(diffs, axis=1)
    elif norm == "l1":
        step_var = np.sum(np.abs(diffs), axis=1)
    else:
        raise ValueError("norm must be 'l2' or 'l1'")
    return float(step_var.mean())


def action_smoothness_caps(history=None, actions=None, dt=None, key: str = "action"):
    if dt is None or dt <= 0:
        raise ValueError("Provide a valid dt > 0 to compute sampling rate fs.")

    if actions is None:
        if history is None:
            raise ValueError("Provide either `actions` or `history`.")
        acts = [h.get(key, None) for h in history]
        acts = [np.asarray(a).ravel() for a in acts if a is not None]
        if len(acts) == 0:
            return np.array([np.nan]), np.nan
        A = np.vstack(acts)
    else:
        A = np.asarray(actions)
        if A.ndim == 1:
            A = A[:, None]

    T, D = A.shape
    if T < 3:
        return np.full(D, np.nan), np.nan

    fs = 1.0 / dt
    sm = np.full(D, np.nan)

    for d in range(D):
        x = np.asarray(A[:, d], dtype=float)
        x = x - np.nanmean(x)
        if not np.any(np.isfinite(x)) or np.allclose(x, 0.0):
            sm[d] = 0.0
            continue
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(T, d=dt)
        amp = np.abs(X) / T
        amp = amp[1:]
        f = freqs[1:]
        if amp.size == 0:
            sm[d] = 0.0
            continue
        n = amp.size
        sm[d] = (2.0 / (n * fs)) * np.sum(amp * f)

    return sm, float(np.nanmean(sm))


# Aggregation helpers
def aggregate_metrics_grouped(
    df: pd.DataFrame,
    group_cols: list[str],
    success_only_cols: set[str] | None = None,
) -> None:
    success_only_cols = success_only_cols or set()
    for keys, group in df.groupby(group_cols):
        title = " — ".join(
            str(k) for k in (keys if isinstance(keys, tuple) else (keys,))
        )
        print(f"\nGroup: {title}")
        for col in group.columns:
            if col in set(
                group_cols + ["evader", "level"]
            ) or not pd.api.types.is_numeric_dtype(group[col]):
                continue
            vec = group[col]
            if col in success_only_cols and "interception_flag" in group.columns:
                vec = group[group["interception_flag"] == 1][col]
            if vec.empty:
                continue
            s = robust_stats(vec)
            print(
                f"  {col:<25} mean={s['mean']:.3f} std={s['std']:.3f} med={s['median']:.3f} Q1={s['Q1']:.3f} Q3={s['Q3']:.3f}"
            )
        print("  " + "-" * 60)


def summarize_sm(
    df: pd.DataFrame, value_col: str = "smoothness_sm_mean"
) -> pd.DataFrame:
    """
    Per (abstraction, smoothing_kind, gamma): mean ± 95% CI of Sm.
    """
    rows = []
    for (abs_, kind, g), sub in df.groupby(
        ["abstraction", "smoothing_kind", "gamma"], dropna=False
    ):
        vals = pd.to_numeric(sub[value_col], errors="coerce").values
        m, (lo, hi) = bootstrap_ci_mean(vals)
        rows.append(
            {
                "abstraction": str(abs_).upper(),  # normalize to CTBR/MOTOR in caps
                "smoothing_kind": str(kind),
                "gamma": float(g),
                "mean": m,
                "lo": lo,
                "hi": hi,
                "n_trials": int(len(sub)),
            }
        )
    out = pd.DataFrame(rows).sort_values(["abstraction", "smoothing_kind", "gamma"])
    return out


def aggregate_vs_gamma(
    df: pd.DataFrame,
    ycol: str,
    successes_only: bool,
    by_evader: bool = False,
) -> pd.DataFrame:
    group_keys = ["abstraction", "smoothing_kind", "gamma"] + (
        ["evader"] if by_evader else []
    )
    rows = []
    for keys, sub in df.groupby(group_keys, dropna=False):
        s = sub.copy()
        if successes_only and "interception_flag" in s.columns:
            s = s[s["interception_flag"] == 1]
        y = pd.to_numeric(s[ycol], errors="coerce").values
        m, (lo, hi) = bootstrap_ci_mean(y)
        row = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                f"mean": m,
                f"lo": lo,
                f"hi": hi,
                "n_trials": int(len(sub)),
                "n_succ": int(
                    pd.to_numeric(sub.get("interception_flag", 0), errors="coerce")
                    .fillna(0)
                    .sum()
                ),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_keys = ["abstraction", "gamma", "smoothing_kind"] + (
        ["evader"] if by_evader else []
    )
    if not out.empty:
        out.sort_values(sort_keys, inplace=True)
    return out


def robust_stats(data):
    """Compute robust statistics for a list of data values."""
    if len(data) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "Q1": np.nan,
            "Q3": np.nan,
        }
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "median": np.median(data),
        "Q1": np.percentile(data, 25),
        "Q3": np.percentile(data, 75),
    }


def compute_simulation_metrics(
    simulation_results, near_miss_threshold=0.30, cluster_radius=0.15
):
    """
    Compute metrics for each simulation in the raw simulation data.
    """
    analysis_data = []
    for sim in simulation_results:
        history = sim.get("history", [])
        interceptions = sim.get("interceptions", [])

        # Initialize arrays
        if len(history) == 0:
            distances = np.array([])
            closest_distance = float("inf")
            time_in_near_miss = 0.0
            pursuer_pos = evader_pos = np.empty((0, 3))
        else:
            pursuer_pos = np.array([h["p_state"]["true_position"] for h in history])
            evader_pos = np.array([h["e_state"]["true_position"] for h in history])
            distances = np.linalg.norm(pursuer_pos - evader_pos, axis=1)
            closest_distance = np.min(distances)
            time_in_near_miss = (
                100.0
                * np.count_nonzero(distances < near_miss_threshold)
                / len(distances)
            )

        # Interception stats
        total_interceptions = len(interceptions)
        if total_interceptions > 0:
            interception_positions = [i["evader_pos"] for i in interceptions]
            clustering = DBSCAN(eps=cluster_radius, min_samples=1).fit(
                interception_positions
            )
            num_clusters = len(set(clustering.labels_))
            first_interception_time = interceptions[0]["time"]
        else:
            num_clusters = 0
            first_interception_time = float("inf")

        computed_metrics = {
            "name": sim.get("name", ""),
            "first_interception_time": first_interception_time,
            "closest_distance": closest_distance,
            "time_in_near_miss": time_in_near_miss,
            "total_interceptions": total_interceptions,
            "num_clusters": num_clusters,
            "mean_distance": np.mean(distances) if distances.size > 0 else np.nan,
            "std_distance": np.std(distances) if distances.size > 0 else np.nan,
        }

        analysis_data.append(
            {
                "name": sim.get("name", ""),
                "evader_pos": evader_pos,
                "pursuer_pos": pursuer_pos,
                "interceptions": interceptions,
                "metrics": computed_metrics,
                "history": history,
            }
        )

    return analysis_data
