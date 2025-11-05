"""
Edit the EXPERIMENTS list below to point to your roots and their labels.
Each root can contain subfolders: part_1, part_2, part_3, ...

Inside each part_*:
  - one spoof CSV whose name starts with "spoof_starts"
  - one or more real log CSVs (any *.csv that is not a spoof file)

Output: a single combined CSV with one row per trial per real CSV.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# 0) PUT YOUR ROOTS + LABELS HERE
#    level_key ∈ {"motor", "ctbr", "acc_indi", "PointMass"}
#    dr_pct is an integer like 0, 10, 20, 30
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # Examples — replace with your own:
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_ctbr_logs/dr_00",
        "level_key": "ctbr",
        "dr_pct": 0,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_ctbr_logs/dr_10",
        "level_key": "ctbr",
        "dr_pct": 10,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_ctbr_logs/dr_20",
        "level_key": "ctbr",
        "dr_pct": 20,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_ctbr_logs/dr_30",
        "level_key": "ctbr",
        "dr_pct": 30,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_motor/dr_00",
        "level_key": "motor",
        "dr_pct": 0,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_motor/dr_10",
        "level_key": "motor",
        "dr_pct": 10,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_motor/dr_20",
        "level_key": "motor",
        "dr_pct": 20,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_motor/dr_30",
        "level_key": "motor",
        "dr_pct": 30,
    },
    # {"root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/ctbr_logs/dr_10", "level_key": "ctbr",  "dr_pct": 10},
    # {"root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/motor_logs/dr_00","level_key": "motor", "dr_pct": 0},
    # {"root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/motor_logs/dr_20","level_key": "motor", "dr_pct": 20},
]

# Where to save the combined CSV:
OUT_CSV = "analysis_out/cyber_zoo/real_metrics_domain_randomization.csv"

# Also drop a per-part CSV inside each part_* folder (one file per real CSV)?
SAVE_PER_PART = False

# ---------------------------------------------------------------------------
# 1) Segmentation / metrics knobs (aligned with your module defaults)
# ---------------------------------------------------------------------------
try:
    from metrics_multiple_trials_real_life import (
        split_real_trials,
        SegmentationConfig,
    )
except Exception as e:
    print("Could not import split_real_trials_by_spoof_positions.py")
    print("   Make sure that file is alongside this script or on PYTHONPATH.")
    raise

# ! This config is not linked to my actual config file rn --> silent errors/unexpected behaviour

CFG = SegmentationConfig(
    tol_m=0.05,  # meters
    min_spoof_duration_s=1.0,  # seconds
    post_spoof_guard_s=0.0,  # seconds
    max_spoof_speed_mps=0.05,  # m/s (set to None to disable)
    intercept_radius=0.15,  # meters
    near_miss_threshold=0.30,  # meters
)


# ---------------------------------------------------------------------------
# 2) Helpers
# ---------------------------------------------------------------------------
def _level_label_from_key(level_key: str) -> str:
    mapping = {
        "motor": "MOTOR",
        "ctbr": "CTBR",
        "acc_indi": "ACC_INDI",
        "PointMass": "POINTMASS",
        "pointmass": "POINTMASS",
    }
    return mapping.get(level_key, str(level_key).upper())


def _is_spoof_csv(p: Path) -> bool:
    return (
        p.is_file()
        and p.suffix.lower() == ".csv"
        and p.name.lower().startswith("spoof_starts")
    )


def _is_real_csv(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".csv" and not _is_spoof_csv(p)


def _find_part_dirs(root: Path) -> List[Path]:
    """Return sorted part_* directories; if none, treat root itself as one part."""
    if not root.is_dir():
        return []
    parts = [
        d
        for d in sorted(root.iterdir())
        if d.is_dir() and d.name.lower().startswith("part_")
    ]
    return parts or [root]


def _find_spoof_csv(part_dir: Path) -> Optional[Path]:
    cands = [p for p in part_dir.glob("*.csv") if _is_spoof_csv(p)]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # If multiple spoofs exist, pick the most recently modified
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _find_real_csvs(part_dir: Path) -> List[Path]:
    return [p for p in part_dir.glob("*.csv") if _is_real_csv(p)]


@dataclass
class RowMeta:
    root_dir: str
    part_dir: str
    real_csv: str
    spoof_csv: str
    level_key: str
    level_label: str
    dr_pct: int


# ---------------------------------------------------------------------------
# 3) Core processing
# ---------------------------------------------------------------------------
def process_part(
    part_dir: Path,
    level_key: str,
    dr_pct: int,
    cfg: SegmentationConfig,
    save_per_part: bool = False,
) -> pd.DataFrame:
    spoof = _find_spoof_csv(part_dir)
    if spoof is None:
        print(f"No spoof CSV in {part_dir} — skipping this part.")
        return pd.DataFrame()

    real_csvs = _find_real_csvs(part_dir)
    if not real_csvs:
        print(f"No real CSVs in {part_dir} — skipping this part.")
        return pd.DataFrame()

    level_label = _level_label_from_key(level_key)
    rows = []
    for real_csv in sorted(real_csvs):
        try:
            df_trials = split_real_trials(
                str(real_csv), str(spoof), cfg, save_metrics_csv=None
            )

            meta = RowMeta(
                root_dir=str(part_dir.parent),
                part_dir=str(part_dir),
                real_csv=str(real_csv),
                spoof_csv=str(spoof),
                level_key=level_key,
                level_label=level_label,
                dr_pct=int(dr_pct),
            )
            # attach metadata to each trial row
            for _, r in df_trials.iterrows():
                rows.append(
                    {
                        **asdict(meta),
                        "trial": int(r.get("trial", 0)),
                        "i_start": int(r.get("i_start", 0)),
                        "i_end_excl": int(r.get("i_end_excl", 0)),
                        "n_rows": int(r.get("n_rows", 0)),
                        "first_interception_time": float(
                            r.get("first_interception_time", float("inf"))
                        ),
                        "closest_distance": float(
                            r.get("closest_distance", float("inf"))
                        ),
                        "time_in_near_miss": float(
                            r.get("time_in_near_miss", 0.0)
                        ),  # percent (0..100)
                        "total_interceptions": int(r.get("total_interceptions", 0)),
                        "num_clusters": int(r.get("num_clusters", 0)),
                        "mean_distance": float(r.get("mean_distance", float("nan"))),
                        "std_distance": float(r.get("std_distance", float("nan"))),
                        "interception_flag": int(r.get("interception_flag", 0)),
                    }
                )

            if save_per_part and rows:
                # write a per-real CSV right inside this part
                out_p = part_dir / f"metrics_{Path(real_csv).stem}.csv"
                pd.DataFrame(
                    [row for row in rows if row["real_csv"] == str(real_csv)]
                ).to_csv(out_p, index=False)
        except Exception as e:
            print(
                f"Error processing real={Path(real_csv).name} in {part_dir.name}: {e}"
            )
            continue

    return pd.DataFrame(rows)


def process_all(
    experiments: List[dict], cfg: SegmentationConfig, save_per_part: bool = False
) -> pd.DataFrame:
    all_rows: List[pd.DataFrame] = []

    for exp in experiments:
        root = Path(exp["root"]).expanduser().resolve()
        level_key = exp["level_key"]
        dr_pct = int(exp["dr_pct"])

        if not root.exists():
            print(f"Root not found: {root}")
            continue

        part_dirs = _find_part_dirs(root)
        print(f"{root} → {len(part_dirs)} part(s)")

        for part in part_dirs:
            df_part = process_part(
                part, level_key, dr_pct, cfg, save_per_part=save_per_part
            )
            if not df_part.empty:
                all_rows.append(df_part)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(
        ["level_key", "dr_pct", "root_dir", "part_dir", "real_csv", "trial"],
        kind="stable",
    )
    return out.reset_index(drop=True)


# ===================== VISUALIZATION & SUMMARY =====================
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- knobs ---
WINSORIZE = True
WINSOR_Q = (0.05, 0.95)  # clamp to 5th–95th pct for boxplots if enabled
COLOR_BY_DR = {0: "#1f77b4", 10: "#2ca02c", 20: "#ff7f0e", 30: "#d62728"}
# --- scatter overlay knobs ---
SHOW_POINTS = True  # toggle overlay on/off
POINT_SIZE = 26  # dot size
POINT_ALPHA = 0.75  # dot transparency
POINT_EDGEWIDTH = 0.6  # outline width
JITTER_FRAC = 0.30  # fraction of box width used for horizontal jitter


def _dr_color(dr):
    return COLOR_BY_DR.get(int(dr), "gray")


def _bootstrap_ci_rate(
    flags: np.ndarray, iters: int = 2000, alpha: float = 0.05, seed: int | None = 123
):
    flags = np.asarray(flags, dtype=float)
    flags = flags[~np.isnan(flags)]
    n = flags.size
    if n == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = flags[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(flags.mean()), (lo, hi)


def _winsorize_series(s: pd.Series, qlo=0.05, qhi=0.95):
    if s.empty:
        return s
    lo, hi = float(s.quantile(qlo)), float(s.quantile(qhi))
    return s.clip(lo, hi)


def _mad_outlier_mask(s: pd.Series, thresh=3.5):
    # robust z via MAD (L1). Returns boolean mask of outliers.
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return s.notna() & False
    med = float(x.median())
    mad = float((x - med).abs().median()) or 1e-9
    r = (x - med).abs() / (1.4826 * mad)
    mask = pd.Series(False, index=s.index)
    mask.loc[r.index] = r > thresh
    return mask


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    def _agg(group):
        g = group.copy()
        succ = g[g["interception_flag"] == 1]

        s = {}
        s["n_trials"] = int(len(g))
        s["n_success"] = int(succ["interception_flag"].sum())

        rate, (lo, hi) = _bootstrap_ci_rate(g["interception_flag"].to_numpy())
        s["interception_rate"] = rate
        s["rate_lo95"] = lo
        s["rate_hi95"] = hi

        for col in [
            "first_interception_time",
            "closest_distance",
            "time_in_near_miss",
            "mean_distance",
        ]:
            vec = succ[col] if col == "first_interception_time" else g[col]
            vec = pd.to_numeric(vec, errors="coerce").dropna()
            if vec.empty:
                s[f"{col}_median"] = np.nan
                s[f"{col}_IQR"] = np.nan
            else:
                s[f"{col}_median"] = float(vec.median())
                s[f"{col}_IQR"] = float(vec.quantile(0.75) - vec.quantile(0.25))
        return pd.Series(s)

    gb = df.groupby(["level_key", "level_label", "dr_pct"], dropna=False)

    try:
        out = gb.apply(_agg, include_groups=False).reset_index()
    except TypeError:
        # Fallback: exclude groups by selecting no value columns
        out = gb[[]].apply(_agg).reset_index()

    return out


def make_figures(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        print("No data to plot.")
        return
    out_dir = _ensure_dir(out_dir)

    # Save summary CSV
    summary = _summarize(df)
    summary_path = out_dir / "real_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")

    # Interception rate with 95% CI, annotated with n
    levels = sorted(df["level_label"].dropna().unique().tolist())
    drs = sorted(df["dr_pct"].dropna().unique().tolist())
    x = np.arange(len(levels))
    w = 0.8 / max(1, len(drs))

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, dr in enumerate(drs):
        means, los, his, ns = [], [], [], []
        for lvl in levels:
            sub = df[(df["level_label"] == lvl) & (df["dr_pct"] == dr)]
            m, (lo, hi) = _bootstrap_ci_rate(sub["interception_flag"].to_numpy())
            means.append(m)
            los.append(lo)
            his.append(hi)
            ns.append(len(sub))

        xpos = x - 0.4 + w / 2 + j * w
        xpos = np.asarray(xpos, dtype=float)
        means = np.asarray(means, dtype=float)
        los = np.asarray(los, dtype=float)
        his = np.asarray(his, dtype=float)

        # bars only where mean is finite
        mask = np.isfinite(means)
        ax.bar(
            xpos[mask],
            means[mask],
            width=w,
            color=_dr_color(dr),
            alpha=0.9,
            label=f"DR {int(dr)}%",
        )

        # error bars only where all finite
        ci_mask = mask & np.isfinite(los) & np.isfinite(his)
        if np.any(ci_mask):
            yerr = np.vstack(
                [means[ci_mask] - los[ci_mask], his[ci_mask] - means[ci_mask]]
            )
            ax.errorbar(
                xpos[ci_mask],
                means[ci_mask],
                yerr=yerr,
                fmt="none",
                ecolor="black",
                capsize=3,
                lw=1,
            )

        # annotate counts for the ones we plotted
        for xi, yi, n in zip(xpos[mask], means[mask], np.asarray(ns)[mask]):
            ax.text(
                xi,
                (yi if np.isfinite(yi) else 0) + 0.03,
                f"n={int(n)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Interception rate (95% CI) — Real")
    ax.set_xlabel("Abstraction level")
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(
        title="DR level",
        ncol=min(4, len(drs)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    fig.tight_layout()
    fig.savefig(out_dir / "interception_rate_real.png", dpi=300)
    plt.close(fig)

    def _grouped_boxplot(
        ycol: str,
        title: str,
        ylabel: str,
        fname: str,
        successes_only: bool = False,
        logy: bool = False,
        show_points: bool = SHOW_POINTS,
    ):
        dd = df.copy()
        if successes_only:
            dd = dd[dd["interception_flag"] == 1]

        levels = sorted(dd["level_label"].dropna().unique().tolist())
        drs = sorted(dd["dr_pct"].dropna().unique().tolist())
        if not levels or not drs:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        centers = np.arange(len(levels)) + 1
        k = max(1, len(drs))
        width = 0.8
        box_w = width / (k + 0.5)

        rng = np.random.default_rng(12345)  # deterministic jitter across runs
        samples_legend_done = False

        for j, dr in enumerate(drs):
            data = []
            pos = []

            # We also cache raw (non-winsorized) values for the scatter overlay
            pts_x = []
            pts_y = []

            for i, lvl in enumerate(levels):
                sub = pd.to_numeric(
                    dd[(dd["level_label"] == lvl) & (dd["dr_pct"] == dr)][ycol],
                    errors="coerce",
                ).dropna()

                # Skip empty cells entirely
                if sub.size == 0:
                    continue

                # Winsorize only for the box statistics (optional)
                box_vals = sub.copy()
                if WINSORIZE and box_vals.size >= 2:
                    lo_q, hi_q = WINSOR_Q
                    lo, hi = float(box_vals.quantile(lo_q)), float(
                        box_vals.quantile(hi_q)
                    )
                    box_vals = box_vals.clip(lo, hi)

                x_center = centers[i] + (j - (k - 1) / 2) * box_w
                data.append(box_vals.values)
                pos.append(x_center)

                # Prepare scatter points (use original sub, not winsorized)
                if show_points:
                    jitter = box_w * JITTER_FRAC
                    # If logy, drop non-positive values to avoid issues
                    yvals = sub.values
                    if logy:
                        yvals = yvals[yvals > 0]
                    if yvals.size:
                        xs = x_center + rng.uniform(-jitter, jitter, size=yvals.size)
                        pts_x.append(xs)
                        pts_y.append(yvals)

            # Draw the boxes for this DR
            if data:
                bp = ax.boxplot(
                    data,
                    positions=pos,
                    widths=box_w * 0.9,
                    patch_artist=True,
                    showfliers=False,
                )
                for box in bp["boxes"]:
                    c = _dr_color(dr)
                    box.set_facecolor(c)
                    box.set_edgecolor(c)
                    box.set_alpha(0.6)
                for med in bp["medians"]:
                    med.set_color("black")
                    med.set_linewidth(1.8)

                # Overlay the points for this DR
                if show_points and pts_x:
                    c = _dr_color(dr)
                    ax.scatter(
                        np.concatenate(pts_x),
                        np.concatenate(pts_y),
                        s=POINT_SIZE,
                        alpha=POINT_ALPHA,
                        facecolor=c,
                        edgecolor="black",
                        linewidths=POINT_EDGEWIDTH,
                        zorder=3,  # on top of boxes
                    )
                    # Add a single legend entry for "Samples"
                    if not samples_legend_done:
                        samples_handle = Line2D(
                            [],
                            [],
                            marker="o",
                            linestyle="",
                            markersize=np.sqrt(POINT_SIZE),
                            markerfacecolor=c,
                            markeredgecolor="black",
                            markeredgewidth=POINT_EDGEWIDTH,
                            alpha=POINT_ALPHA,
                            label="Samples",
                        )
                        samples_legend_done = True

        # Axes formatting
        ax.set_title(title)
        ax.set_xlabel("Abstraction level")
        ax.set_ylabel(ylabel)
        ax.set_xticks(centers)
        ax.set_xticklabels(levels, rotation=0)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # DR legend + optional "Samples" entry
        dr_handles = [
            Patch(facecolor=_dr_color(dr), label=f"DR {int(dr)}%", alpha=0.6)
            for dr in drs
        ]
        if samples_legend_done:
            # keep DR color legend and add a "Samples" marker
            ax.legend(
                handles=dr_handles + [samples_handle],
                title="DR level / Data",
                loc="best",
            )
        else:
            ax.legend(handles=dr_handles, title="DR level", loc="best")

        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=300)
        plt.close(fig)

    # Boxplots
    _grouped_boxplot(
        "first_interception_time",
        "First interception time (successes only)",
        "Time to first interception (s)",
        "fit_box_real.png",
        successes_only=True,
    )

    _grouped_boxplot(
        "closest_distance",
        "Closest distance",
        "Distance (m)",
        "closest_distance_box_real.png",
    )

    _grouped_boxplot(
        "time_in_near_miss",
        "Time in near miss",
        "Percent of time within threshold (%)",
        "near_miss_box_real.png",
    )

    # Scatter: FIT vs closest_distance (successes only), highlight MAD outliers
    dd = df[df["interception_flag"] == 1].copy()
    if not dd.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for dr in drs:
            for lvl in levels:
                sub = dd[(dd["dr_pct"] == dr) & (dd["level_label"] == lvl)]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["first_interception_time"],
                    sub["closest_distance"],
                    s=28,
                    alpha=0.7,
                    label=f"{lvl} — DR {dr}",
                    edgecolors="none",
                    c=_dr_color(dr),
                )
        # outliers by MAD on FIT (optional annotate)
        out_mask = _mad_outlier_mask(dd["first_interception_time"])
        if out_mask.any():
            out = dd.loc[out_mask]
            ax.scatter(
                out["first_interception_time"],
                out["closest_distance"],
                s=60,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                label="MAD outlier (FIT)",
            )
        ax.set_title("FIT vs Closest distance — successes only (Real)")
        ax.set_xlabel("First interception time (s)")
        ax.set_ylabel("Closest distance (m)")
        ax.grid(True, linestyle="--", alpha=0.4)
        # one legend per DR; keep concise
        handles = [
            Patch(facecolor=_dr_color(dr), label=f"DR {int(dr)}%", alpha=0.7)
            for dr in drs
        ]
        if out_mask.any():
            handles.append(
                Patch(facecolor="none", edgecolor="red", label="MAD outlier")
            )
        leg = ax.legend(handles=handles, title="DR level", loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter_fit_vs_closest_real.png", dpi=300)
        plt.close(fig)

    print(f"Figures saved to: {out_dir.resolve()}")


# ===================== EXTRA STATS & OUTLIER SUMMARY =====================
def _count_mad_outliers(series: pd.Series, successes_only: bool = False) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0
    med = float(s.median())
    mad = float((s - med).abs().median()) or 1e-9
    robust_z = (s - med).abs() / (1.4826 * mad)
    return int((robust_z > 3.5).sum())


def _group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (level_label, dr_pct) stats:
      - counts (trials, successes, rate)
      - medians/IQR for main metrics
      - MAD outlier counts for each metric (FIT uses successes only)
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (lvl, dr), g in df.groupby(["level_label", "dr_pct"], dropna=False):
        g = g.copy()
        succ = g[g["interception_flag"] == 1]
        n_trials = int(len(g))
        n_success = int(succ["interception_flag"].sum())
        rate = float(n_success / n_trials) if n_trials else float("nan")

        def _med_iqr(vec: pd.Series) -> tuple[float, float]:
            x = pd.to_numeric(vec, errors="coerce").dropna()
            if x.empty:
                return float("nan"), float("nan")
            return float(x.median()), float(x.quantile(0.75) - x.quantile(0.25))

        fit_med, fit_iqr = _med_iqr(succ["first_interception_time"])
        cd_med, cd_iqr = _med_iqr(g["closest_distance"])
        nm_med, nm_iqr = _med_iqr(g["time_in_near_miss"])

        row = {
            "level_label": lvl,
            "dr_pct": int(dr) if pd.notna(dr) else dr,
            "n_trials": n_trials,
            "n_success": n_success,
            "interception_rate": rate,
            "FIT_median": fit_med,
            "FIT_IQR": fit_iqr,
            "closest_distance_median": cd_med,
            "closest_distance_IQR": cd_iqr,
            "near_miss_median": nm_med,
            "near_miss_IQR": nm_iqr,
            "outliers_FIT": _count_mad_outliers(succ["first_interception_time"]),
            "outliers_closest_distance": _count_mad_outliers(g["closest_distance"]),
            "outliers_time_in_near_miss": _count_mad_outliers(g["time_in_near_miss"]),
        }
        rows.append(row)
    return (
        pd.DataFrame(rows).sort_values(["level_label", "dr_pct"]).reset_index(drop=True)
    )


def _dr_overall_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across levels: per-DR counts and rate."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for dr, g in df.groupby("dr_pct", dropna=False):
        g = g.copy()
        succ = g[g["interception_flag"] == 1]
        rows.append(
            {
                "dr_pct": int(dr) if pd.notna(dr) else dr,
                "n_trials": int(len(g)),
                "n_success": int(succ["interception_flag"].sum()),
                "interception_rate": float(
                    succ["interception_flag"].sum() / max(1, len(g))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("dr_pct").reset_index(drop=True)


def _bootstrap_diff_of_medians(
    a: np.ndarray,
    b: np.ndarray,
    iters: int = 5000,
    alpha: float = 0.05,
    seed: int | None = 123,
):
    """
    Nonparametric 'test': bootstrap CI for median(A) - median(B).
    Returns: (point_estimate, lo, hi, n_a, n_b)
    """
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy()
    n_a, n_b = a.size, b.size
    if n_a < 1 or n_b < 1:
        return float("nan"), float("nan"), float("nan"), n_a, n_b
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(iters):
        a_s = a[rng.integers(0, n_a, size=n_a)]
        b_s = b[rng.integers(0, n_b, size=n_b)]
        boots.append(np.median(a_s) - np.median(b_s))
    boots = np.asarray(boots)
    est = float(np.median(a) - np.median(b))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return est, lo, hi, n_a, n_b


def _pairwise_dr_effects_for_level(
    df: pd.DataFrame, level_label: str, ycol: str, successes_only: bool
) -> pd.DataFrame:
    """
    For a given level_label and metric ycol, compute pairwise DR comparisons with bootstrap 95% CIs.
    """
    sub = df[df["level_label"] == level_label].copy()
    if successes_only:
        sub = sub[sub["interception_flag"] == 1]
    drs = sorted(sub["dr_pct"].dropna().unique().tolist())
    rows = []
    for i in range(len(drs)):
        for j in range(i + 1, len(drs)):
            d0, d1 = drs[i], drs[j]
            a = sub[sub["dr_pct"] == d0][ycol].values
            b = sub[sub["dr_pct"] == d1][ycol].values
            est, lo, hi, n_a, n_b = _bootstrap_diff_of_medians(a, b)
            rows.append(
                {
                    "level_label": level_label,
                    "metric": ycol,
                    "dr_A": int(d0),
                    "dr_B": int(d1),
                    "median_diff_A_minus_B": est,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_A": int(n_a),
                    "n_B": int(n_b),
                }
            )
    return pd.DataFrame(rows).sort_values(["dr_A", "dr_B"]).reset_index(drop=True)


def make_stat_summaries(df: pd.DataFrame, out_dir: Path):
    """
    Saves:
      - real_group_stats.csv                  (per level & DR, with MAD outliers)
      - real_dr_overall_counts.csv           (per DR across levels)
      - effects_<LEVEL>_<METRIC>.csv         (pairwise DR bootstrap CIs per level)
    Prints:
      - compact console report with counts and outliers per group
    """
    out_dir = _ensure_dir(out_dir)
    group_stats = _group_stats(df)
    overall = _dr_overall_counts(df)
    group_path = out_dir / "real_group_stats.csv"
    overall_path = out_dir / "real_dr_overall_counts.csv"
    group_stats.to_csv(group_path, index=False)
    overall.to_csv(overall_path, index=False)
    print(f"Saved group stats to:   {group_path}")
    print(f"Saved DR totals to:     {overall_path}")

    # Pairwise DR effects per level for three metrics
    levels = sorted(df["level_label"].dropna().unique().tolist())
    metrics = [
        ("first_interception_time", True),
        ("closest_distance", False),
        ("time_in_near_miss", False),
    ]
    for lvl in levels:
        for ycol, succ_only in metrics:
            eff = _pairwise_dr_effects_for_level(
                df, lvl, ycol, successes_only=succ_only
            )
            eff_path = out_dir / f"effects_{lvl}_{ycol}.csv"
            eff.to_csv(eff_path, index=False)
            print(f"Saved effects to:       {eff_path}")

    # --------- Console report ----------
    if not group_stats.empty:
        print("\n===== REAL metrics — counts & outliers by (Level, DR) =====")
        for _, r in group_stats.iterrows():
            print(
                f"{r['level_label']:<8} DR {int(r['dr_pct']):>2}:  "
                f"n={int(r['n_trials'])}, success={int(r['n_success'])} ({r['interception_rate']:.2f})  |  "
                f"outliers — FIT:{int(r['outliers_FIT'])}, CD:{int(r['outliers_closest_distance'])}, NM:{int(r['outliers_time_in_near_miss'])}"
            )
        print("-----------------------------------------------------------")


# ---------------------------------------------------------------------------
# 4) Run
# ---------------------------------------------------------------------------
def main():
    if not EXPERIMENTS:
        print("EXPERIMENTS is empty. Add your roots/labels at the top of this file.")
        return 1

    df = process_all(EXPERIMENTS, CFG, save_per_part=SAVE_PER_PART)
    if df.empty:
        print("No metrics collected. Check folder structure and spoof/real CSVs.")
        return 1

    out_path = Path(OUT_CSV).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    out_path = Path(OUT_CSV).expanduser().resolve()
    figs_dir = out_path.with_suffix("")  # strip .csv
    figs_dir = figs_dir.parent / (figs_dir.stem + "_figs")
    make_figures(df, figs_dir)
    make_stat_summaries(df, figs_dir)

    print(f"Saved combined metrics to: {out_path}")
    print(f"Total rows: {len(df)}   (unique real CSVs: {df['real_csv'].nunique()})")
    # with pd.option_context("display.max_columns", None, "display.width", 180):
    #     print(df.head(12))
    # return 0


if __name__ == "__main__":
    sys.exit(main())
