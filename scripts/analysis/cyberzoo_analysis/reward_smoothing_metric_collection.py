"""
Reward smoothing (gamma) analysis on REAL logs.

Directory expectations (per root):
  root/
    part_1/
      spoof_starts*.csv
      <one or more real *.csv logs>
    part_2/
      ...

Edit the EXPERIMENTS list below to point to your roots & labels.
Each entry carries:
  - level_key     ∈ {"motor", "ctbr", "acc_indi", "PointMass"}
  - smoothing_kind ∈ {"none", "rate_only", "gamma"}
  - gamma          (float; for 'none' and 'rate_only' set 0.0)

Output:
  - A single combined CSV (one row per trial per real CSV)
  - Figures and summary CSVs in a sibling *_figs folder
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# 0) PUT YOUR ROOTS + LABELS HERE
#    smoothing_kind ∈ {"none", "rate_only", "gamma"}
#    gamma is a float; for 'none' and 'rate_only' use 0.0
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # ---------------- CTBR ----------------
    # {
    #     "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_reward_smoothing_logs/ctbr/nosmooth_no-rate_g0p0",
    #     "level_key": "ctbr",
    #     "smoothing_kind": "none",
    #     "gamma": 0.0,
    # },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_ctbr_logs/dr_10",
        "level_key": "ctbr",
        "smoothing_kind": "rate_only",
        "gamma": 0.0,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/reward_smoothing_logs/ctbr/gamma_05_dr_10",
        "level_key": "ctbr",
        "smoothing_kind": "gamma",
        "gamma": 5.0,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/reward_smoothing_logs/ctbr/gamma_10_dr_10",
        "level_key": "ctbr",
        "smoothing_kind": "gamma",
        "gamma": 10.0,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/reward_smoothing_logs/ctbr/gamma_20_dr_10",
        "level_key": "ctbr",
        "smoothing_kind": "gamma",
        "gamma": 20.0,
    },
    # ---------------- MOTOR ----------------
    # {
    #     "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_reward_smoothing_logs/motor/nosmooth_no-rate_g0p0",
    #     "level_key": "motor",
    #     "smoothing_kind": "none",
    #     "gamma": 0.0,
    # },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/real_dr_abstraction_logs/new_params_motor/dr_10",
        "level_key": "motor",
        "smoothing_kind": "rate_only",
        "gamma": 0.0,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/reward_smoothing_logs/motor/gamma_05_dr_10",
        "level_key": "motor",
        "smoothing_kind": "gamma",
        "gamma": 5.0,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/reward_smoothing_logs/motor/gamma_10_dr10",
        "level_key": "motor",
        "smoothing_kind": "gamma",
        "gamma": 10.0,
    },
    {
        "root": "/Users/merlijnbroekers/Desktop/Drone_Interception/reward_smoothing_logs/motor/gamma_20_dr10",
        "level_key": "motor",
        "smoothing_kind": "gamma",
        "gamma": 20.0,
    },
]

# Where to save the combined CSV:
OUT_CSV = "real_metrics_reward_smoothing.csv"

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
    print("Could not import metrics_multiple_trials_real_life.py")
    print("Make sure that file is alongside this script or on PYTHONPATH.")
    raise

CFG = SegmentationConfig(
    tol_m=0.05,  # meters
    min_spoof_duration_s=1.0,  # seconds
    post_spoof_guard_s=0.0,  # seconds
    max_spoof_speed_mps=0.05,  # m/s (set to None to disable)
    intercept_radius=0.15,  # meters
    near_miss_threshold=0.30,  # meters
)


# ---------------------------------------------------------------------------
# 2) Helpers (unchanged except for labels & metadata)
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
    smoothing_kind: str
    gamma: float


# ---------------------------------------------------------------------------
# 3) Core processing (unchanged, but attach new metadata)
# ---------------------------------------------------------------------------
def process_part(
    part_dir: Path,
    level_key: str,
    smoothing_kind: str,
    gamma: float,
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
                smoothing_kind=str(smoothing_kind),
                gamma=float(gamma),
            )
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
                        "time_in_near_miss": float(r.get("time_in_near_miss", 0.0)),
                        "total_interceptions": int(r.get("total_interceptions", 0)),
                        "num_clusters": int(r.get("num_clusters", 0)),
                        "mean_distance": float(r.get("mean_distance", float("nan"))),
                        "std_distance": float(r.get("std_distance", float("nan"))),
                        "interception_flag": int(r.get("interception_flag", 0)),
                    }
                )

            if save_per_part and rows:
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
        smoothing_kind = exp["smoothing_kind"]
        gamma = float(exp["gamma"])

        if not root.exists():
            print(f"Root not found: {root}")
            continue

        part_dirs = _find_part_dirs(root)
        print(f"{root} → {len(part_dirs)} part(s)")

        for part in part_dirs:
            df_part = process_part(
                part, level_key, smoothing_kind, gamma, cfg, save_per_part=save_per_part
            )
            if not df_part.empty:
                all_rows.append(df_part)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(
        [
            "level_key",
            "smoothing_kind",
            "gamma",
            "root_dir",
            "part_dir",
            "real_csv",
            "trial",
        ],
        kind="stable",
    )
    return out.reset_index(drop=True)


# ===================== VISUALIZATION & SUMMARY =====================
WINSORIZE = True
WINSOR_Q = (0.05, 0.95)
SHOW_POINTS = True
POINT_SIZE = 26
POINT_ALPHA = 0.75
POINT_EDGEWIDTH = 0.6
JITTER_FRAC = 0.30

COLOR_BY_KIND = {
    "gamma": "#1f77b4",  # line color for gamma smoothing curve
    "none": "#2ca02c",  # baseline 1 (γ=0, no rate)
    "rate_only": "#ff7f0e",  # baseline 2 (γ=0, with rate)
}

KIND_LABELS = {
    "gamma": "Gamma smoothing",
    "none": "No smoothing",
    "rate_only": "Rate-only",
}


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

    gb = df.groupby(
        ["level_key", "level_label", "smoothing_kind", "gamma"], dropna=False
    )
    try:
        out = gb.apply(_agg, include_groups=False).reset_index()
    except TypeError:
        out = gb[[]].apply(_agg).reset_index()
    return out


def _line_and_points_vs_gamma(
    ax,
    sub: pd.DataFrame,
    ycol_mean: str,
    ycol_lo: str,
    ycol_hi: str,
    title: str,
    ylabel: str,
):
    ax.set_title(title)
    ax.set_xlabel("Gamma")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)

    # 1) Gamma curve
    s_gamma = sub[sub["smoothing_kind"] == "gamma"].sort_values("gamma")
    if not s_gamma.empty:
        ax.plot(
            s_gamma["gamma"],
            s_gamma[ycol_mean],
            marker="D",
            color=COLOR_BY_KIND["gamma"],
            label=KIND_LABELS["gamma"],
        )
        ax.fill_between(
            s_gamma["gamma"],
            s_gamma[ycol_lo],
            s_gamma[ycol_hi],
            alpha=0.22,
            color=COLOR_BY_KIND["gamma"],
        )

    # 2) Baselines at gamma=0 with slight x-jitter
    xs = sub["gamma"].values
    if xs.size:
        x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
        span = max(1e-9, x_max - x_min)
        jitter = 0.02 * (span if span > 0 else 1.0)
    else:
        jitter = 0.02

    for kind, off in [("none", -jitter), ("rate_only", +jitter)]:
        s0 = sub[sub["smoothing_kind"] == kind]
        if not s0.empty:
            x0 = s0["gamma"].astype(float).values + off
            y0 = s0[ycol_mean].values
            lo = s0[ycol_lo].values
            hi = s0[ycol_hi].values
            ax.errorbar(
                x0,
                y0,
                yerr=[y0 - lo, hi - y0],
                fmt="o",
                capsize=3,
                color=COLOR_BY_KIND[kind],
                label=KIND_LABELS[kind],
            )

    # legend de-dup
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    hh, ll = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            hh.append(h)
            ll.append(l)
    if hh:
        ax.legend(hh, ll, title="Smoothing", loc="best")


def _bootstrap_ci_mean(
    x: np.ndarray, iters: int = 2000, alpha: float = 0.05, seed: int | None = 123
):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(x)), (lo, hi)


def _aggregate_vs_gamma(
    df: pd.DataFrame, ycol: str, successes_only: bool
) -> pd.DataFrame:
    """
    Per (level_label, smoothing_kind, gamma) aggregate mean ± 95% CI.
    If successes_only: only intercepted trials are used to compute the metric.
    """
    rows = []
    for (lvl, kind, g), sub in df.groupby(
        ["level_label", "smoothing_kind", "gamma"], dropna=False
    ):
        s = sub.copy()
        if successes_only:
            s = s[s["interception_flag"] == 1]
        y = pd.to_numeric(s[ycol], errors="coerce").values
        m, (lo, hi) = _aggregate_ci = _bootstrap_ci_mean(y)
        rows.append(
            {
                "level_label": lvl,
                "smoothing_kind": kind,
                "gamma": float(g),
                f"mean_{ycol}": m,
                f"lo_{ycol}": lo,
                f"hi_{ycol}": hi,
                "n_trials": int(len(sub)),
                "n_succ": int(sub["interception_flag"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["level_label", "gamma", "smoothing_kind"], inplace=True)
    return out


def make_figures(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        print("No data to plot.")
        return
    out_dir = _ensure_dir(out_dir)

    # ---------- Summary CSV ----------
    summary = _summarize(df)
    summary_path = out_dir / "real_reward_smoothing_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")

    # ---------- Interception rate vs gamma (95% CI), per level ----------
    levels = sorted(df["level_label"].dropna().unique().tolist())
    if levels:
        fig, axes = plt.subplots(
            1, len(levels), figsize=(7.2 * len(levels), 5), sharey=True
        )
        if len(levels) == 1:
            axes = [axes]
        for ax, lvl in zip(axes, levels):
            # get per (kind, gamma) rate ± CI
            rows = []
            for (kind, g), sub in df[df["level_label"] == lvl].groupby(
                ["smoothing_kind", "gamma"], dropna=False
            ):
                rate, (lo, hi) = _bootstrap_ci_rate(sub["interception_flag"].to_numpy())
                rows.append(
                    {
                        "smoothing_kind": kind,
                        "gamma": float(g),
                        "mean_rate": rate,
                        "lo_rate": lo,
                        "hi_rate": hi,
                    }
                )
            rate_df = pd.DataFrame(rows)
            _line_and_points_vs_gamma(
                ax,
                rate_df,
                "mean_rate",
                "lo_rate",
                "hi_rate",
                title=f"{lvl} — Interception rate (95% CI)",
                ylabel="Rate",
            )
            ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(out_dir / "interception_rate_vs_gamma.png", dpi=300)
        plt.close(fig)

    # ---------- FIT vs gamma (successes only) ----------
    ag_fit = _aggregate_vs_gamma(
        df, ycol="first_interception_time", successes_only=True
    )
    if not ag_fit.empty:
        fig, axes = plt.subplots(
            1, len(levels), figsize=(7.2 * len(levels), 5), sharey=False
        )
        if len(levels) == 1:
            axes = [axes]
        for ax, lvl in zip(axes, levels):
            sub = ag_fit[ag_fit["level_label"] == lvl]
            _line_and_points_vs_gamma(
                ax,
                sub.rename(
                    columns={
                        "mean_first_interception_time": "mean_fit",
                        "lo_first_interception_time": "lo_fit",
                        "hi_first_interception_time": "hi_fit",
                    }
                ),
                "mean_fit",
                "lo_fit",
                "hi_fit",
                title=f"{lvl} — First interception time (succ. only)",
                ylabel="Time to first interception (s)",
            )
        fig.tight_layout()
        fig.savefig(out_dir / "fit_vs_gamma.png", dpi=300)
        plt.close(fig)

    # ---------- Time in near miss vs gamma (all trials) ----------
    ag_nm = _aggregate_vs_gamma(df, ycol="time_in_near_miss", successes_only=False)
    if not ag_nm.empty:
        fig, axes = plt.subplots(
            1, len(levels), figsize=(7.2 * len(levels), 5), sharey=False
        )
        if len(levels) == 1:
            axes = [axes]
        for ax, lvl in zip(axes, levels):
            sub = ag_nm[ag_nm["level_label"] == lvl]
            _line_and_points_vs_gamma(
                ax,
                sub.rename(
                    columns={
                        "mean_time_in_near_miss": "mean_nm",
                        "lo_time_in_near_miss": "lo_nm",
                        "hi_time_in_near_miss": "hi_nm",
                    }
                ),
                "mean_nm",
                "lo_nm",
                "hi_nm",
                title=f"{lvl} — Time in near miss",
                ylabel="Time in near miss (s)",
            )
        fig.tight_layout()
        fig.savefig(out_dir / "near_miss_vs_gamma.png", dpi=300)
        plt.close(fig)

    # ---------- Boxplots per level × gamma-kind (optional; similar to your DR figs) ----------
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
        # positions: per-level panel, inside: x=gamma buckets, split by kind
        for lvl in levels:
            sub = dd[dd["level_label"] == lvl].copy()
            if sub.empty:
                continue
            gammas = sorted(sub["gamma"].dropna().unique().tolist())
            kinds = ["gamma", "none", "rate_only"]  # order in legend
            centers = np.arange(len(gammas)) + 1
            k = len(kinds)
            width = 0.8
            box_w = width / (k + 0.4)

            fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(gammas) * k), 5))
            rng = np.random.default_rng(12345)
            samples_legend_done = False

            for j, kind in enumerate(kinds):
                data = []
                pos = []
                pts_x = []
                pts_y = []
                for i, g in enumerate(gammas):
                    cell = pd.to_numeric(
                        sub[(sub["smoothing_kind"] == kind) & (sub["gamma"] == g)][
                            ycol
                        ],
                        errors="coerce",
                    ).dropna()
                    if cell.size == 0:
                        continue
                    box_vals = cell.copy()
                    if WINSORIZE and box_vals.size >= 2:
                        lo_q, hi_q = WINSOR_Q
                        lo, hi = float(box_vals.quantile(lo_q)), float(
                            box_vals.quantile(hi_q)
                        )
                        box_vals = box_vals.clip(lo, hi)
                    x_center = centers[i] + (j - (k - 1) / 2) * box_w
                    data.append(box_vals.values)
                    pos.append(x_center)

                    if show_points:
                        jitter = box_w * JITTER_FRAC
                        yvals = cell.values
                        if logy:
                            yvals = yvals[yvals > 0]
                        if yvals.size:
                            xs = x_center + rng.uniform(
                                -jitter, jitter, size=yvals.size
                            )
                            pts_x.append(xs)
                            pts_y.append(yvals)

                if data:
                    bp = ax.boxplot(
                        data,
                        positions=pos,
                        widths=box_w * 0.9,
                        patch_artist=True,
                        showfliers=False,
                    )
                    for box in bp["boxes"]:
                        c = COLOR_BY_KIND.get(kind, "gray")
                        box.set_facecolor(c)
                        box.set_edgecolor(c)
                        box.set_alpha(0.6)
                    for med in bp["medians"]:
                        med.set_color("black")
                        med.set_linewidth(1.8)
                    if show_points and pts_x:
                        c = COLOR_BY_KIND.get(kind, "gray")
                        ax.scatter(
                            np.concatenate(pts_x),
                            np.concatenate(pts_y),
                            s=POINT_SIZE,
                            alpha=POINT_ALPHA,
                            facecolor=c,
                            edgecolor="black",
                            linewidths=POINT_EDGEWIDTH,
                            zorder=3,
                        )
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

            ax.set_title(f"{title} — {lvl}")
            ax.set_xlabel("Gamma")
            ax.set_ylabel(ylabel)
            ax.set_xticks(centers)
            ax.set_xticklabels([str(g) for g in gammas])
            if logy:
                ax.set_yscale("log")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            kind_handles = [
                Patch(
                    facecolor=COLOR_BY_KIND.get(k, "gray"),
                    label=KIND_LABELS[k],
                    alpha=0.6,
                )
                for k in kinds
            ]
            if samples_legend_done:
                ax.legend(
                    handles=kind_handles + [samples_handle],
                    title="Smoothing / Data",
                    loc="best",
                )
            else:
                ax.legend(handles=kind_handles, title="Smoothing", loc="best")

            fig.tight_layout()
            fig.savefig(out_dir / f"{fname.replace('.png','')}_{lvl}.png", dpi=300)
            plt.close(fig)

    _grouped_boxplot(
        "first_interception_time",
        "First interception time (successes only)",
        "Time to first interception (s)",
        "fit_box_real_gamma.png",
        successes_only=True,
    )
    _grouped_boxplot(
        "closest_distance",
        "Closest distance",
        "Distance (m)",
        "closest_distance_box_real_gamma.png",
    )
    _grouped_boxplot(
        "time_in_near_miss",
        "Time in near miss",
        "Percent of time within threshold (%)",
        "near_miss_box_real_gamma.png",
    )

    # Scatter: FIT vs closest_distance (successes only), color by smoothing_kind
    dd = df[df["interception_flag"] == 1].copy()
    if not dd.empty:
        fig, axes = plt.subplots(
            1, len(levels), figsize=(7.2 * len(levels), 5.5), sharey=True
        )
        if len(levels) == 1:
            axes = [axes]
        for ax, lvl in zip(axes, levels):
            sub = dd[dd["level_label"] == lvl]
            if sub.empty:
                continue
            for kind in ["gamma", "none", "rate_only"]:
                ss = sub[sub["smoothing_kind"] == kind]
                if ss.empty:
                    continue
                ax.scatter(
                    ss["first_interception_time"],
                    ss["closest_distance"],
                    s=28,
                    alpha=0.7,
                    edgecolors="none",
                    c=COLOR_BY_KIND.get(kind, "gray"),
                    label=KIND_LABELS[kind],
                )
            out_mask = _mad_outlier_mask(sub["first_interception_time"])
            if out_mask.any():
                out = sub.loc[out_mask]
                ax.scatter(
                    out["first_interception_time"],
                    out["closest_distance"],
                    s=60,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.5,
                    label="MAD outlier (FIT)",
                )
            ax.set_title(f"{lvl} — FIT vs Closest distance (succ. only)")
            ax.set_xlabel("First interception time (s)")
            ax.set_ylabel("Closest distance (m)")
            ax.grid(True, linestyle="--", alpha=0.4)
            handles = [
                Patch(
                    facecolor=COLOR_BY_KIND.get(k, "gray"),
                    label=KIND_LABELS[k],
                    alpha=0.7,
                )
                for k in ["gamma", "none", "rate_only"]
            ]
            if out_mask.any():
                handles.append(
                    Patch(facecolor="none", edgecolor="red", label="MAD outlier")
                )
            ax.legend(handles=handles, title="Smoothing", loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter_fit_vs_closest_real_gamma.png", dpi=300)
        plt.close(fig)

    print(f"Figures saved to: {out_dir.resolve()}")


# ===================== EXTRA STATS & OUTLIER SUMMARY =====================
def _count_mad_outliers(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0
    med = float(s.median())
    mad = float((s - med).abs().median()) or 1e-9
    robust_z = (s - med).abs() / (1.4826 * mad)
    return int((robust_z > 3.5).sum())


def _group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (level_label, smoothing_kind, gamma) stats:
      - counts (trials, successes, rate)
      - medians/IQR for main metrics
      - MAD outlier counts (FIT uses successes only)
    """
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (lvl, kind, g), grp in df.groupby(
        ["level_label", "smoothing_kind", "gamma"], dropna=False
    ):
        gdf = grp.copy()
        succ = gdf[gdf["interception_flag"] == 1]
        n_trials = int(len(gdf))
        n_success = int(succ["interception_flag"].sum())
        rate = float(n_success / n_trials) if n_trials else float("nan")

        def _med_iqr(vec: pd.Series) -> Tuple[float, float]:
            x = pd.to_numeric(vec, errors="coerce").dropna()
            if x.empty:
                return float("nan"), float("nan")
            return float(x.median()), float(x.quantile(0.75) - x.quantile(0.25))

        fit_med, fit_iqr = _med_iqr(succ["first_interception_time"])
        cd_med, cd_iqr = _med_iqr(gdf["closest_distance"])
        nm_med, nm_iqr = _med_iqr(gdf["time_in_near_miss"])

        rows.append(
            {
                "level_label": lvl,
                "smoothing_kind": kind,
                "gamma": float(g),
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
                "outliers_closest_distance": _count_mad_outliers(
                    gdf["closest_distance"]
                ),
                "outliers_time_in_near_miss": _count_mad_outliers(
                    gdf["time_in_near_miss"]
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["level_label", "gamma", "smoothing_kind"])
        .reset_index(drop=True)
    )


def _gamma_overall_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across levels & kinds: per (smoothing_kind, gamma) counts and rate."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (kind, g), grp in df.groupby(["smoothing_kind", "gamma"], dropna=False):
        succ = grp[grp["interception_flag"] == 1]
        rows.append(
            {
                "smoothing_kind": kind,
                "gamma": float(g),
                "n_trials": int(len(grp)),
                "n_success": int(succ["interception_flag"].sum()),
                "interception_rate": float(
                    succ["interception_flag"].sum() / max(1, len(grp))
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["gamma", "smoothing_kind"])
        .reset_index(drop=True)
    )


def _bootstrap_diff_of_medians(
    a: np.ndarray,
    b: np.ndarray,
    iters: int = 5000,
    alpha: float = 0.05,
    seed: int | None = 123,
):
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


def _pairwise_gamma_effects_for_level(
    df: pd.DataFrame, level_label: str, ycol: str, successes_only: bool
) -> pd.DataFrame:
    """
    For a given level_label and metric ycol, compute pairwise comparisons
    between all (smoothing_kind, gamma) combos, using bootstrap 95% CIs of median diffs.
    """
    sub = df[df["level_label"] == level_label].copy()
    if successes_only:
        sub = sub[sub["interception_flag"] == 1]
    combos = sorted(
        sub[["smoothing_kind", "gamma"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    rows = []
    for i in range(len(combos)):
        for j in range(i + 1, len(combos)):
            (k0, g0), (k1, g1) = combos[i], combos[j]
            a = sub[(sub["smoothing_kind"] == k0) & (sub["gamma"] == g0)][ycol].values
            b = sub[(sub["smoothing_kind"] == k1) & (sub["gamma"] == g1)][ycol].values
            est, lo, hi, n_a, n_b = _bootstrap_diff_of_medians(a, b)
            rows.append(
                {
                    "level_label": level_label,
                    "metric": ycol,
                    "A_kind": k0,
                    "A_gamma": float(g0),
                    "B_kind": k1,
                    "B_gamma": float(g1),
                    "median_diff_A_minus_B": est,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_A": int(n_a),
                    "n_B": int(n_b),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["A_kind", "A_gamma", "B_kind", "B_gamma"])
        .reset_index(drop=True)
    )


def make_stat_summaries(df: pd.DataFrame, out_dir: Path):
    out_dir = _ensure_dir(out_dir)
    group_stats = _group_stats(df)
    overall = _gamma_overall_counts(df)
    group_path = out_dir / "real_group_stats_gamma.csv"
    overall_path = out_dir / "real_overall_counts_gamma.csv"
    group_stats.to_csv(group_path, index=False)
    overall.to_csv(overall_path, index=False)
    print(f"Saved group stats to:   {group_path}")
    print(f"Saved overall totals to:{overall_path}")

    levels = sorted(df["level_label"].dropna().unique().tolist())
    metrics = [
        ("first_interception_time", True),
        ("closest_distance", False),
        ("time_in_near_miss", False),
    ]
    for lvl in levels:
        for ycol, succ_only in metrics:
            eff = _pairwise_gamma_effects_for_level(
                df, lvl, ycol, successes_only=succ_only
            )
            eff_path = out_dir / f"effects_{lvl}_{ycol}_gamma.csv"
            eff.to_csv(eff_path, index=False)
            print(f"Saved effects to:       {eff_path}")

    # --------- Console report ----------
    if not group_stats.empty:
        print("\n===== REAL metrics — counts & outliers by (Level, Kind, Gamma) =====")
        for _, r in group_stats.iterrows():
            print(
                f"{r['level_label']:<8} {r['smoothing_kind']:<9} γ={r['gamma']:>5}:  "
                f"n={int(r['n_trials'])}, success={int(r['n_success'])} ({r['interception_rate']:.2f})  |  "
                f"outliers — FIT:{int(r['outliers_FIT'])}, CD:{int(r['outliers_closest_distance'])}, NM:{int(r['outliers_time_in_near_miss'])}"
            )
        print("--------------------------------------------------------------------")


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

    figs_dir = out_path.with_suffix("")  # strip .csv -> stem
    figs_dir = figs_dir.parent / (figs_dir.stem + "_figs")
    make_figures(df, figs_dir)
    make_stat_summaries(df, figs_dir)

    print(f"Saved combined metrics to: {out_path}")
    print(f"Total rows: {len(df)}   (unique real CSVs: {df['real_csv'].nunique()})")


if __name__ == "__main__":
    sys.exit(main())
