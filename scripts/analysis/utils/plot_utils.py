from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


def save_obs_boxplot(df, col, title, ylabel, filename, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column=col, by="obs_mode_label", grid=False, ax=ax)
    ax.set_title(title)
    fig.suptitle("")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15, fontsize=10)
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_obs_barplot(df, title, ylabel, filename, output_dir):
    rate = df.groupby("obs_mode_label")["interception_flag"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    rate.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Abstraction‑level simple boxplots
# ----------------------------------------------------------------------------
def plot_first_interception_by_label(df: pd.DataFrame):
    d = df[df.get("interception_flag", 1) == 1].copy()
    d["level_label"] = d["level"].astype(str).str.upper()
    plt.figure(figsize=(12, 5))
    d.boxplot(column="first_interception_time", by="level_label", grid=False)
    plt.title("First Interception Time by Control Abstraction")
    plt.suptitle("")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=-15)
    plt.tight_layout()
    plt.show()


def plot_closest_distance_by_label(df: pd.DataFrame):
    d = df[df.get("interception_flag", 1) == 1].copy()
    d["level_label"] = d["level"].astype(str).str.upper()
    plt.figure(figsize=(12, 5))
    d.boxplot(column="closest_distance", by="level_label", grid=False)
    plt.title("Closest Distance by Control Abstraction")
    plt.suptitle("")
    plt.ylabel("Closest Distance (m)")
    plt.xticks(rotation=-15)
    plt.tight_layout()
    plt.show()


def plot_near_miss_by_label(df: pd.DataFrame):
    d = df[df.get("interception_flag", 1) == 1].copy()
    d["level_label"] = d["level"].astype(str).str.upper()
    plt.figure(figsize=(12, 5))
    d.boxplot(column="time_in_near_miss", by="level_label", grid=False)
    plt.title("Time in Near Miss by Control Abstraction")
    plt.suptitle("")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=-15)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Domain randomization overlays (SIM vs REAL)
# ----------------------------------------------------------------------------
def _winsorize(arr, q=(0.05, 0.95)):
    arr = np.asarray(arr)
    if arr.size < 2:
        return arr
    lo, hi = np.quantile(arr, q)
    return np.clip(arr, lo, hi)


def _prep_series(df: pd.DataFrame, ycol, successes_only=False):
    d = df.copy()
    if successes_only:
        d = d[d.get("interception_flag", 1) == 1]
    return pd.to_numeric(d[ycol], errors="coerce").dropna().to_numpy()


def plot_level_sim_vs_real_boxes_with_points(
    level_label: str,
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    ycol: str,
    title: str,
    ylabel: str,
    savepath,
    successes_only: bool = False,
    logy: bool = False,
    COLOR_BY_DR: dict | None = None,
):
    COLOR_BY_DR = COLOR_BY_DR or {
        0: "#1f77b4",
        10: "#2ca02c",
        20: "#ff7f0e",
        30: "#d62728",
    }

    def _dr_color(dr):
        return COLOR_BY_DR.get(int(dr), "gray")

    simL = sim_df[sim_df["level_label"] == level_label]
    realL = real_df[real_df["level_label"] == level_label]
    if successes_only:
        simL = simL[simL["interception_flag"] == 1]
        realL = realL[realL["interception_flag"] == 1]
    if simL.empty and realL.empty:
        print(f"No data for level '{level_label}' / {ycol}.")
        return
    drs = sorted(set(simL["dr_pct"].unique()).union(set(realL["dr_pct"].unique())))
    x = np.arange(len(drs)) + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    group_w = 0.7
    box_w = group_w / 2.2
    sim_pos = x - box_w / 2
    real_pos = x + box_w / 2
    rng = np.random.default_rng(1234)

    sim_data, real_data = [], []
    sim_pts_x, sim_pts_y = [], []
    real_pts_x, real_pts_y = [], []
    for j, dr in enumerate(drs):
        s_vals_raw = _prep_series(
            simL[simL["dr_pct"] == dr], ycol, successes_only=False
        )
        r_vals_raw = _prep_series(
            realL[realL["dr_pct"] == dr], ycol, successes_only=False
        )
        if logy:
            s_vals_raw = s_vals_raw[s_vals_raw > 0]
            r_vals_raw = r_vals_raw[r_vals_raw > 0]
        s_box = s_vals_raw
        r_box = r_vals_raw
        sim_data.append(s_box if s_box.size else np.array([np.nan]))
        real_data.append(r_box if r_box.size else np.array([np.nan]))
        if s_vals_raw.size:
            jitter = box_w * 0.30
            sim_pts_x.append(
                sim_pos[j] + rng.uniform(-jitter, jitter, size=s_vals_raw.size)
            )
            sim_pts_y.append(s_vals_raw)
        if r_vals_raw.size:
            jitter = box_w * 0.30
            real_pts_x.append(
                real_pos[j] + rng.uniform(-jitter, jitter, size=r_vals_raw.size)
            )
            real_pts_y.append(r_vals_raw)

    bp_s = ax.boxplot(
        sim_data,
        positions=sim_pos,
        widths=box_w * 0.9,
        whis=1.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.2, edgecolor="black"),
    )
    for j, box in enumerate(bp_s["boxes"]):
        c = _dr_color(drs[j])
        box.set_facecolor(c)
        box.set_alpha(0.85)
        box.set_edgecolor(c)

    bp_r = ax.boxplot(
        real_data,
        positions=real_pos,
        widths=box_w * 0.9,
        whis=1.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.2, edgecolor="black"),
    )
    for j, box in enumerate(bp_r["boxes"]):
        c = _dr_color(drs[j])
        box.set_facecolor(c)
        box.set_alpha(0.40)
        box.set_edgecolor("black")
        box.set_hatch("///")

    if sim_pts_x:
        ax.scatter(
            np.concatenate(sim_pts_x),
            np.concatenate(sim_pts_y),
            s=26,
            alpha=0.75,
            facecolor="none",
            edgecolor="black",
            linewidths=0.6,
            zorder=3,
            label="_nolegend_",
        )
    if real_pts_x:
        ax.scatter(
            np.concatenate(real_pts_x),
            np.concatenate(real_pts_y),
            s=26,
            alpha=0.75,
            facecolor="white",
            edgecolor="black",
            linewidths=0.6,
            zorder=3,
            label="_nolegend_",
        )

    ax.set_title(f"{title} — {level_label}")
    ax.set_xlabel("DR level (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(d)) for d in drs])
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sim_real_ctbr_motor_side_by_side(
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    ycol: str,
    title: str,
    ylabel: str,
    savepath,
    successes_only: bool = False,
    logy: bool = False,
    COLOR_BY_DR: dict | None = None,
):
    COLOR_BY_DR = COLOR_BY_DR or {
        0: "#1f77b4",
        10: "#2ca02c",
        20: "#ff7f0e",
        30: "#d62728",
    }

    def _dr_color(dr):
        return COLOR_BY_DR.get(int(dr), "gray")

    levels = [
        lvl
        for lvl in ["CTBR", "MOTOR"]
        if lvl
        in set(sim_df.get("level_label", pd.Series(dtype=str)).unique()).union(
            set(real_df.get("level_label", pd.Series(dtype=str)).unique())
        )
    ]
    if not levels:
        print("No overlapping levels between SIM and REAL for requested plot.")
        return
    fig, axes = plt.subplots(1, len(levels), figsize=(12, 5), sharey=True)
    if len(levels) == 1:
        axes = [axes]
    for ax, level_label in zip(axes, levels):
        simL = sim_df[sim_df["level_label"] == level_label].copy()
        realL = real_df[real_df["level_label"] == level_label].copy()
        if successes_only:
            simL = simL[simL["interception_flag"] == 1]
            realL = realL[realL["interception_flag"] == 1]
        drs = sorted(set(simL["dr_pct"].unique()).union(set(realL["dr_pct"].unique())))
        x = np.arange(len(drs)) + 1
        group_w = 0.7
        box_w = group_w / 2.2
        sim_pos = x - box_w / 2
        real_pos = x + box_w / 2
        sim_data, real_data = [], []
        real_pts_x, real_pts_y = [], []
        rng = np.random.default_rng(42)
        for j, dr in enumerate(drs):
            s_vals = (
                pd.to_numeric(simL[simL["dr_pct"] == dr][ycol], errors="coerce")
                .dropna()
                .to_numpy()
            )
            r_vals = (
                pd.to_numeric(realL[realL["dr_pct"] == dr][ycol], errors="coerce")
                .dropna()
                .to_numpy()
            )
            if logy:
                s_vals = s_vals[s_vals > 0]
                r_vals = r_vals[r_vals > 0]
            s_box = s_vals
            r_box = r_vals
            sim_data.append(s_box if s_box.size else np.array([np.nan]))
            real_data.append(r_box if r_box.size else np.array([np.nan]))
            if r_vals.size:
                jitter = box_w * 0.30
                real_pts_x.append(
                    real_pos[j] + rng.uniform(-jitter, jitter, size=r_vals.size)
                )
                real_pts_y.append(r_vals)
        bp_s = ax.boxplot(
            sim_data,
            positions=sim_pos,
            widths=box_w * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            boxprops=dict(linewidth=1.2, edgecolor="black"),
        )
        for j, box in enumerate(bp_s["boxes"]):
            c = _dr_color(drs[j])
            box.set_facecolor(c)
            box.set_alpha(0.85)
            box.set_edgecolor(c)
        bp_r = ax.boxplot(
            real_data,
            positions=real_pos,
            widths=box_w * 0.9,
            whis=1.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            boxprops=dict(linewidth=1.2, edgecolor="black"),
        )
        for j, box in enumerate(bp_r["boxes"]):
            c = _dr_color(drs[j])
            box.set_facecolor(c)
            box.set_alpha(0.40)
            box.set_edgecolor("black")
            box.set_hatch("///")
        if real_pts_x:
            ax.scatter(
                np.concatenate(real_pts_x),
                np.concatenate(real_pts_y),
                s=26,
                alpha=0.75,
                facecolor="white",
                edgecolor="black",
                linewidths=0.6,
                zorder=3,
            )
        ax.set_title(level_label)
        ax.set_xlabel("DR level (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(d)) for d in drs])
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.95)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    plt.tight_layout()
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Reward smoothing line plots
# ----------------------------------------------------------------------------
def lineplot_mav_vs_gamma_one_axes_labeled(
    ag_df: pd.DataFrame,
    title: str,
    ylabel: str,
    out_path,
):
    ABS_STYLE = {
        "ctbr": {"color": "#1f77b4", "marker": "o", "label": "ctbr"},
        "motor": {"color": "#d62728", "marker": "o", "label": "motor"},
    }
    GAMMAS_KEEP = {5.0, 10.0, 15.0, 20.0}
    BASE_X = {"none": 1.5, "rate_only": 3.5}
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    JIT = 0.25
    for abs_ in ["ctbr", "motor"]:
        sub = ag_df[ag_df["abstraction"] == abs_].copy()
        style = ABS_STYLE[abs_]
        s_gamma = sub[
            (sub["smoothing_kind"] == "gamma") & (sub["gamma"].isin(GAMMAS_KEEP))
        ].sort_values("gamma")
        if not s_gamma.empty:
            ax.plot(
                s_gamma["gamma"].values,
                s_gamma["mean"].values,
                marker=style["marker"],
                lw=2.0,
                color=style["color"],
                label=style["label"],
            )
            ax.fill_between(
                s_gamma["gamma"].values,
                s_gamma["lo"].values,
                s_gamma["hi"].values,
                alpha=0.22,
                color=style["color"],
            )
        base = sub[sub["gamma"] == 0.0]
        if not base.empty:
            b_none = base[base["smoothing_kind"] == "none"]
            if not b_none.empty and np.isfinite(b_none["mean"]).all():
                x = np.full_like(
                    b_none["mean"].values,
                    BASE_X["none"] - JIT if abs_ == "CTBR" else BASE_X["none"] + JIT,
                )
                y = b_none["mean"].values
                yerr = np.vstack([y - b_none["lo"].values, b_none["hi"].values - y])
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=style["marker"],
                    capsize=3,
                    color=style["color"],
                    linestyle="none",
                    label=None,
                )
            b_rate = base[base["smoothing_kind"] == "rate_only"]
            if not b_rate.empty and np.isfinite(b_rate["mean"]).all():
                x = np.full_like(
                    b_rate["mean"].values,
                    (
                        BASE_X["rate_only"] - JIT
                        if abs_ == "ctbr"
                        else BASE_X["rate_only"] + JIT
                    ),
                )
                y = b_rate["mean"].values
                yerr = np.vstack([y - b_rate["lo"].values, b_rate["hi"].values - y])
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=style["marker"],
                    capsize=3,
                    color=style["color"],
                    linestyle="none",
                    label=None,
                )
    xticks = [BASE_X["none"], BASE_X["rate_only"]] + sorted(GAMMAS_KEEP)
    xticklabels = ["none", "rate-only"] + [
        f"{int(g)}" if float(g).is_integer() else f"{g:g}" for g in sorted(GAMMAS_KEEP)
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    for i, lbl in enumerate(ax.get_xticklabels()):
        if i < 2:
            lbl.set_rotation(-30)
        else:
            lbl.set_rotation(0)
    ax.tick_params(axis="x", pad=4)
    ax.set_xlabel(r"Reward Smoothing Method ($\gamma_s$ ≥ 0 →)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", ncol=2)
    xmin = min(BASE_X.values()) - 1.0
    xmax = max(sorted(GAMMAS_KEEP)) + 0.5
    ax.set_xlim(xmin, xmax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to :{out_path}")
    plt.close(fig)


def lineplot_sm_vs_gamma_one_axes_labeled(
    ag_df: pd.DataFrame,
    title: str,
    ylabel: str,
    out_path,
    gammas_keep: set[float] | None = None,
):
    ag = ag_df[ag_df["abstraction"].isin(["ctbr", "motor"])].copy()
    ABS_STYLE = {
        "ctbr": {"color": "#1f77b4", "marker": "o", "label": "ctbr"},
        "motor": {"color": "#d62728", "marker": "o", "label": "motor"},
    }
    if gammas_keep is None:
        g_seen = (
            ag.loc[ag["smoothing_kind"] == "gamma", "gamma"].dropna().unique().tolist()
        )
        gammas_keep = set(sorted(g_seen))
    BASE_X = {"none": 1.5, "rate_only": 3.5}
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    JIT = 0.25
    for abs_ in ["ctbr", "motor"]:
        sub = ag[ag["abstraction"] == abs_].copy()
        style = ABS_STYLE[abs_]
        s_gamma = sub[
            (sub["smoothing_kind"] == "gamma") & (sub["gamma"].isin(gammas_keep))
        ].sort_values("gamma")
        if not s_gamma.empty:
            ax.plot(
                s_gamma["gamma"].values,
                s_gamma["mean"].values,
                marker=style["marker"],
                lw=2.0,
                color=style["color"],
                label=style["label"],
            )
            ax.fill_between(
                s_gamma["gamma"].values,
                s_gamma["lo"].values,
                s_gamma["hi"].values,
                alpha=0.22,
                color=style["color"],
            )
        base = sub[sub["gamma"] == 0.0]
        if not base.empty:
            b_none = base[base["smoothing_kind"] == "none"]
            if not b_none.empty and np.isfinite(b_none["mean"]).all():
                x = np.full_like(
                    b_none["mean"].values,
                    BASE_X["none"] - JIT if abs_ == "ctbr" else BASE_X["none"] + JIT,
                    dtype=float,
                )
                y = b_none["mean"].values
                yerr = np.vstack([y - b_none["lo"].values, b_none["hi"].values - y])
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=style["marker"],
                    capsize=3,
                    color=style["color"],
                    linestyle="none",
                    label=None,
                )
            b_rate = base[base["smoothing_kind"] == "rate_only"]
            if not b_rate.empty and np.isfinite(b_rate["mean"]).all():
                x = np.full_like(
                    b_rate["mean"].values,
                    (
                        BASE_X["rate_only"] - JIT
                        if abs_ == "ctbr"
                        else BASE_X["rate_only"] + JIT
                    ),
                    dtype=float,
                )
                y = b_rate["mean"].values
                yerr = np.vstack([y - b_rate["lo"].values, b_rate["hi"].values - y])
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=style["marker"],
                    capsize=3,
                    color=style["color"],
                    linestyle="none",
                    label=None,
                )
    xticks = [BASE_X["none"], BASE_X["rate_only"]] + sorted(gammas_keep)
    xticklabels = ["none", "rate-only"] + [
        f"{int(g)}" if float(g).is_integer() else f"{g:g}" for g in sorted(gammas_keep)
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    for i, lbl in enumerate(ax.get_xticklabels()):
        if i < 2:
            lbl.set_rotation(-30)
        else:
            lbl.set_rotation(0)
    ax.tick_params(axis="x", pad=4)
    ax.set_xlabel(r"Reward Smoothing Method ($\gamma_s$ ≥ 0 →)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", ncol=2)
    xmin = min(BASE_X.values()) - 1.0
    xmax = max(sorted(gammas_keep)) + 0.5 if gammas_keep else 5.0
    ax.set_xlim(xmin, xmax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")
    plt.close(fig)


def lineplot_vs_gamma(
    metric_df: pd.DataFrame, metric_key: str, title: str, ylabel: str, out_path
):
    VIEW_COLOR = "#1f77b4"
    KIND_MARKERS = {"none": "o", "rate_only": "s", "gamma": "D"}
    KIND_LABELS = {
        "none": "No smoothing",
        "rate_only": "Rate-only",
        "gamma": "Gamma smoothing",
    }
    abstractions = list(metric_df["abstraction"].dropna().unique())
    abstractions.sort(
        key=lambda a: {"ctbr": 0, "motor": 1, "CTBR": 0, "MOTOR": 1}.get(str(a), 99)
    )
    fig, axes = plt.subplots(
        1, len(abstractions), figsize=(7.2 * len(abstractions), 5), sharey=True
    )
    if len(abstractions) == 1:
        axes = [axes]
    for ax, abs_ in zip(axes, abstractions):
        sub = metric_df[metric_df["abstraction"] == abs_]
        xs = sub["gamma"].values
        if len(xs) == 0:
            continue
        x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
        span = max(1e-9, x_max - x_min)
        jitter = 0.02 * (span if span > 0 else 1.0)
        s_gamma = sub[sub["smoothing_kind"] == "gamma"].sort_values("gamma")
        if not s_gamma.empty:
            ax.plot(
                s_gamma["gamma"].values,
                s_gamma[f"mean"].values,
                marker=KIND_MARKERS.get("gamma", "D"),
                label=KIND_LABELS["gamma"],
                color=VIEW_COLOR,
            )
            ax.fill_between(
                s_gamma["gamma"].values,
                s_gamma[f"lo"].values,
                s_gamma[f"hi"].values,
                alpha=0.22,
                color=VIEW_COLOR,
            )
        for kind, off in [("none", -jitter), ("rate_only", +jitter)]:
            s0 = sub[(sub["smoothing_kind"] == kind)]
            if not s0.empty:
                x0 = s0["gamma"].values.astype(float)
                y0 = s0[f"mean"].values
                lo = s0[f"lo"].values
                hi = s0[f"hi"].values
                ax.errorbar(
                    x0 + off,
                    y0,
                    yerr=[y0 - lo, hi - y0],
                    fmt=KIND_MARKERS.get(kind, "o"),
                    label=KIND_LABELS[kind],
                    color=VIEW_COLOR,
                    capsize=3,
                )
        ax.set_title(str(abs_).upper())
        ax.set_xlabel("Gamma (action-difference penalty)")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel(ylabel)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        fig.legend(
            handles,
            labels,
            title="Smoothing",
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.07),
        )
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- small helper: pretty metric names ----------
def _metric_label(ycol: str) -> str:
    m = {
        "first_interception_time": "First interception time",
        "time_in_near_miss": "Time in near miss",
        "closest_distance": "Closest distance",
        "interception_flag": "Interception rate",
    }
    return m.get(ycol, ycol.replace("_", " ").title())


def _per_gamma_stats(df, ycol, successes_only=False):
    d = df[
        (df["smoothing_kind"] == "gamma")
        & (pd.to_numeric(df["gamma"], errors="coerce") >= 0)
    ].copy()
    if successes_only:
        d = d[d["interception_flag"] == 1]
    d["y"] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=["y", "gamma", "level_label"])
    rows = []
    for (lvl, g), sub in d.groupby(["level_label", "gamma"]):
        y = sub["y"].to_numpy()
        if y.size == 0:
            continue
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        rows.append(
            {
                "level_label": lvl,
                "gamma": float(g),
                "q1": q1,
                "median": med,
                "q3": q3,
                "n": int(y.size),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["level_label", "gamma"]) if not out.empty else out


def _collect_baseline_samples(df, ycol, successes_only=False):
    out = {}
    for lvl, sub in df.groupby("level_label"):
        out[lvl] = {}
        for kind in ("none", "rate_only"):
            s = sub[sub["smoothing_kind"] == kind]
            if successes_only:
                s = s[s["interception_flag"] == 1]
            y = pd.to_numeric(s[ycol], errors="coerce").dropna().to_numpy()
            out[lvl][kind] = y
    return out


# Reward smoothing: overlays (SIM vs REAL) — gamma with baselines
def plot_sim_real_gamma_with_baselines_IQR(
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    ycol: str,
    title: str | None = None,
    ylabel: str | None = None,
    out_png=None,
    successes_only: bool = False,
    show_real_samples: bool = True,
    *,
    dx: float = 0.5,  # spacing between sim & real within the same baseline
    box_w: float = 0.8,  # width of each baseline box
):
    """
    Two panels (CTBR, MOTOR), baselines on the left, γ-sweep on the right.
    This version:
      - drops gamma == 15 from both sim and real
      - draws the x=0 divider as a light, dashed line
    """
    # Colors
    SIM_COLOR = "#1f77b4"
    REAL_COLOR = "#d62728"

    # Baseline group centers
    BASE_X = {"none": -0.0, "rate_only": 2.5}
    SEP_X = 0.0  # visual divider

    # ignore this gamma everywhere
    GAMMA_EXCLUDE = {15.0}

    if sim_df.empty and real_df.empty:
        print("Nothing to plot.")
        return

    # --- per-gamma IQR (drop gamma==15) ---
    sim_stats = _per_gamma_stats(sim_df, ycol, successes_only=successes_only)
    sim_stats = sim_stats[~sim_stats["gamma"].isin(GAMMA_EXCLUDE)]
    real_stats = _per_gamma_stats(real_df, ycol, successes_only=successes_only)
    real_stats = real_stats[~real_stats["gamma"].isin(GAMMA_EXCLUDE)]

    # baselines
    sim_base = _collect_baseline_samples(sim_df, ycol, successes_only=successes_only)
    real_base = _collect_baseline_samples(real_df, ycol, successes_only=successes_only)

    # Levels to show
    levels_all = list(
        {
            *sim_df.get("level_label", pd.Series(dtype=str)).dropna().unique(),
            *real_df.get("level_label", pd.Series(dtype=str)).dropna().unique(),
        }
    )
    levels = [lvl for lvl in ["CTBR", "MOTOR"] if lvl in levels_all] or levels_all
    if not levels:
        print("No levels present.")
        return

    fig, axes = plt.subplots(
        1, len(levels), figsize=(7.5 * len(levels), 5), sharey=True
    )
    if len(levels) == 1:
        axes = [axes]

    if title is None:
        title = (
            f"{_metric_label(ycol)} vs reward smoothing (baselines left, γ sweep right)"
        )
    if ylabel is None:
        ylabel = _metric_label(ycol)

    # Legend handles
    legend_handles = [
        plt.Line2D([0], [0], color=SIM_COLOR, lw=2.2, label="Simulation"),
        plt.Line2D([0], [0], color=REAL_COLOR, lw=2.2, label="Physical"),
    ]

    for ax, lvl in zip(axes, levels):
        BASE_CENTER = {"none": BASE_X["none"], "rate_only": BASE_X["rate_only"]}

        sim_data, sim_pos = [], []
        real_data, real_pos = [], []

        # --- baselines ---
        for kind in ("none", "rate_only"):
            c = BASE_CENTER[kind]
            ys_sim = sim_base.get(lvl, {}).get(kind, np.array([]))
            ys_real = real_base.get(lvl, {}).get(kind, np.array([]))

            if ys_sim.size and ys_real.size:
                sim_data.append(ys_sim)
                sim_pos.append(c - dx)
                real_data.append(ys_real)
                real_pos.append(c + dx)
            elif ys_sim.size:
                sim_data.append(ys_sim)
                sim_pos.append(c)
            elif ys_real.size:
                real_data.append(ys_real)
                real_pos.append(c)

        # sim boxes
        if sim_data:
            bp = ax.boxplot(
                sim_data,
                positions=sim_pos,
                widths=box_w,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=2.0),
                whiskerprops=dict(color="black", linewidth=1.5),
                capprops=dict(color="black", linewidth=1.5),
                boxprops=dict(linewidth=2.0, edgecolor="black"),
                zorder=3,
            )
            for b in bp["boxes"]:
                b.set_facecolor(SIM_COLOR)
                b.set_alpha(0.85)

        # real boxes
        if real_data:
            bp = ax.boxplot(
                real_data,
                positions=real_pos,
                widths=box_w,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=2.0),
                whiskerprops=dict(color="black", linewidth=1.5),
                capprops=dict(color="black", linewidth=1.5),
                boxprops=dict(linewidth=2.0, edgecolor="black"),
                zorder=3,
            )
            for b in bp["boxes"]:
                b.set_facecolor(REAL_COLOR)
                b.set_alpha(0.45)
                b.set_hatch("///")

        # real baseline scatters
        if show_real_samples and lvl in real_base:
            rng = np.random.default_rng(123)
            for kind in ("none", "rate_only"):
                yr = real_base.get(lvl, {}).get(kind, np.array([]))
                if yr.size:
                    c = BASE_CENTER[kind]
                    xs_center = c
                    if sim_base.get(lvl, {}).get(kind, np.array([])).size:
                        xs_center = c + dx
                    xr = xs_center + rng.uniform(-0.2, +0.2, size=yr.size)
                    ax.scatter(
                        xr,
                        yr,
                        s=26,
                        alpha=0.95,
                        facecolor="white",
                        edgecolor=REAL_COLOR,
                        linewidths=0.9,
                        zorder=4,
                    )

        # --- gamma sweep (sim) ---
        s_lvl = sim_stats[sim_stats["level_label"] == lvl]
        if not s_lvl.empty:
            s_lvl = s_lvl.sort_values("gamma")
            ax.fill_between(
                s_lvl["gamma"],
                s_lvl["q1"],
                s_lvl["q3"],
                alpha=0.22,
                color=SIM_COLOR,
                label="Simulation",
            )
            ax.plot(s_lvl["gamma"], s_lvl["median"], color=SIM_COLOR, lw=2.0)

        # --- gamma sweep (real) ---
        r_lvl = real_stats[real_stats["level_label"] == lvl]
        if not r_lvl.empty:
            r_lvl = r_lvl.sort_values("gamma")
            ax.fill_between(
                r_lvl["gamma"],
                r_lvl["q1"],
                r_lvl["q3"],
                alpha=0.22,
                color=REAL_COLOR,
                label="Physical",
            )
            ax.plot(r_lvl["gamma"], r_lvl["median"], color=REAL_COLOR, lw=2.0)

        # real raw gamma points (drop gamma==15 here too)
        if show_real_samples:
            r_pts = real_df[
                (real_df.get("level_label", "") == lvl)
                & (real_df.get("smoothing_kind", "") == "gamma")
            ].copy()
            r_pts["gamma"] = pd.to_numeric(r_pts["gamma"], errors="coerce")
            r_pts = r_pts[~r_pts["gamma"].isin(GAMMA_EXCLUDE)]
            if successes_only:
                r_pts = r_pts[r_pts["interception_flag"] == 1]
            r_pts["y"] = pd.to_numeric(r_pts[ycol], errors="coerce")
            r_pts = r_pts.dropna(subset=["y", "gamma"])
            if not r_pts.empty:
                gs = np.sort(r_pts["gamma"].astype(float).unique())
                if gs.size >= 2:
                    min_step = np.min(np.diff(gs))
                    gjit = 0.06 * float(min_step)
                else:
                    gjit = 0.06
                rng_g = np.random.default_rng(321)
                xs = []
                ys = []
                for gval, subg in r_pts.groupby(r_pts["gamma"].astype(float)):
                    j = rng_g.uniform(-gjit, +gjit, size=len(subg))
                    xs.append(gval + j)
                    ys.append(subg["y"].to_numpy())
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                ax.scatter(
                    xs,
                    ys,
                    s=24,
                    alpha=0.85,
                    facecolor="white",
                    edgecolor=REAL_COLOR,
                    linewidths=0.9,
                    zorder=4,
                )

        # --- x / y cosmetics ---
        gammas = []
        if not s_lvl.empty:
            gammas += s_lvl["gamma"].tolist()
        if not r_lvl.empty:
            gammas += r_lvl["gamma"].tolist()
        # drop 15 from ticks too
        g_ticks = sorted(
            set(float(g) for g in gammas if np.isfinite(g) and g not in GAMMA_EXCLUDE)
        )

        xticks = [BASE_X["none"], BASE_X["rate_only"]] + g_ticks
        xticklabels = ["none", "rate-only"] + [
            f"{int(g)}" if float(g).is_integer() else f"{g:g}" for g in g_ticks
        ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # make divider light/dashed
        ax.axvline(SEP_X, color="k", lw=0.5, alpha=0.25, linestyle="--")

        # y-lims from IQRs
        qmins, qmaxs = [], []
        for src in (sim_base.get(lvl, {}), real_base.get(lvl, {})):
            for kind in ("none", "rate_only"):
                ys = src.get(kind, np.array([]))
                if ys.size:
                    q1, q3 = np.percentile(ys, [25, 75])
                    qmins.append(q1)
                    qmaxs.append(q3)
        for sta in (s_lvl, r_lvl):
            if sta is not None and not sta.empty:
                qmins += sta["q1"].tolist()
                qmaxs += sta["q3"].tolist()
        if qmins and qmaxs:
            lo = float(np.nanmin(qmins))
            hi = float(np.nanmax(qmaxs))
            pad = 0.08 * (hi - lo if hi > lo else (abs(hi) + 1.0))
            ax.set_ylim(lo - pad - 0.25, hi + pad + 0.5)

        left_edge = min(BASE_X.values()) - dx - 1.0
        x_right = max(g_ticks) if g_ticks else 0.0
        ax.set_xlim(left_edge, x_right + 0.05 * (x_right if x_right > 0 else 1.0))
        ax.set_ylim()

        ax.set_title(lvl)
        ax.set_xlabel(r"Reward Smoothing Method ($\gamma_s ≥ 0$)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    # single legend
    axes[0].legend(handles=legend_handles, loc="best")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_png:
        out = Path(out_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("saved:", out)

    plt.show()
    plt.close(fig)
