# What this does:
# - Loads an SB3 PPO policy (.zip) and inspects the FIRST linear layer on the policy path.
# - Computes per-feature importances (sum of |weights|).
# - Groups features into semantic groups (pu_pos, ev_pos, pu_vel, ev_vel, rel_*).
# - Core-only plot: excludes motors (omega), attitude_mat, and body_rates.
# - Uses per-entry MEAN within each group, then normalizes across core groups to show PERCENT SHARES.
#
# Notes:
# - Designed for your 64-64-64 PPO MLP; works for world/body variants and (optionally) LOS rate.
# ------------------------------------------------------------

from pathlib import Path
from typing import List, Optional, Tuple, Dict, OrderedDict as TOrdered
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from stable_baselines3 import PPO


# ---------- SB3 helpers ----------
def _load_ppo(model_zip: str, device: str = "cpu") -> PPO:
    model_zip = str(model_zip)
    if not Path(model_zip).exists():
        raise FileNotFoundError(f"Model not found: {model_zip}")
    model = PPO.load(model_zip, device=device)
    if not isinstance(model, PPO):
        raise TypeError("Loaded model is not PPO (did you pass the right file?)")
    return model


def _find_first_actor_linear(policy: nn.Module) -> Tuple[str, nn.Linear, int]:
    """
    For SB3 PPO MlpPolicy, prefer the first Linear in policy.mlp_extractor.policy_net.
    Returns (layer_name, layer_module, features_dim).
    """
    features_dim = getattr(policy.features_extractor, "features_dim", None)
    if features_dim is None:
        features_dim = getattr(policy, "features_dim", None)
    if features_dim is None:
        raise RuntimeError(
            "Could not determine features_dim from policy/features_extractor."
        )

    # Preferred path for PPO MlpPolicy
    try:
        seq = policy.mlp_extractor.policy_net  # nn.Sequential
        for idx, layer in enumerate(seq):
            if isinstance(layer, nn.Linear) and layer.in_features == features_dim:
                return f"mlp_extractor.policy_net[{idx}]", layer, int(features_dim)
    except Exception:
        pass

    # Fallbacks
    candidates = []
    for name, mod in policy.named_modules():
        if (
            isinstance(mod, nn.Linear)
            and getattr(mod, "in_features", None) == features_dim
        ):
            candidates.append((name, mod))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][0], candidates[0][1], int(features_dim)

    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            return name, mod, int(getattr(mod, "in_features", -1))

    raise RuntimeError("No nn.Linear layer found on the policy path.")


def _default_obs_names(dim: int) -> List[str]:
    return [f"x[{i}]" for i in range(dim)]


def _compute_importances(first_linear: nn.Linear) -> Dict[str, np.ndarray]:
    """
    Per-input importances from the first linear layer:
      - sum_abs: sum_j |W[j,i]|
      - l2: sqrt(sum_j W[j,i]^2)
      - mean_abs: mean_j |W[j,i]|
    """
    W = first_linear.weight.detach().cpu().numpy()  # [out, in]
    sum_abs = np.sum(np.abs(W), axis=0)
    l2 = np.sqrt(np.sum(W**2, axis=0))
    mean_abs = np.mean(np.abs(W), axis=0)
    return {"sum_abs": sum_abs, "l2": l2, "mean_abs": mean_abs}


# ---------- Grouping ----------
def make_groups_for_current_config(obs_names: List[str]) -> TOrdered[str, List[str]]:
    """
    Build groups by exact name membership for your current OBS_NAMES.
    Overfits to the 37-dim layout you provided; also tolerant to missing subsets.
    """
    has = set(obs_names)

    def present(names: List[str]) -> bool:
        return all(n in has for n in names)

    def ensure(names: List[str]) -> List[str]:
        missing = [n for n in names if n not in has]
        if missing:
            # Instead of hard failing, return only those present (to be robust across configs)
            return [n for n in names if n in has]
        return names

    groups = OrderedDict()
    # Absolute (world) positions/velocities per-agent
    groups["pu_pos"] = ensure(
        ["dir_pu_pos_x", "dir_pu_pos_y", "dir_pu_pos_z", "pu_pos_mag"]
    )
    groups["ev_pos"] = ensure(
        ["dir_ev_pos_x", "dir_ev_pos_y", "dir_ev_pos_z", "ev_pos_mag"]
    )
    groups["pu_vel"] = ensure(
        ["dir_pu_vel_x", "dir_pu_vel_y", "dir_pu_vel_z", "pu_vel_mag"]
    )
    groups["ev_vel"] = ensure(
        ["dir_ev_vel_x", "dir_ev_vel_y", "dir_ev_vel_z", "ev_vel_mag"]
    )

    # Relative (body/world)
    groups["rel_pos_body"] = ensure(
        [
            "dir_rel_pos_body_x",
            "dir_rel_pos_body_y",
            "dir_rel_pos_body_z",
            "rel_pos_body_mag",
        ]
    )
    groups["rel_vel_body"] = ensure(
        [
            "dir_rel_vel_body_x",
            "dir_rel_vel_body_y",
            "dir_rel_vel_body_z",
            "rel_vel_body_mag",
        ]
    )
    # If world-frame relatives exist (optional)
    groups["rel_pos"] = ensure(
        ["dir_rel_pos_x", "dir_rel_pos_y", "dir_rel_pos_z", "rel_pos_mag"]
    )
    groups["rel_vel"] = ensure(
        ["dir_rel_vel_x", "dir_rel_vel_y", "dir_rel_vel_z", "rel_vel_mag"]
    )

    # LOS rate (world or body optional)
    groups["los_ang_rate"] = ensure(
        ["los_ang_rate_x", "los_ang_rate_y", "los_ang_rate_z"]
    )
    groups["los_ang_rate_body"] = ensure(
        ["los_ang_rate_body_x", "los_ang_rate_body_y", "los_ang_rate_body_z"]
    )

    # Non-core (will be excluded from the main plot)
    groups["attitude_mat"] = ensure(
        [
            "attitude_mat_0",
            "attitude_mat_1",
            "attitude_mat_2",
            "attitude_mat_3",
            "attitude_mat_4",
            "attitude_mat_5",
        ]
    )
    groups["body_rates"] = ensure(["body_rates_x", "body_rates_y", "body_rates_z"])
    groups["omega_norm"] = ensure(
        ["omega_norm_0", "omega_norm_1", "omega_norm_2", "omega_norm_3"]
    )

    # Drop empty groups (none of the members present)
    for k in list(groups.keys()):
        if len(groups[k]) == 0:
            del groups[k]
    return groups


def group_importances_per_entry_mean(
    df_features: pd.DataFrame,
    groups: TOrdered[str, List[str]],
    metric: str = "sum_abs",
) -> pd.DataFrame:
    """
    Collapse per-feature importances into group-level *per-entry normalized* importances.
    - importance_sum : Σ metric over members
    - importance_mean: (Σ metric over members) / count  <-- used for % share
    - rel_mean       : importance_mean normalized to sum to 1 over ALL groups in 'groups'
    """
    rows = []
    for g, members in groups.items():
        sub = df_features[df_features["feature"].isin(members)]
        imp_sum = float(sub[metric].sum())
        cnt = int(len(members)) if len(members) > 0 else 1
        imp_mean = imp_sum / cnt
        rows.append(
            {
                "group": g,
                "importance_mean": imp_mean,
                "importance_sum": imp_sum,
                "count": cnt,
            }
        )
    gdf = pd.DataFrame(rows)
    total_mean = gdf["importance_mean"].sum()
    gdf["rel_mean"] = gdf["importance_mean"] / (total_mean if total_mean > 0 else 1.0)
    gdf = gdf.sort_values("importance_mean", ascending=False, ignore_index=True)
    return gdf


# Which groups are “core” (everything else excluded from the main plot)
CORE_GROUPS = {
    "pu_pos",
    "ev_pos",
    "pu_vel",
    "ev_vel",
    "rel_pos",
    "rel_vel",
    "rel_pos_body",
    "rel_vel_body",
    "los_ang_rate",
    "los_ang_rate_body",
}

# LaTeX labels per group (mathtext-friendly)
GROUP_TEX = {
    "pu_pos": r"$p_p$",
    "ev_pos": r"$p_e$",
    "pu_vel": r"$v_p$",
    "ev_vel": r"$v_e$",
    "rel_pos": r"$\Delta p$",
    "rel_vel": r"$\Delta v$",
    "rel_pos_body": r"$\Delta p_b$",
    "rel_vel_body": r"$\Delta v_b$",
    "los_ang_rate": r"$\dot{\boldsymbol{\phi}}$",
    "los_ang_rate_body": r"$\dot{\boldsymbol{\phi}}_b$",
}


def _config_order() -> List[str]:
    # order for title composition
    return [
        "pu_pos",
        "ev_pos",
        "pu_vel",
        "ev_vel",
        "rel_pos",
        "rel_pos_body",
        "rel_vel",
        "rel_vel_body",
        "los_ang_rate",
        "los_ang_rate_body",
    ]


def build_config_tex(group_keys: List[str]) -> str:
    """
    Build a mathtext config string like $[p_p,\ p_e,\ v_p,\ v_e,\ \Delta p_b,\ \Delta v_b]$
    from the present core groups. We STRIP inner $...$ to avoid nested-$ problems.
    """
    order = _config_order()
    # strip any $ in tokens to avoid nested math; also keep the spacing '\ '
    tokens = [GROUP_TEX[g].replace("$", "") for g in order if g in group_keys]
    if not tokens:
        return r"$[\ ]$"
    inner = r",\ ".join(tokens)
    return rf"$[{inner}]$"


def core_group_percentages(
    df_features: pd.DataFrame, groups: TOrdered[str, List[str]]
) -> pd.DataFrame:
    """
    Compute per-entry-mean importance per group, then normalize among CORE_GROUPS only to get %
    """
    gdf = group_importances_per_entry_mean(df_features, groups, metric="sum_abs")
    gdf = gdf[gdf["group"].isin(CORE_GROUPS)].copy()
    if gdf.empty:
        return gdf
    total = gdf["importance_mean"].sum()
    gdf["share_pct"] = 100.0 * gdf["importance_mean"] / (total if total > 0 else 1.0)
    gdf["label"] = [GROUP_TEX.get(g, g) for g in gdf["group"]]
    gdf = gdf.sort_values("share_pct", ascending=False).reset_index(drop=True)
    return gdf


def plot_core_group_pct(
    gdf_core: pd.DataFrame,
    title_tex: str,
    save_png: Optional[str] = None,
    show: bool = True,
):
    """
    Core-only horizontal bar chart in percentage (per-entry mean normalized among core groups).
    Rescales x-axis for better visibility and applies gradient coloring.
    Produces a square figure (useful for two-column report layouts).
    """
    if gdf_core.empty:
        print("No core groups found to plot.")
        return

    ordered = gdf_core.sort_values("share_pct", ascending=True)

    # Normalize values for gradient coloring
    vals = ordered["share_pct"].values
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)

    # Colormap: from light to dark (Blues as default)
    cmap = plt.cm.Blues
    colors = cmap(norm)

    # Square figure (fits two-column papers better)
    fig, ax = plt.subplots(figsize=(5, 3))  # square shape

    ax.barh(ordered["label"], ordered["share_pct"], color=colors, edgecolor="black")

    # Labels & title
    ax.set_xlabel("Importance share (%) — per-entry mean")
    ax.set_ylabel("Feature group")
    ax.set_title(title_tex)

    # Rescale x-axis for better visibility
    ax.set_xlim(10, max(vals) * 1.10)

    plt.tight_layout()
    if save_png:
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_png, dpi=300, bbox_inches="tight")  # high quality for report
    if show:
        plt.show()
    else:
        plt.close()


# ---------- Main entry ----------
def analyze_first_layer(
    model_zip: str,
    obs_names: Optional[List[str]] = None,
    device: str = "cpu",
    topk: int = 30,
    show_plot: bool = False,
    save_plot: Optional[str] = None,
    save_csv: Optional[str] = None,
    # Grouping:
    make_groups: bool = True,
    save_group_csv: Optional[str] = None,
    also_plot_stacked: bool = False,
    save_group_stacked_plot: Optional[str] = None,
    # Core-only percent figure:
    show_core_pct_plot: bool = True,
    save_core_pct_plot: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load PPO .zip, extract the first actor linear layer, compute per-feature importances,
    return a DataFrame sorted by importance, and (optionally) grouped outputs and core-only % plot.
    """
    # Load & compute
    model = _load_ppo(model_zip, device=device)
    layer_name, first_linear, in_dim = _find_first_actor_linear(model.policy)
    if in_dim <= 0:
        raise RuntimeError(
            f"Resolved first layer '{layer_name}', but couldn't infer a valid input dim."
        )

    names = (
        obs_names
        if (obs_names and len(obs_names) == in_dim)
        else _default_obs_names(in_dim)
    )
    if obs_names and len(obs_names) != in_dim:
        print(
            f"[warn] obs_names length ({len(obs_names)}) != expected ({in_dim}). Using x[i] placeholders instead."
        )

    imps = _compute_importances(first_linear)
    sum_abs = imps["sum_abs"]
    rel = sum_abs / (sum_abs.sum() + 1e-12)

    df = pd.DataFrame(
        {
            "feature": names,
            "sum_abs": sum_abs,
            "rel_importance": rel,
            "l2": imps["l2"],
            "mean_abs": imps["mean_abs"],
            "index": np.arange(len(names), dtype=int),
        }
    ).sort_values("sum_abs", ascending=False, ignore_index=True)

    # Console summary (per-feature)
    print(f"Model: {model_zip}")
    print(f"Policy net: 64-64-64")
    print(
        f"First layer: {layer_name}  weight[{first_linear.out_features}, {first_linear.in_features}]"
    )
    print("\nTop features by sum of |weights| into the first linear layer:")
    for i, row in enumerate(
        df.head(min(topk, len(df))).itertuples(index=False), start=1
    ):
        print(
            f"{i:>2}. {row.feature:>25s}  sum|w|={row.sum_abs:10.6f}  rel={100*row.rel_importance:6.2f}%"
        )

    # Optional: export per-feature CSV
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"\nSaved per-feature CSV: {save_csv}")

    # Optional: per-feature plot (kept, but off by default)
    if show_plot or save_plot:
        top = df.head(min(topk, len(df)))[::-1]
        plt.figure(figsize=(8, max(3, 0.35 * len(top))))
        plt.barh(top["feature"], top["sum_abs"])
        plt.xlabel("Sum of |weights| into first linear layer")
        plt.ylabel("Feature")
        plt.title("First-layer input importances")
        plt.tight_layout()
        if save_plot:
            Path(save_plot).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_plot, dpi=160)
            print(f"Saved per-feature plot: {save_plot}")
        plt.show() if show_plot else plt.close()

    # Grouped analysis + core-only percent plot
    if make_groups:
        groups = make_groups_for_current_config(names)
        gdf_all = group_importances_per_entry_mean(df, groups, metric="sum_abs")

        if save_group_csv:
            Path(save_group_csv).parent.mkdir(parents=True, exist_ok=True)
            gdf_all.to_csv(save_group_csv, index=False)
            print(f"Saved grouped CSV: {save_group_csv}")

        if also_plot_stacked or save_group_stacked_plot:
            # Member-level stacked bars (uses raw per-feature metrics)
            group_order = list(
                gdf_all.sort_values("importance_mean", ascending=False)["group"]
            )
            components = OrderedDict()
            for g in group_order:
                members = groups[g]
                sub = df[df["feature"].isin(members)].copy()
                sub = sub.set_index("feature").reindex(members)
                arr = sub["sum_abs"].values if not sub.empty else np.zeros(len(members))
                components[g] = arr

            plt.figure(figsize=(10, max(3, 0.6 * len(group_order))))
            bottoms = np.zeros(len(group_order))
            for i_member in range(
                max((len(v) for v in components.values()), default=0)
            ):
                vals = []
                for g in group_order:
                    arr = components[g]
                    vals.append(arr[i_member] if i_member < len(arr) else 0.0)
                plt.barh(group_order, vals, left=bottoms)
                bottoms += np.array(vals)
            plt.xlabel("sum|w| into first linear layer (member-level)")
            plt.ylabel("Group")
            plt.title("Member contributions within groups")
            plt.tight_layout()
            if save_group_stacked_plot:
                Path(save_group_stacked_plot).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_group_stacked_plot, dpi=160)
                print(f"Saved stacked plot: {save_group_stacked_plot}")
            plt.show() if also_plot_stacked else plt.close()

        # Core-only percent plot (EXCLUDES attitude_mat, body_rates, omega_norm)
        gdf_core = core_group_percentages(df, groups)
        if not gdf_core.empty:
            # Build a LaTeX-style config title from the present core groups
            # cfg_title = (
            #     "Normalized feature-weight (%) of first policy layer for"
            #     + build_config_tex(list(gdf_core["group"]))
            # )
            cfg_title = "Normalized feature-weight (%) of first policy layer"
            plot_core_group_pct(
                gdf_core,
                title_tex=cfg_title,
                save_png=save_core_pct_plot,
                show=show_core_pct_plot,
            )

            # Console dump of percentages
            print("\nCore groups (per-entry mean) — importance share (%):")
            for r in gdf_core.itertuples(index=False):
                print(f"  {r.label:>16s}: {r.share_pct:6.2f}%  (n={r.count})")

    return df


# ---------- Example: edit paths/names or import the function elsewhere ----------
if __name__ == "__main__":
    MODEL_ZIP = "/Users/merlijnbroekers/Desktop/Drone_Interception/observation_testing/models/all_body_no_phi_rate/best_model.zip"
    OBS_NAMES = [
        # 0–2
        "dir_pu_pos_x",
        "dir_pu_pos_y",
        "dir_pu_pos_z",
        # 3
        "pu_pos_mag",
        # 4–6
        "dir_ev_pos_x",
        "dir_ev_pos_y",
        "dir_ev_pos_z",
        # 7
        "ev_pos_mag",
        # 8–10
        "dir_pu_vel_x",
        "dir_pu_vel_y",
        "dir_pu_vel_z",
        # 11
        "pu_vel_mag",
        # 12–14
        "dir_ev_vel_x",
        "dir_ev_vel_y",
        "dir_ev_vel_z",
        # 15
        "ev_vel_mag",
        # 16–18
        "dir_rel_pos_body_x",
        "dir_rel_pos_body_y",
        "dir_rel_pos_body_z",
        # 19
        "rel_pos_body_mag",
        # 20–22
        "dir_rel_vel_body_x",
        "dir_rel_vel_body_y",
        "dir_rel_vel_body_z",
        # 23
        "rel_vel_body_mag",
        # 24–29
        "attitude_mat_0",
        "attitude_mat_1",
        "attitude_mat_2",
        "attitude_mat_3",
        "attitude_mat_4",
        "attitude_mat_5",
        # 30–32
        "body_rates_x",
        "body_rates_y",
        "body_rates_z",
        # 33–36
        "omega_norm_0",
        "omega_norm_1",
        "omega_norm_2",
        "omega_norm_3",
    ]

    analyze_first_layer(
        model_zip=MODEL_ZIP,
        obs_names=OBS_NAMES,
        topk=40,
        show_plot=False,  # per-feature plot (optional)
        save_plot=False,
        save_csv=None,
        make_groups=True,
        save_group_csv=None,  # e.g., "grouped_importance_mean.csv"
        also_plot_stacked=False,  # member-level stacked (optional)
        save_group_stacked_plot=None,
        show_core_pct_plot=True,  # <-- core-only percentage plot with LaTeX labels/title
        save_core_pct_plot="figures/feature_weights.png",  # e.g., "core_group_percent.png"
    )
