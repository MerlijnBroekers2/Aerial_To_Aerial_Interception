from __future__ import annotations
import numpy as np
from src.utils.helpers import split_vector
from typing import Dict, List
from typing import Optional, Tuple
import pandas as pd


def get_rot_columns(phi, theta, psi):
    """Return first two columns of the body rotation matrix (shape (N, 6))"""
    c, s = np.cos, np.sin

    # assuming phi, theta, psi are (N,) arrays
    R11 = c(theta) * c(psi)
    R21 = c(theta) * s(psi)
    R31 = -s(theta)

    R12 = s(phi) * s(theta) * c(psi) - c(phi) * s(psi)
    R22 = s(phi) * s(theta) * s(psi) + c(phi) * c(psi)
    R32 = s(phi) * c(theta)

    col1 = np.stack([R11, R21, R31], axis=1)
    col2 = np.stack([R12, R22, R32], axis=1)
    return np.concatenate([col1, col2], axis=1)


def _rot_world_to_body(phi, theta, psi):
    """Return R_wb (world→body) for each env. shape: (N,3,3)."""
    cx, cy, cz = np.cos(phi), np.cos(theta), np.cos(psi)
    sx, sy, sz = np.sin(phi), np.sin(theta), np.sin(psi)

    R = np.empty((phi.size, 3, 3))
    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = cz * sx * sy - cx * sz
    R[:, 0, 2] = sx * sz + cx * cz * sy
    R[:, 1, 0] = cy * sz
    R[:, 1, 1] = cx * cz + sx * sy * sz
    R[:, 1, 2] = cx * sy * sz - cz * sx
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cx * cy
    return R.transpose(0, 2, 1)


def get_observation_size(config):
    obs_cfg = config["OBSERVATIONS"]
    mode = obs_cfg["OBS_MODE"]
    base = 0

    mode_dims = {
        "pos": 2 * (3 + 1),  # dir+mag for pu_pos, ev_pos
        "pos+vel": 4 * (3 + 1),  # pu+ev pos+vel
        "rel_pos": 1 * (3 + 1),
        "vel": 2 * (3 + 1),
        "rel_vel": 1 * (3 + 1),
        "rel_pos_body": 1 * (3 + 1),
        "LOS_LOS_rate": 1 * (3 + 1) + 3,
        "rel_vel_body": 1 * (3 + 1),  # 3D direction + 1 magnitude
        "rel_pos+vel": 2 * (3 + 1),  # rel pos + rel vel
        "rel_pos+vel_boundaries": 2 * (3 + 1) + 3,
        "rel_pos_vel_los_rate": 2 * (3 + 1) + 3,  # above + LOS angular rate (3)
        "rel_pos+vel_body": 2 * (3 + 1),
        "rel_pos_vel_los_rate_body": 2 * (3 + 1) + 3,
        "all": 4 * (3 + 1) + 2 * (3 + 1) + 3,
        "all_body_no_phi_rate": 4 * (3 + 1) + 2 * (3 + 1),
        "all_no_phi_rate": 4 * (3 + 1) + 2 * (3 + 1),
        "all_no_phi_rate_boundaries": 4 * (3 + 1) + 2 * (3 + 1) + 3,
    }

    try:
        base = mode_dims[mode]
    except KeyError:
        raise ValueError(f"Unknown OBS_MODE: {mode}")

    # Optional feature sizes
    opt_dims = {
        "attitude": 3,
        "attitude_mat": 6,
        "rates": 3,
        "omega_norm": 4,
        "T_force": 1,
        "acc_measured": 3,
    }
    for feat in obs_cfg["OPTIONAL_FEATURES"]:
        if feat not in opt_dims:
            raise KeyError(f"Unrecognised optional feature '{feat}'")
        base += opt_dims[feat]

    total = base

    if obs_cfg["INCLUDE_HISTORY"]:
        total += obs_cfg["HISTORY_STEPS"] * base
    if obs_cfg["INCLUDE_ACTION_HISTORY"]:
        total += obs_cfg["ACTION_HISTORY_STEPS"] * obs_cfg["ACTION_DIM"]

    return base, total


def _batched(arr: np.ndarray) -> np.ndarray:
    """Ensure arr has shape (N, …); if 1-D, promote to (1, …)."""
    return arr if arr.ndim > 1 else arr[None, :]


def get_observation(config, ev_state, pu_state, obs_history=None, act_history=None):
    N = _batched(ev_state["true_position"]).shape[0]
    obs_cfg = config["OBSERVATIONS"]

    mode = obs_cfg["OBS_MODE"]
    include_history = obs_cfg["INCLUDE_HISTORY"]
    hist_steps = obs_cfg["HISTORY_STEPS"]
    include_action_history = obs_cfg["INCLUDE_ACTION_HISTORY"]
    act_steps = obs_cfg["ACTION_HISTORY_STEPS"]
    use_features = obs_cfg["OPTIONAL_FEATURES"]

    b = config["PURSUER"]["BOUNDARIES"]["ENV_BOUNDS"]
    bounds_min = np.array([min(b["x"]), min(b["y"]), min(b["z"])], dtype=np.float32)
    bounds_max = np.array([max(b["x"]), max(b["y"]), max(b["z"])], dtype=np.float32)

    # Core states
    pu_pos = _batched(pu_state["noisy_position"])
    ev_pos = _batched(ev_state["filtered_position"])
    pu_vel = _batched(pu_state["noisy_velocity"])
    ev_vel = _batched(ev_state["filtered_velocity"])

    # distance to each side along each axis; nearest-wall distance per axis
    d_to_min = pu_pos - bounds_min[None, :]  # (N,3)
    d_to_max = bounds_max[None, :] - pu_pos  # (N,3)
    dist_to_wall = np.minimum(d_to_min, d_to_max)  # (N,3)

    rel_pos = ev_pos - pu_pos
    rel_vel = ev_vel - pu_vel

    R_wb = _rot_world_to_body(*pu_state["attitude"].T)  # (N,3,3)
    rel_pos_body = (R_wb @ rel_pos[..., None]).squeeze(-1)
    rel_vel_body = (R_wb @ rel_vel[..., None]).squeeze(-1)

    R = np.linalg.norm(rel_pos, axis=1, keepdims=True)
    Ir = np.divide(rel_pos, R, out=np.zeros_like(rel_pos), where=R > 0)
    phi_dot = np.divide(
        np.cross(Ir, rel_vel), R**2, out=np.zeros_like(rel_vel), where=R > 0
    )
    phi_dot_body = (R_wb @ phi_dot[..., None]).squeeze(-1)

    features = []

    def add_dir_mag(vec):
        dir_, mag = split_vector(vec)
        features.extend([dir_, mag])

    if mode == "pos":
        add_dir_mag(pu_pos)
        add_dir_mag(ev_pos)

    elif mode == "pos+vel":
        add_dir_mag(pu_pos)
        add_dir_mag(ev_pos)
        add_dir_mag(pu_vel)
        add_dir_mag(ev_vel)

    elif mode == "rel_pos":
        add_dir_mag(rel_pos)

    elif mode == "vel":
        add_dir_mag(pu_vel)
        add_dir_mag(ev_vel)

    elif mode == "rel_vel":
        add_dir_mag(rel_vel)

    elif mode == "rel_vel_body":
        add_dir_mag(rel_vel_body)

    elif mode == "rel_pos_body":
        add_dir_mag(rel_pos_body)

    elif mode == "LOS_LOS_rate":
        add_dir_mag(rel_pos_body)
        features.append(phi_dot)

    elif mode == "rel_pos+vel":
        add_dir_mag(rel_pos)
        add_dir_mag(rel_vel)

    elif mode == "rel_pos+vel_boundaries":
        add_dir_mag(rel_pos)
        add_dir_mag(rel_vel)
        features.append(dist_to_wall)

    elif mode == "rel_pos_vel_los_rate":
        add_dir_mag(rel_pos)
        add_dir_mag(rel_vel)
        features.append(phi_dot)

    elif mode == "rel_pos+vel_body":
        add_dir_mag(rel_pos_body)
        add_dir_mag(rel_vel_body)

    elif mode == "rel_pos_vel_los_rate_body":
        add_dir_mag(rel_pos_body)
        add_dir_mag(rel_vel_body)
        features.append(phi_dot_body)

    elif mode == "all":
        add_dir_mag(pu_pos)
        add_dir_mag(ev_pos)
        add_dir_mag(pu_vel)
        add_dir_mag(ev_vel)
        add_dir_mag(rel_pos)
        add_dir_mag(rel_vel)
        features.append(phi_dot)

    elif mode == "all_no_phi_rate":
        add_dir_mag(pu_pos)
        add_dir_mag(ev_pos)
        add_dir_mag(pu_vel)
        add_dir_mag(ev_vel)
        add_dir_mag(rel_pos)
        add_dir_mag(rel_vel)

    elif mode == "all_body_no_phi_rate":
        add_dir_mag(pu_pos)
        add_dir_mag(ev_pos)
        add_dir_mag(pu_vel)
        add_dir_mag(ev_vel)
        add_dir_mag(rel_pos_body)
        add_dir_mag(rel_vel_body)

    elif mode == "all_no_phi_rate_boundaries":
        add_dir_mag(pu_pos)
        add_dir_mag(ev_pos)
        add_dir_mag(pu_vel)
        add_dir_mag(ev_vel)
        add_dir_mag(rel_pos)
        add_dir_mag(rel_vel)
        features.append(dist_to_wall)

    else:
        raise ValueError(f"Unknown OBS_MODE: {mode}")

    # Optional features (attitude must always be 6D rotmat)
    optional_feature_map = {
        "attitude": _batched(pu_state["attitude"]),
        "attitude_mat": get_rot_columns(*_batched(pu_state["attitude"]).T),
        "rates": _batched(pu_state["rates"]),
        "omega_norm": _batched(pu_state["omega_norm"]),
        "T_force": np.atleast_2d(pu_state["T_force"]),
        "acc_measured": _batched(pu_state["acc_measured"]),
    }

    for key in use_features:
        if key not in optional_feature_map:
            raise KeyError(f"Requested optional feature '{key}' not found in pu_state.")
        features.append(optional_feature_map[key])

    # Stack base obsß
    for i, f in enumerate(features):
        assert f.shape[0] == N, f"feature {i} has wrong batch dim {f.shape}"

    base_obs = np.concatenate(features, axis=1)
    obs_parts = [base_obs]

    # Histories
    if include_history and obs_history is not None:
        obs_parts.append(obs_history[:, -hist_steps:, :].reshape(N, -1))
    if include_action_history and act_history is not None:
        obs_parts.append(act_history[:, -act_steps:, :].reshape(N, -1))

    return np.concatenate(obs_parts, axis=1).astype(np.float32)


# ----------------------------------------------------------------------
def _describe_observation_layout(config: Dict) -> pd.DataFrame:
    """Return a DataFrame with start/end/size/field for the given CONFIG."""
    obs_cfg = config["OBSERVATIONS"]
    mode = obs_cfg["OBS_MODE"]

    idx = 0
    rows: List[Dict] = []

    def add(field: str, size: int) -> None:
        nonlocal idx
        rows.append({"start": idx, "end": idx + size - 1, "size": size, "field": field})
        idx += size

    # ---------- core slice --------------------------------------------
    if mode == "pos":
        add("dir_pu_pos (x,y,z)", 3)
        add("|pu_pos|", 1)
        add("dir_ev_pos (x,y,z)", 3)
        add("|ev_pos|", 1)

    elif mode == "rel_pos":
        add("dir_rel_pos (x,y,z)", 3)
        add("|rel_pos|", 1)

    elif mode == "vel":
        add("dir_vel (x,y,z)", 3)
        add("|vel|", 1)

    elif mode == "rel_vel":
        add("dir_rel_vel (x,y,z)", 3)
        add("|rel_vel|", 1)

    elif mode == "rel_vel_body":
        add("dir_rel_vel_body (x,y,z)", 3)
        add("|rel_vel_body|", 1)

    elif mode == "rel_pos_body":
        add("dir_rel_pos_body (x,y,z)", 3)
        add("|rel_pos_body|", 1)

    elif mode == "LOS_LOS_rate":
        add("dir_rel_pos_body (x,y,z)", 3)
        add("|rel_pos_body|", 1)
        add("LOS angular rate (x,y,z)", 3)

    elif mode == "pos+vel":
        add("dir_pu_pos (x,y,z)", 3)
        add("|pu_pos|", 1)
        add("dir_ev_pos (x,y,z)", 3)
        add("|ev_pos|", 1)
        add("dir_pu_vel (x,y,z)", 3)
        add("|pu_vel|", 1)
        add("dir_ev_vel (x,y,z)", 3)
        add("|ev_vel|", 1)

    elif mode == "rel_pos+vel":
        add("dir_rel_pos (x,y,z)", 3)
        add("|rel_pos|", 1)
        add("dir_rel_vel (x,y,z)", 3)
        add("|rel_vel|", 1)

    elif mode == "rel_pos+vel_boundaries":
        add("dir_rel_pos (x,y,z)", 3)
        add("|rel_pos|", 1)
        add("dir_rel_vel (x,y,z)", 3)
        add("|rel_vel|", 1)
        add("dist_to_wall (dx,dy,dz)", 3)

    elif mode == "rel_pos_vel_los_rate":
        add("dir_rel_pos (x,y,z)", 3)
        add("|rel_pos|", 1)
        add("dir_rel_vel (x,y,z)", 3)
        add("|rel_vel|", 1)
        add("LOS angular rate (x,y,z)", 3)

    elif mode == "rel_pos+vel_body":
        add("dir_rel_pos_body (x,y,z)", 3)
        add("|rel_pos_body|", 1)
        add("dir_rel_vel_body (x,y,z)", 3)
        add("|rel_vel_body|", 1)

    elif mode == "rel_pos_vel_los_rate_body":
        add("dir_rel_pos_body (x,y,z)", 3)
        add("|rel_pos_body|", 1)
        add("dir_rel_vel_body (x,y,z)", 3)
        add("|rel_vel_body|", 1)
        add("LOS angular rate (x,y,z)", 3)

    elif mode == "all":
        add("dir_pu_pos (x,y,z)", 3)
        add("|pu_pos|", 1)
        add("dir_ev_pos (x,y,z)", 3)
        add("|ev_pos|", 1)
        add("dir_pu_vel (x,y,z)", 3)
        add("|pu_vel|", 1)
        add("dir_ev_vel (x,y,z)", 3)
        add("|ev_vel|", 1)
        add("dir_rel_pos (x,y,z)", 3)
        add("|rel_pos|", 1)
        add("dir_rel_vel (x,y,z)", 3)
        add("|rel_vel|", 1)
        add("LOS angular rate (x,y,z)", 3)

    elif mode == "all_no_phi_rate":
        add("dir_pu_pos (x,y,z)", 3)
        add("|pu_pos|", 1)
        add("dir_ev_pos (x,y,z)", 3)
        add("|ev_pos|", 1)
        add("dir_pu_vel (x,y,z)", 3)
        add("|pu_vel|", 1)
        add("dir_ev_vel (x,y,z)", 3)
        add("|ev_vel|", 1)
        add("dir_rel_pos (x,y,z)", 3)
        add("|rel_pos|", 1)
        add("dir_rel_vel (x,y,z)", 3)
        add("|rel_vel|", 1)

    elif mode == "all_no_phi_rate_boundaries":
        add("dir_pu_pos (x,y,z)", 3)
        add("|pu_pos|", 1)
        add("dir_ev_pos (x,y,z)", 3)
        add("|ev_pos|", 1)
        add("dir_pu_vel (x,y,z)", 3)
        add("|pu_vel|", 1)
        add("dir_ev_vel (x,y,z)", 3)
        add("|ev_vel|", 1)
        add("dir_rel_pos (x,y,z)", 3)
        add("|rel_pos|", 1)
        add("dir_rel_vel (x,y,z)", 3)
        add("|rel_vel|", 1)
        add("dist_to_wall (dx,dy,dz)", 3)

    elif mode == "all_body_no_phi_rate":
        add("dir_pu_pos (x,y,z)", 3)
        add("|pu_pos|", 1)
        add("dir_ev_pos (x,y,z)", 3)
        add("|ev_pos|", 1)
        add("dir_pu_vel (x,y,z)", 3)
        add("|pu_vel|", 1)
        add("dir_ev_vel (x,y,z)", 3)
        add("|ev_vel|", 1)
        add("dir_rel_pos_body (x,y,z)", 3)
        add("|rel_pos_body|", 1)
        add("dir_rel_vel_body (x,y,z)", 3)
        add("|rel_vel_body|", 1)

    else:
        raise ValueError(f"Unknown OBS_MODE '{mode}'")

    # ---------- optional features -------------------------------------
    opt_dims = {
        "attitude": 3,
        "attitude_mat": 6,
        "rates": 3,
        "omega_norm": 4,
        "T_force": 1,
        "acc_measured": 3,
    }
    for feat in obs_cfg["OPTIONAL_FEATURES"]:
        if feat not in opt_dims:
            raise KeyError(f"Unrecognised optional feature '{feat}'")
        add(feat, opt_dims[feat])

    base_len = idx

    # ---------- history blocks ----------------------------------------
    if obs_cfg["INCLUDE_HISTORY"]:
        for k in range(1, obs_cfg["HISTORY_STEPS"] + 1):
            add(f"history_t-{k}", base_len)

    if obs_cfg["INCLUDE_ACTION_HISTORY"]:
        a_dim = obs_cfg["ACTION_DIM"]
        for k in range(1, obs_cfg["ACTION_HISTORY_STEPS"] + 1):
            add(f"action_t-{k}", a_dim)

    rows.append({"start": 0, "end": idx - 1, "size": idx, "field": "(total length)"})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def save_observation_layout_csv(config: Dict, csv_path: str) -> None:
    """
    Build the observation-layout table from *config* and save it to *csv_path*.
    """
    df = _describe_observation_layout(config)
    df.to_csv(csv_path, index=False)
    print(f"[obs-layout] CSV written to: {csv_path}")


class ObservationManager:
    """
    Keeps fixed-size rolling buffers so we can reproduce the same observation
    vector (core + optional features + history slices + action slices) that the
    PPO agent saw during training.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_dim, self.obs_dim = get_observation_size(cfg)

        obs_cfg = cfg["OBSERVATIONS"]
        self.max_hist = obs_cfg["MAX_HISTORY_STEPS"]
        self.act_dim = obs_cfg["ACTION_DIM"]

        # Buffers; filled in `reset()`
        self.hist_obs: Optional[np.ndarray] = None  # (B, max_hist, base_dim)
        self.hist_act: Optional[np.ndarray] = None  # (B, max_hist, act_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, batch: int = 1) -> None:
        """Zero-fill the rolling buffers (call once per episode)."""
        self.hist_obs = np.zeros(
            (batch, self.max_hist, self.base_dim), dtype=np.float32
        )
        self.hist_act = np.zeros((batch, self.max_hist, self.act_dim), dtype=np.float32)

    def push(self, obs_core: np.ndarray, action: np.ndarray) -> None:
        """
        Insert the newest core-observation (first `base_dim` elements)
        and the action just taken.  Shapes:
            obs_core : (B, base_dim)
            action   : (B, act_dim)
        """
        self.hist_obs = np.roll(self.hist_obs, -1, axis=1)
        self.hist_obs[:, -1, :] = obs_core

        self.hist_act = np.roll(self.hist_act, -1, axis=1)
        self.hist_act[:, -1, :] = action

    def build(self, ev_state, pu_state) -> np.ndarray:
        """
        Assemble the full observation vector with history and
        action-history slices (shape (B, obs_dim)).
        """
        return get_observation(
            self.cfg,
            ev_state,
            pu_state,
            obs_history=self.hist_obs,
            act_history=self.hist_act,
        )
