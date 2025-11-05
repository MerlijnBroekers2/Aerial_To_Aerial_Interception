# src/models/evaders/vectorized_pliska.py
import os
import numpy as np
import pandas as pd
from src.models.evaders.base import IEvader

# ============================ Vectorized ============================


class VectorizedPliskaEvader(IEvader):
    """
    Vectorised version with precomputed, speed-scaled resampling.

    Runtime behavior:
      - step(): advances an index by 1 per env (no math).
      - get_state(): returns precomputed rows (O(1)).
    """

    def __init__(self, num_envs: int, config: dict):
        self.folder_path = config["EVADER"]["PLISKA_CSV_FOLDER"]
        self.vel_from_pos = config["EVADER"]["PLISKA_VEL_FROM_POS"]
        self.noise_std = config["EVADER"]["NOISE_STD_POS"]
        self.position_bound = config["EVADER"]["PLISKA_POSITION_BOUND"]
        self.filter_type = config["EVADER"]["FILTER_TYPE"]
        self.filter_params = config["EVADER"]["FILTER_PARAMS"]
        self.dt = float(config["DT"])
        self.eval_use_filtered_as_gt = config["EVADER"]["EVAL_USE_FILTERED_AS_GT"]

        self.speed_mult = float(config["EVADER"]["PLISKA_SPEED_MULT"])
        if self.speed_mult <= 0:
            raise ValueError("EVADER.PLISKA_SPEED_MULT must be > 0")

        if self.filter_type == "ekf":
            self.q_acc = float(self.filter_params["process_noise"])
            self.r_pos = float(self.filter_params["measurement_noise"])
        elif self.filter_type == "window_smoothing":
            self.pos_window_samples = int(
                self.filter_params.get("pos_window_samples", 0)
            )
            self.vel_window_samples = int(
                self.filter_params.get("vel_window_samples", 0)
            )
            self.vel_from_filtered_pos_for_filter = bool(
                self.filter_params.get("vel_from_filtered_pos", True)
            )

        self.max_spawn_skip_s = 5

        # ---- load & preprocess every trajectory (data timeline)
        (
            self.true_pos_data,
            self.noisy_pos_data,
            self.filtered_pos_data,
            self.true_vel_data,
            self.filtered_vel_data,
            self.times_data,
        ) = self._load_all_trajs()

        self.num_trajectories, self.traj_len, _ = self.true_pos_data.shape
        self.num_envs = int(num_envs)
        self.assigned_traj_idxs = np.array(
            [i % self.num_trajectories for i in range(self.num_envs)], dtype=np.int32
        )

        # ---- precompute resampled (simulation timeline) for all trajectories
        (
            self.rs_true_pos,  # (Ntr, Tsim, 3)
            self.rs_noisy_pos,
            self.rs_filt_pos,
            self.rs_true_vel_sim,  # scaled by speed_mult
            self.rs_filt_vel_sim,  # scaled by speed_mult
            self.rs_T_sim,
        ) = self._precompute_resampled_all()

        # runtime indices/times per env
        self.idx = np.zeros(self.num_envs, dtype=np.int32)
        self.current_time = np.zeros(self.num_envs, dtype=np.float32)
        self.shared_end_time = (
            self.rs_T_sim * self.dt
        )  # common sim duration across envs

        # --- spawn offset: choose a different start index per env
        # Max steps we’re allowed to skip at episode start
        max_skip_steps = int(
            np.clip(np.floor(self.max_spawn_skip_s / self.dt), 0, self.rs_T_sim - 1)
        )
        self._max_spawn_skip_steps = max_skip_steps
        # Per-env spawn indices in [0, max_skip_steps]
        self.spawn_idx = np.random.randint(
            0, max_skip_steps + 1, size=self.num_envs, dtype=np.int32
        )
        # runtime indices/times per env start at the per-env spawn
        self.idx = self.spawn_idx.copy()
        self.current_time = self.idx.astype(np.float32) * self.dt

    # --------------------------------------------------------------------- utils
    def _load_all_trajs(self):
        all_true, all_noisy, all_filt, all_true_vel, all_filt_vel, all_times = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        csv_files = sorted(
            f for f in os.listdir(self.folder_path) if f.endswith(".csv")
        )
        if not csv_files:
            raise ValueError(f"No csv files found in {self.folder_path}")

        for file in csv_files:
            df = pd.read_csv(os.path.join(self.folder_path, file))
            if (
                "time" not in df.columns
                or df[["pos_x", "pos_y", "pos_z"]].isna().any(axis=1).any()
            ):
                print(f"[VectorizedPliskaEvader] Skipping malformed file {file}")
                continue

            t = df["time"].values.astype(np.float32)
            t -= t[0]

            raw_pos = df[["pos_x", "pos_y", "pos_z"]].values.astype(np.float32)
            has_true_vel = all(c in df.columns for c in ["vel_x", "vel_y", "vel_z"])
            raw_vel = (
                df[["vel_x", "vel_y", "vel_z"]].values.astype(np.float32)
                if has_true_vel
                else None
            )

            # shift-origin, scale to bound
            shifted = raw_pos - raw_pos[0]
            min_vals = shifted.min(axis=0)
            max_vals = shifted.max(axis=0)
            peak_range = np.max(np.abs([min_vals, max_vals]), axis=0)
            scale_per_axis = self.position_bound / np.where(
                peak_range == 0, 1.0, peak_range
            )
            scale = float(np.min(scale_per_axis)) if np.all(peak_range > 0) else 1.0

            pos_scaled = shifted * scale

            # true vel on data timeline
            if (not self.vel_from_pos) and has_true_vel:
                vel_true = raw_vel * scale
            else:
                vel_true = np.gradient(pos_scaled, t, axis=0).astype(np.float32)

            # noise on pos
            if self.noise_std > 0:
                noise = np.random.normal(
                    0.0, self.noise_std, size=pos_scaled.shape
                ).astype(np.float32)
                noise[0] = 0.0
            else:
                noise = np.zeros_like(pos_scaled, dtype=np.float32)
            pos_noisy = (pos_scaled + noise).astype(np.float32)

            # filtering
            if self.filter_type == "passthrough":
                pos_filt = pos_noisy
                vel_filt = np.gradient(pos_filt, t, axis=0).astype(np.float32)
            elif self.filter_type == "ekf":
                pos_filt, vel_filt = self._ekf(pos_noisy, t)
            elif self.filter_type == "window_smoothing":
                if self.pos_window_samples and self.pos_window_samples > 1:
                    pos_filt = self._boxcar_smooth_axes(
                        pos_noisy, self.pos_window_samples
                    )
                else:
                    pos_filt = pos_noisy
                if self.vel_from_filtered_pos_for_filter:
                    vel_base = np.gradient(pos_filt, t, axis=0)
                else:
                    vel_base = np.gradient(pos_noisy, t, axis=0)
                if self.vel_window_samples and self.vel_window_samples > 1:
                    vel_filt = self._boxcar_smooth_axes(
                        vel_base, self.vel_window_samples
                    )
                else:
                    vel_filt = vel_base
                vel_filt = vel_filt.astype(np.float32)
            else:
                raise ValueError(f"Unknown filter_type {self.filter_type}")

            all_true.append(pos_scaled.astype(np.float32))
            all_noisy.append(pos_noisy.astype(np.float32))
            all_filt.append(pos_filt.astype(np.float32))
            all_true_vel.append(vel_true.astype(np.float32))
            all_filt_vel.append(vel_filt.astype(np.float32))
            all_times.append(t.astype(np.float32))

        # equalize length (truncate to shortest along data timeline)
        min_len = min(len(p) for p in all_true)

        def pack(lst):
            return np.stack([x[:min_len] for x in lst]).astype(np.float32)

        return (
            pack(all_true),  # (Ntr, T, 3)
            pack(all_noisy),  # (Ntr, T, 3)
            pack(all_filt),  # (Ntr, T, 3)
            pack(all_true_vel),  # (Ntr, T, 3)
            pack(all_filt_vel),  # (Ntr, T, 3)
            pack(all_times),  # (Ntr, T)
        )

    def _precompute_resampled_all(self):
        """
        Build a common simulation timeline using the shortest available data end,
        then precompute for every trajectory:
          - pos sampled at τ = t_sim * speed_mult
          - vel sampled at τ, then scaled by speed_mult
        Returns arrays with shape (Ntr, Tsim, 3).
        """
        times = self.times_data  # (Ntr, T)
        Ntr, T, _ = self.true_pos_data.shape

        # common end across trajectories
        tau_end_each = times[:, -1]
        tau_end = float(np.min(tau_end_each))
        sim_end = tau_end / self.speed_mult
        T_sim = int(np.floor(sim_end / self.dt)) + 1
        sim_times = np.arange(T_sim, dtype=np.float32) * self.dt
        tau_grid = np.clip(sim_times * self.speed_mult, times[0, 0], tau_end)

        rs_true_pos = np.empty((Ntr, T_sim, 3), dtype=np.float32)
        rs_noisy_pos = np.empty((Ntr, T_sim, 3), dtype=np.float32)
        rs_filt_pos = np.empty((Ntr, T_sim, 3), dtype=np.float32)
        rs_true_vel_sim = np.empty((Ntr, T_sim, 3), dtype=np.float32)
        rs_filt_vel_sim = np.empty((Ntr, T_sim, 3), dtype=np.float32)

        for k in range(Ntr):
            t_k = times[k]  # (T,)

            def interp_cols(Yk):
                out = np.empty((T_sim, 3), dtype=np.float32)
                for d in range(3):
                    out[:, d] = np.interp(tau_grid, t_k, Yk[:, d]).astype(np.float32)
                return out

            pos_true_k = interp_cols(self.true_pos_data[k])
            pos_noisy_k = interp_cols(self.noisy_pos_data[k])
            pos_filt_k = interp_cols(self.filtered_pos_data[k])

            vel_true_data_k = interp_cols(self.true_vel_data[k])
            vel_filt_data_k = interp_cols(self.filtered_vel_data[k])

            rs_true_pos[k] = pos_true_k
            rs_noisy_pos[k] = pos_noisy_k
            rs_filt_pos[k] = pos_filt_k
            rs_true_vel_sim[k] = (vel_true_data_k * self.speed_mult).astype(np.float32)
            rs_filt_vel_sim[k] = (vel_filt_data_k * self.speed_mult).astype(np.float32)

        return (
            rs_true_pos,
            rs_noisy_pos,
            rs_filt_pos,
            rs_true_vel_sim,
            rs_filt_vel_sim,
            T_sim,
        )

    def _boxcar_smooth_axes(self, arr: np.ndarray, window_len: int):
        T, D = arr.shape
        if T == 0:
            return arr
        k = int(max(1, window_len))
        if k % 2 == 0:
            k += 1
        if k > T:
            k = T if T % 2 == 1 else max(1, T - 1)
        if k == 1:
            return arr.copy()
        w = np.ones(k, dtype=np.float32) / float(k)
        pad = k // 2
        out = np.empty_like(arr, dtype=np.float32)
        for axis in range(D):
            x = arr[:, axis].astype(np.float32)
            if T == 1 or pad == 0:
                out[:, axis] = x
                continue
            xp = np.pad(x, (pad, pad), mode="reflect")
            y = np.convolve(xp, w, mode="valid")
            out[:, axis] = y.astype(np.float32)
        return out

    def _ekf(self, z, t):
        N = len(z)
        dt_seq = np.diff(t, prepend=t[0])
        pos_est = np.zeros_like(z, dtype=np.float32)
        vel_est = np.zeros_like(z, dtype=np.float32)
        H = np.array([[1, 0, 0]], dtype=np.float32)
        Q_base = np.diag([0.0, 0.0, self.q_acc]).astype(np.float32)
        R = np.array([[self.r_pos]], dtype=np.float32)
        for axis in range(3):
            x = np.zeros((3, N), dtype=np.float32)
            P = np.eye(3, dtype=np.float32)
            x[:, 0] = [z[0, axis], 0.0, 0.0]
            for k in range(1, N):
                dt = float(dt_seq[k])
                F = np.array(
                    [[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]], dtype=np.float32
                )
                x_pred = F @ x[:, k - 1]
                P_pred = F @ P @ F.T + Q_base
                y = z[k, axis] - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                x[:, k] = x_pred + (K @ y).flatten()
                P = (np.eye(3, dtype=np.float32) - K @ H) @ P_pred
            pos_est[:, axis] = x[0]
            vel_est[:, axis] = x[1]
        return pos_est, vel_est

    # ---------------------------------------------------------------- runtime API
    def reset(self, dones: np.ndarray):
        # resample new spawn points for the envs that are resetting
        num = int(np.count_nonzero(dones))
        if num > 0:
            self.spawn_idx[dones] = np.random.randint(
                0, self._max_spawn_skip_steps + 1, size=num, dtype=np.int32
            )
            self.idx[dones] = self.spawn_idx[dones]
            self.current_time[dones] = self.idx[dones].astype(np.float32) * self.dt

    def step(self, **kwargs):
        # advance fixed sim step per env
        self.idx = np.minimum(self.idx + 1, self.rs_T_sim - 1)
        self.current_time = self.idx.astype(np.float32) * self.dt

    def get_state(self):
        # O(1) indexing only
        j = self.assigned_traj_idxs
        i = self.idx

        true_p = np.where(
            self.eval_use_filtered_as_gt,
            self.rs_filt_pos[j, i],
            self.rs_true_pos[j, i],
        )
        true_v = np.where(
            self.eval_use_filtered_as_gt,
            self.rs_filt_vel_sim[j, i],
            self.rs_true_vel_sim[j, i],
        )

        return {
            "true_position": true_p,
            "noisy_position": self.rs_noisy_pos[j, i],
            "filtered_position": self.rs_filt_pos[j, i],
            "filtered_velocity": self.rs_filt_vel_sim[j, i],
            "velocity": true_v,
        }
