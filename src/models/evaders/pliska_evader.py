import os
import numpy as np
import pandas as pd
from src.models.evaders.base import IEvader


class PliskaEvader(IEvader):
    def __init__(self, config):
        # --- Config
        self.csv_file = config["EVADER"]["CSV_FILE"]
        self.noise_std_pos = config["EVADER"]["NOISE_STD"]
        self.sim_dt = float(config["DT"])
        self.position_bound = config["EVADER"]["PLISKA_POSITION_BOUND"]

        self.filter_type = config["EVADER"]["FILTER_TYPE"]
        self.filter_params = config["EVADER"]["FILTER_PARAMS"]
        self.eval_use_filtered_as_gt = config["EVADER"]["EVAL_USE_FILTERED_AS_GT"]

        self.vel_from_pos = config["EVADER"]["PLISKA_VEL_FROM_POS"]

        # Speed multiplier (time scaling). 1.0 = normal speed
        self.speed_mult = float(config["EVADER"]["PLISKA_SPEED_MULT"])
        if self.speed_mult <= 0:
            raise ValueError("EVADER.PLISKA_SPEED_MULT must be > 0")

        if self.filter_type == "ekf":
            self.q_acc = float(self.filter_params["process_noise"])
            self.r_pos = float(self.filter_params["measurement_noise"])
        elif self.filter_type == "window_smoothing":
            self.pos_window_samples = int(self.filter_params["pos_window_samples"])
            self.vel_window_samples = int(self.filter_params["vel_window_samples"])
            self.vel_from_filtered_pos_for_filter = bool(
                self.filter_params.get("vel_from_filtered_pos", True)
            )

        self.max_spawn_skip_s = 0  # allow spawning anywhere in the first XXX seconds

        # --- Load & filter on original data timeline
        self._read_data()  # fills *_data arrays and self.times_data
        self._apply_filter()  # fills filtered arrays on same timeline

        # --- Precompute resampled arrays on simulation timeline (fast get_state)
        self._precompute_resampled()

        # --- spawn offset: choose a start index for this single env
        max_skip_steps = int(
            np.clip(np.floor(self.max_spawn_skip_s / self.sim_dt), 0, self.rs_T_sim - 1)
        )
        self._max_spawn_skip_steps = max_skip_steps
        self.spawn_idx = int(np.random.randint(0, max_skip_steps + 1))

        # runtime index/time start at the spawn
        self.idx = self.spawn_idx
        self.current_time = self.idx * self.sim_dt

        # keep existing end_time logic
        self.end_time = self.rs_T_sim * self.sim_dt  # inclusive duration

    # ----------------------------------------------------------- runtime API
    def reset(self) -> None:
        self.spawn_idx = int(np.random.randint(0, self._max_spawn_skip_steps + 1))
        self.idx = self.spawn_idx
        self.current_time = self.idx * self.sim_dt

    def step(self, **kwargs) -> None:
        # advance fixed sim step
        if self.idx < self.rs_T_sim - 1:
            self.idx += 1
        self.current_time = self.idx * self.sim_dt

    def get_state(self) -> dict:
        # O(1): pull precomputed row
        i = self.idx
        if self.eval_use_filtered_as_gt:
            true_p = self.rs_filt_pos[i]
            true_v = self.rs_filt_vel_sim[i]
        else:
            true_p = self.rs_true_pos[i]
            true_v = self.rs_true_vel_sim[i]

        return {
            "true_position": true_p.copy(),
            "noisy_position": self.rs_noisy_pos[i].copy(),
            "filtered_position": self.rs_filt_pos[i].copy(),
            "filtered_velocity": self.rs_filt_vel_sim[i].copy(),
            "velocity": true_v.copy(),
        }

    # ----------------------------------------------------------- internals
    def _read_data(self) -> None:
        df = pd.read_csv(self.csv_file).dropna(subset=["time"])
        t = df["time"].values.astype(np.float32)
        t = t - t[0]  # start at 0 for stability
        self.times_data = t  # (T,)

        # Positions: shift to origin, scale uniformly to fit bound
        raw_pos = df[["pos_x", "pos_y", "pos_z"]].values.astype(np.float32)
        origin_shift = raw_pos[0]
        shifted_pos = raw_pos - origin_shift

        min_vals = shifted_pos.min(axis=0)
        max_vals = shifted_pos.max(axis=0)
        peak_range = np.max(np.abs([min_vals, max_vals]), axis=0)
        scale_per_axis = self.position_bound / np.where(
            peak_range == 0, 1.0, peak_range
        )
        self.scale_factor = (
            float(np.min(scale_per_axis)) if np.all(peak_range > 0) else 1.0
        )

        true_positions = shifted_pos * self.scale_factor  # (T,3)

        # TRUE vel on data timeline
        has_true_vel = all(c in df.columns for c in ["vel_x", "vel_y", "vel_z"])
        if (not self.vel_from_pos) and has_true_vel:
            raw_vel = df[["vel_x", "vel_y", "vel_z"]].values.astype(np.float32)
            velocities = (raw_vel * self.scale_factor).astype(np.float32)
        else:
            velocities = np.gradient(true_positions, self.times_data, axis=0).astype(
                np.float32
            )

        # Noisy pos
        if self.noise_std_pos > 0:
            pos_noise = np.random.normal(
                0.0, self.noise_std_pos, size=true_positions.shape
            ).astype(np.float32)
            pos_noise[0] = 0.0
        else:
            pos_noise = np.zeros_like(true_positions, dtype=np.float32)
        noisy_positions = (true_positions + pos_noise).astype(np.float32)

        # Noisy vel (unused after filtering but keep for completeness)
        noisy_velocities = np.gradient(noisy_positions, self.times_data, axis=0).astype(
            np.float32
        )

        # Stash
        self.true_positions_data = true_positions.astype(np.float32)
        self.velocities_data = velocities.astype(np.float32)
        self.noisy_positions_data = noisy_positions.astype(np.float32)
        self.noisy_velocities_data = noisy_velocities.astype(np.float32)

    def _apply_filter(self) -> None:
        t = self.times_data
        if self.filter_type == "passthrough":
            pos_filt = self.noisy_positions_data.copy()
            vel_filt = np.gradient(pos_filt, t, axis=0).astype(np.float32)
        elif self.filter_type == "ekf":
            z = self.noisy_positions_data
            N = len(t)
            x_est = np.zeros((N, 3), dtype=np.float32)
            v_est = np.zeros((N, 3), dtype=np.float32)
            H = np.array([[1, 0, 0]], dtype=np.float32)
            Q_base = np.diag([0.0, 0.0, self.q_acc]).astype(np.float32)
            R = np.array([[self.r_pos]], dtype=np.float32)
            dt_seq = np.diff(t, prepend=t[0]).astype(np.float32)
            for dim in range(3):
                x = np.zeros((3, N), dtype=np.float32)
                P = np.eye(3, dtype=np.float32)
                x[:, 0] = [z[0, dim], 0.0, 0.0]
                for k in range(1, N):
                    dt = float(dt_seq[k])
                    F = np.array(
                        [[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]],
                        dtype=np.float32,
                    )
                    x_pred = F @ x[:, k - 1]
                    P_pred = F @ P @ F.T + Q_base
                    y = z[k, dim] - (H @ x_pred)
                    S = H @ P_pred @ H.T + R
                    K = P_pred @ H.T @ np.linalg.inv(S)
                    x[:, k] = x_pred + (K @ y).flatten()
                    P = (np.eye(3, dtype=np.float32) - K @ H) @ P_pred
                x_est[:, dim] = x[0]
                v_est[:, dim] = x[1]
            pos_filt = x_est.astype(np.float32)
            vel_filt = v_est.astype(np.float32)
        elif self.filter_type == "window_smoothing":
            if getattr(self, "pos_window_samples", 0) and self.pos_window_samples > 1:
                pos_filt = self._boxcar_smooth_axes(
                    self.noisy_positions_data, self.pos_window_samples
                )
            else:
                pos_filt = self.noisy_positions_data.copy()
            if getattr(self, "vel_from_filtered_pos_for_filter", True):
                vel_base = np.gradient(pos_filt, t, axis=0)
            else:
                vel_base = self.noisy_velocities_data
            if getattr(self, "vel_window_samples", 0) and self.vel_window_samples > 1:
                vel_filt = self._boxcar_smooth_axes(vel_base, self.vel_window_samples)
            else:
                vel_filt = vel_base
            vel_filt = vel_filt.astype(np.float32)
        else:
            raise ValueError(f"Unknown filter type '{self.filter_type}'")

        self.filtered_positions_data = pos_filt.astype(np.float32)
        self.filtered_velocities_data = vel_filt.astype(np.float32)

    def _precompute_resampled(self) -> None:
        """
        Build a uniform simulation timeline and precompute:
          - positions sampled at τ = t_sim * speed_mult
          - velocities sampled at τ then scaled by speed_mult
        Results: arrays of shape (T_sim, 3) for fast O(1) access.
        """
        t_data = self.times_data
        tau_end = float(t_data[-1])  # max data time
        sim_end = tau_end / self.speed_mult  # max sim time
        T_sim = int(np.floor(sim_end / self.sim_dt)) + 1
        sim_times = np.arange(T_sim, dtype=np.float32) * self.sim_dt
        tau_grid = np.clip(sim_times * self.speed_mult, t_data[0], t_data[-1])

        def interp_cols(Y, tau):
            # Y: (T,3), tau: (T_sim,)
            out = np.empty((T_sim, 3), dtype=np.float32)
            for d in range(3):
                out[:, d] = np.interp(tau, t_data, Y[:, d]).astype(np.float32)
            return out

        # positions on data timeline
        self.rs_true_pos = interp_cols(self.true_positions_data, tau_grid)
        self.rs_noisy_pos = interp_cols(self.noisy_positions_data, tau_grid)
        self.rs_filt_pos = interp_cols(self.filtered_positions_data, tau_grid)

        # velocities on data timeline → scale to simulation timeline
        true_v_data = interp_cols(self.velocities_data, tau_grid)
        filt_v_data = interp_cols(self.filtered_velocities_data, tau_grid)
        self.rs_true_vel_sim = (true_v_data * self.speed_mult).astype(np.float32)
        self.rs_filt_vel_sim = (filt_v_data * self.speed_mult).astype(np.float32)

        self.rs_T_sim = T_sim

    # ------------------------------ helpers
    def _boxcar_smooth_axes(self, arr: np.ndarray, window_len: int) -> np.ndarray:
        T, D = arr.shape
        if T == 0:
            return arr
        k = int(max(1, window_len))
        if k % 2 == 0:
            k += 1
        if k > T:
            k = T if (T % 2 == 1) else max(1, T - 1)
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
