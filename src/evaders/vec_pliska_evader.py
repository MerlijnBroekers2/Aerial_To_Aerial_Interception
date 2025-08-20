# src/models/evaders/vectorized_pliska.py
import os
import numpy as np
import pandas as pd
from src.evaders.base import IEvader


class VectorizedPliskaEvader(IEvader):
    """
    Vectorised version of PliskaEvader – handles many environments in parallel.

    Config keys reused from scalar version:
        EVADER:
            CSV_FOLDER          folder with one or more CSV files
            NOISE_STD_POS       std-dev of Gaussian position noise (m)
            FILTER_TYPE         "passthrough" | "ekf"
            FILTER_PARAMS       dict with ekf params  (q_acc, r_pos)
            PLISKA_POSITION_BOUND   half-size of cubic arena (m)
        DT                      sim Δt
    """

    def __init__(self, num_envs: int, config: dict):
        self.folder_path = config["EVADER"]["PLISKA_CSV_FOLDER"]
        self.noise_std = config["EVADER"]["NOISE_STD_POS"]
        self.position_bound = config["EVADER"]["PLISKA_POSITION_BOUND"]
        self.filter_type = config["EVADER"]["FILTER_TYPE"]
        self.filter_params = config["EVADER"]["FILTER_PARAMS"]
        self.dt = config["DT"]

        if self.filter_type == "ekf":
            self.q_acc = self.filter_params["process_noise"]
            self.r_pos = self.filter_params["measurement_noise"]

        # ---- load and preprocess every trajectory in the folder
        (
            self.true_pos,
            self.noisy_pos,
            self.filtered_pos,
            self.true_vel,
            self.filtered_vel,
            self.times,
        ) = self._load_all_trajs()

        # trajectory bookkeeping
        self.num_trajectories, self.traj_len, _ = self.true_pos.shape
        self.num_envs = num_envs
        self.assigned_traj_idxs = np.array(
            [i % self.num_trajectories for i in range(num_envs)], dtype=np.int32
        )  # round-robin assignment
        self.idx = np.zeros(num_envs, dtype=np.int32)  # per-env cursor
        self.current_time = np.zeros(num_envs, dtype=np.float32)

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
                or df[["pos_x", "pos_y", "pos_z"]].isna().all(axis=1).any()
            ):
                print(f"[VectorizedPliskaEvader] Skipping malformed file {file}")
                continue

            t = df["time"].values.astype(np.float32)
            t -= t[0]  # start at 0 s

            raw_pos = df[["pos_x", "pos_y", "pos_z"]].values.astype(np.float32)

            # ---------- scale so every axis ≤ position_bound
            peak = np.max(np.abs(raw_pos), axis=0)
            scale = float(self.position_bound / np.max(peak))
            pos_scaled = raw_pos * scale

            # ---------- shift start to origin
            pos_scaled -= pos_scaled[0]

            # ---------- noise
            noise = np.random.normal(0, self.noise_std, size=pos_scaled.shape)
            noise[0] = 0.0
            pos_noisy = pos_scaled + noise

            # ---------- velocity (from noisy position)
            dt_seq = np.gradient(t)
            vel_noisy = np.gradient(pos_noisy, axis=0) / dt_seq[:, None]

            # ---------- filtering
            if self.filter_type == "passthrough":
                pos_filt = pos_noisy
                vel_filt = vel_noisy.copy()
            elif self.filter_type == "ekf":
                pos_filt, vel_filt = self._ekf(pos_noisy, t)
            else:
                raise ValueError(f"Unknown filter_type {self.filter_type}")

            # collect
            all_true.append(pos_scaled)
            all_noisy.append(pos_noisy)
            all_true_vel.append(vel_noisy)  # true_vel == noisy_vel by construction
            all_filt.append(pos_filt)
            all_filt_vel.append(vel_filt)
            all_times.append(t)

        # equalise length (truncate to shortest)
        min_len = min(len(p) for p in all_true)
        pack = lambda lst: np.stack([x[:min_len] for x in lst]).astype(np.float32)
        return (
            pack(all_true),
            pack(all_noisy),
            pack(all_filt),
            pack(all_true_vel),
            pack(all_filt_vel),
            pack(all_times),
        )

    # EKF identical to scalar version, vectorised per-axis
    def _ekf(self, z, t):
        N = len(z)
        dt_seq = np.diff(t, prepend=t[0])

        pos_est = np.zeros_like(z)
        vel_est = np.zeros_like(z)

        H = np.array([[1, 0, 0]])
        Q_base = np.diag([0.0, 0.0, self.q_acc])
        R = np.array([[self.r_pos]])

        for axis in range(3):
            x = np.zeros((3, N))
            P = np.eye(3)
            x[:, 0] = [z[0, axis], 0.0, 0.0]

            for k in range(1, N):
                dt = dt_seq[k]
                F = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])
                # predict
                x_pred = F @ x[:, k - 1]
                P_pred = F @ P @ F.T + Q_base
                # update
                y = z[k, axis] - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                x[:, k] = x_pred + (K @ y).flatten()
                P = (np.eye(3) - K @ H) @ P_pred

            pos_est[:, axis] = x[0]
            vel_est[:, axis] = x[1]

        return pos_est, vel_est

    # ---------------------------------------------------------------- public API
    def reset(self, dones: np.ndarray):
        """
        `dones` is a boolean array (len = num_envs) indicating which envs finished.
        Those envs are rewound to t=0 and idx=0.
        """
        self.idx[dones] = 0
        self.current_time[dones] = 0.0

    def step(self, **kwargs):
        """Advance every environment by dt."""
        self.current_time += self.dt
        # move index so that time[idx] >= current_time
        for env in range(self.num_envs):
            traj = self.assigned_traj_idxs[env]
            times = self.times[traj]
            self.idx[env] = min(
                np.searchsorted(times, self.current_time[env], side="right"),
                self.traj_len - 1,
            )

    def get_state(self):
        i = self.idx
        j = self.assigned_traj_idxs
        return {
            "true_position": self.true_pos[j, i],
            "noisy_position": self.noisy_pos[j, i],
            "filtered_position": self.filtered_pos[j, i],
            "filtered_velocity": self.filtered_vel[j, i],
            "velocity": self.true_vel[j, i],
        }
