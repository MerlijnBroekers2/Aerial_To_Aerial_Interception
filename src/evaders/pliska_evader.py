import numpy as np
import pandas as pd
from src.evaders.base import IEvader


class PliskaEvader(IEvader):
    def __init__(self, config):
        self.csv_file = config["EVADER"]["CSV_FILE"]
        self.noise_std_pos = config["EVADER"]["NOISE_STD_POS"]
        self.sim_dt = config["DT"]
        self.position_bound = config["EVADER"]["PLISKA_POSITION_BOUND"]  # in meters

        self.filter_type = config["EVADER"]["FILTER_TYPE"]
        self.filter_params = config["EVADER"]["FILTER_PARAMS"]
        self.eval_use_filtered_as_gt = config["EVADER"]["EVAL_USE_FILTERED_AS_GT"]

        self.current_time = 0.0
        self._read_data()
        self._apply_filter()
        self._idx = 0
        self.end_time = self.times[-1]

    def reset(self) -> None:
        self._idx = 0
        self.current_time = 0.0

    def step(self, **kwargs) -> None:
        self.current_time += self.sim_dt
        while (
            self._idx < len(self.times) - 1
            and self.times[self._idx] < self.current_time
        ):
            self._idx += 1

    def get_state(self) -> dict:
        true_pos = (
            self.filtered_positions[self._idx]
            if self.eval_use_filtered_as_gt
            else self.true_positions[self._idx]
        )
        return {
            "true_position": true_pos.copy(),
            "noisy_position": self.noisy_positions[self._idx].copy(),
            "filtered_position": self.filtered_positions[self._idx].copy(),
            "filtered_velocity": self.filtered_velocities[self._idx].copy(),
            "velocity": self.velocities[self._idx].copy(),
        }

    def _read_data(self) -> None:
        df = pd.read_csv(self.csv_file)
        df = df.dropna(subset=["time"])
        self.times = df["time"].values

        # === Scale positions ===
        raw_positions = df[["pos_x", "pos_y", "pos_z"]].values.astype(np.float32)
        min_vals = raw_positions.min(axis=0)
        max_vals = raw_positions.max(axis=0)
        peak_range = np.max(np.abs([min_vals, max_vals]), axis=0)
        scale_per_axis = self.position_bound / peak_range
        self.scale_factor = float(np.min(scale_per_axis))
        scaled_positions = raw_positions * self.scale_factor

        # === Shift to start at origin ===
        origin_shift = scaled_positions[0]
        self.true_positions = scaled_positions - origin_shift

        # === Add noise to position ===
        if self.noise_std_pos > 0:
            pos_noise = np.random.normal(
                0, self.noise_std_pos, size=self.true_positions.shape
            )
            pos_noise[0] = 0.0  # preserve first point
        else:
            pos_noise = np.zeros_like(self.true_positions)
        self.noisy_positions = self.true_positions + pos_noise

        # === Derive velocity from noisy positions ===
        dt = np.gradient(self.times)
        self.noisy_velocities = (
            np.gradient(self.noisy_positions, axis=0) / dt[:, np.newaxis]
        )
        self.velocities = self.noisy_velocities.copy()

    def _apply_filter(self) -> None:
        if self.filter_type == "passthrough":
            self.filtered_positions = self.noisy_positions.copy()
            self.filtered_velocities = np.gradient(
                self.filtered_positions, self.times, axis=0
            )
            return

        elif self.filter_type == "ekf":
            q_acc = self.filter_params.get("q_acc", 100.0)
            r_pos = self.filter_params.get("r_pos", 1.0)
            z = self.noisy_positions
            t = self.times
            N = len(z)

            x_est = np.zeros((N, 3))
            v_est = np.zeros((N, 3))

            for dim in range(3):
                x_axis = np.zeros((3, N))  # [pos, vel, acc]
                P = np.eye(3)
                x_hat = np.zeros((3, N))
                x_hat[:, 0] = [z[0, dim], 0.0, 0.0]

                for k in range(1, N):
                    dt = t[k] - t[k - 1]
                    F = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])
                    Q = np.diag([0.0, 0.0, q_acc])
                    H = np.array([[1, 0, 0]])
                    R = np.array([[r_pos]])

                    # Predict
                    x_pred = F @ x_axis[:, k - 1]
                    P_pred = F @ P @ F.T + Q

                    # Update
                    y = z[k, dim] - H @ x_pred
                    S = H @ P_pred @ H.T + R
                    K = P_pred @ H.T @ np.linalg.inv(S)
                    x_axis[:, k] = x_pred + (K @ y).flatten()
                    P = (np.eye(3) - K @ H) @ P_pred

                    x_hat[:, k] = x_axis[:, k]

                x_est[:, dim] = x_hat[0]
                v_est[:, dim] = x_hat[1]

            self.filtered_positions = x_est
            self.filtered_velocities = v_est
            return

        raise ValueError(f"Unknown filter type '{self.filter_type}'")
