import numpy as np
import pandas as pd
from src.evaders.base import IEvader


def enu_to_ned(arr: np.ndarray) -> np.ndarray:
    """
    Convert an (..., 3) array from ENU to NED.
    Returns a *new* array – does NOT modify in place.
    """
    e, n, u = arr[..., 0], arr[..., 1], arr[..., 2]
    return np.stack([n, e, -u], axis=-1)


class MothEvader(IEvader):
    def __init__(
        self,
        config,
    ):
        self.csv_file = config["EVADER"]["CSV_FILE"]
        self.noise_std = config["EVADER"]["NOISE_STD"]
        self.filter_type = config["EVADER"]["FILTER_TYPE"]
        self.filter_params = config["EVADER"]["FILTER_PARAMS"]
        self.sim_dt = config["DT"]
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

        # Find the next data index where dataset time is >= simulation time THIS IS VERY INEFFICIENT BUT SINCE THIS IS NOT USED FOR TRAINING ONLY EVAL FINE FOR NOW
        while (
            self._idx < len(self.times) - 1
            and self.times[self._idx] < self.current_time
        ):
            self._idx += 1

    def get_state(self) -> dict:
        return {
            "true_position": self.true_positions[self._idx].copy(),
            "noisy_position": self.noisy_positions[self._idx].copy(),
            "filtered_position": self.filtered_positions[self._idx].copy(),
            "filtered_velocity": self.filtered_velocities[self._idx].copy(),
            "velocity": self.velocities[self._idx].copy(),
        }

    def _read_data(self) -> None:
        df = pd.read_csv(self.csv_file, sep=";")
        df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
        raw = df["elapsed"].values
        self.times = raw - raw[0]

        # Load and center positions
        pos = df[["sposX_insect", "sposY_insect", "sposZ_insect"]].astype(float).values
        pos = enu_to_ned(pos)
        pos -= pos[0]
        self.true_positions = pos

        # Load velocities directly
        self.velocities = (
            df[["svelX_insect", "svelY_insect", "svelZ_insect"]].astype(float).values
        )
        self.velocities = enu_to_ned(self.velocities)

        # Add noise to positions only
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=pos.shape)
            noise[0] = 0.0  # preserve starting point
            self.noisy_positions = pos + noise
        else:
            self.noisy_positions = pos

    def _apply_filter(self) -> None:
        if self.filter_type == "passthrough":
            self.filtered_positions = self.noisy_positions.copy()
            self.filtered_velocities = np.gradient(
                self.filtered_positions, self.times, axis=0
            )
            return

        if self.filter_type == "ekf":
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
                    F = np.array(
                        [
                            [1, dt, 0.5 * dt**2],
                            [0, 1, dt],
                            [0, 0, 1],
                        ]
                    )
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

    def apply_shift(self, shift_vec):
        # Apply shift to all internal position arrays
        self.true_positions += shift_vec
        self.noisy_positions += shift_vec
        self.filtered_positions += shift_vec
