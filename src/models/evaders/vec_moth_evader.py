import os
import numpy as np
import pandas as pd
from src.models.evaders.base import IEvader


def pats_to_ned(arr: np.ndarray) -> np.ndarray:
    """
    Convert an (..., 3) array from PATS to NED.
    Returns a *new* array – does NOT modify in place.
    """
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    return np.stack([-z, -x, -y], axis=-1)


class VectorizedMothEvader(IEvader):
    def __init__(
        self,
        num_envs: int,
        config: dict,
    ):
        self.folder_path = config["MOTH_FOLDER"]
        self.noise_std = config["EVADER"]["NOISE_STD"]
        self.num_envs = num_envs
        self.filter_type = config["EVADER"]["FILTER_TYPE"]
        self.filter_params = config["EVADER"]["FILTER_PARAMS"]
        self.dt = config["DT"]
        self.current_time = np.zeros(num_envs, dtype=np.float32)

        if self.filter_type == "ekf":
            self.q_acc = self.filter_params["process_noise"]
            self.r_pos = self.filter_params["measurement_noise"]

        self._load_and_preprocess_all()

        self.assigned_traj_idxs = np.array(
            [i % self.num_trajectories for i in range(num_envs)], dtype=np.int32
        )

        self.idx = np.zeros(num_envs, dtype=np.int32)

    def _load_and_preprocess_all(self):
        trajs = []
        times_list = []
        vel_trajs = []

        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith(".csv"):
                path = os.path.join(self.folder_path, file)
                try:
                    df = pd.read_csv(path, sep=";")
                    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
                    times = df["elapsed"].values
                    times -= times[0]

                    pos = (
                        df[["sposX_insect", "sposY_insect", "sposZ_insect"]]
                        .astype(float)
                        .values
                    )
                    pos = pats_to_ned(pos)
                    pos -= pos[0] # ! DO NOT SHIFT KEEP THE PATS POSITIONS

                    vel = (
                        df[["svelX_insect", "svelY_insect", "svelZ_insect"]]
                        .astype(float)
                        .values
                    )

                    vel = pats_to_ned(vel)

                    trajs.append(pos)
                    vel_trajs.append(vel)
                    times_list.append(times)

                except Exception as e:
                    print(f"[Error] Failed to load {file}: {e}")

        if not trajs:
            raise ValueError("No valid moth trajectories found.")

        min_len = min(len(p) for p in trajs)
        self.traj_len = min_len
        self.num_trajectories = len(trajs)

        self.true_pos = np.zeros((self.num_trajectories, min_len, 3), dtype=np.float32)
        self.noisy_pos = np.zeros_like(self.true_pos)
        self.true_vel = np.zeros_like(self.true_pos)
        self.filtered_pos = np.zeros_like(self.true_pos)
        self.filtered_vel = np.zeros_like(self.true_pos)

        for i in range(self.num_trajectories):
            pos = trajs[i][:min_len]
            vel = vel_trajs[i][:min_len]
            times = times_list[i][:min_len]

            self.true_pos[i] = pos
            self.true_vel[i] = vel

            # Add noise to position
            noise = np.random.normal(0, self.noise_std, size=pos.shape)
            noise[0] = 0.0
            self.noisy_pos[i] = pos + noise

            # Filtering
            if self.filter_type == "passthrough":
                self.filtered_pos[i] = self.noisy_pos[i]
                self.filtered_vel[i] = np.gradient(self.filtered_pos[i], times, axis=0)
            elif self.filter_type == "ekf":
                f_pos, f_vel = self._apply_ekf(self.noisy_pos[i], times)
                self.filtered_pos[i] = f_pos
                self.filtered_vel[i] = f_vel
            else:
                raise ValueError(f"Unknown filter type: {self.filter_type}")

        self.times = np.zeros((self.num_trajectories, min_len), dtype=np.float32)

        for i in range(self.num_trajectories):
            times = times_list[i][:min_len]
            self.times[i] = times

    def _apply_ekf(self, z_meas, times):
        N = len(z_meas)
        dt_seq = np.diff(times, prepend=times[0])
        x_hat = np.zeros((N, 3, 3))  # axis, state_dim, time
        pos_est = np.zeros((N, 3))
        vel_est = np.zeros((N, 3))

        for axis in range(3):
            x = np.zeros((3, N))
            P = np.eye(3)
            F = np.eye(3)
            Q = np.diag([0.0, 0.0, self.q_acc])
            H = np.array([[1, 0, 0]])
            R = np.array([[self.r_pos]])

            x[:, 0] = [z_meas[0, axis], 0.0, 0.0]
            for k in range(1, N):
                dt = dt_seq[k]
                F = np.array(
                    [
                        [1, dt, 0.5 * dt**2],
                        [0, 1, dt],
                        [0, 0, 1],
                    ]
                )
                # Predict
                x_pred = F @ x[:, k - 1]
                P_pred = F @ P @ F.T + Q

                # Update
                y = z_meas[k, axis] - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                x[:, k] = x_pred + (K @ y).flatten()
                P = (np.eye(3) - K @ H) @ P_pred

            pos_est[:, axis] = x[0]
            vel_est[:, axis] = x[1]

        return pos_est, vel_est

    def reset(self, dones: np.ndarray):
        self.idx[dones] = 0
        self.current_time[dones] = 0.0

    def step(self, **kwargs):
        self.current_time += self.dt
        for i in range(self.num_envs):
            traj_idx = self.assigned_traj_idxs[i]
            # Find index in trajectory where time is just greater than current_time. VERY INEFFICIENT FINE FOR NOW
            time_array = self.times[traj_idx]
            next_idx = np.searchsorted(time_array, self.current_time[i], side="right")
            self.idx[i] = min(next_idx, self.traj_len - 1)

    def get_state(self):
        idxs = self.idx
        trajs = self.assigned_traj_idxs

        return {
            "true_position": self.true_pos[trajs, idxs],
            "noisy_position": self.noisy_pos[trajs, idxs],
            "filtered_position": self.filtered_pos[trajs, idxs],
            "filtered_velocity": self.filtered_vel[trajs, idxs],
            "velocity": self.true_vel[trajs, idxs],
        }
