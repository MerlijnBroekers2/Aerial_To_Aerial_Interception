import numpy as np
from src.models.evaders.base import IEvader


class VectorizedClassicEvader(IEvader):
    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        self.path_type = config["EVADER"]["PATH_TYPE"]
        self.velocity_magnitude = config["EVADER"]["VELOCITY_MAGNITUDE"]
        self.radius = config["EVADER"]["RADIUS"]
        self.noise_std = config["EVADER"]["NOISE_STD"]
        self.dt = config["DT"]
        self.total_steps = config["TIME_LIMIT"]

        self.filter_type = config["EVADER"]["FILTER_TYPE"]
        self.filter_params = config["EVADER"]["FILTER_PARAMS"]

        self._omega = self.velocity_magnitude / self.radius if self.radius != 0 else 0.0
        self._idx = np.zeros(self.num_envs, dtype=np.int32)

        self._generate_trajectories()
        self._compute_derivatives()
        self._apply_filter()

    def reset(self, dones):
        self._idx[dones] = 0

    def step(self, **kwargs):
        self._idx = np.clip(self._idx + 1, 0, self.total_steps - 1)

    def get_state(self):
        idx = self._idx
        return {
            "true_position": self._true_positions[np.arange(self.num_envs), idx],
            "noisy_position": self._noisy_positions[np.arange(self.num_envs), idx],
            "filtered_position": self._filtered_positions[
                np.arange(self.num_envs), idx
            ],
            "filtered_velocity": self._filtered_velocities[
                np.arange(self.num_envs), idx
            ],
            "velocity": self._velocities[np.arange(self.num_envs), idx],
            "acceleration": self._accelerations[np.arange(self.num_envs), idx],
        }

    def _generate_trajectories(self):
        self._true_positions = np.zeros((self.num_envs, self.total_steps, 3))
        for i in range(self.num_envs):
            for t in range(self.total_steps):
                time = t * self.dt
                if self.path_type == "straight":
                    pos = np.array([time * self.velocity_magnitude, 0, 0])
                elif self.path_type == "circular":
                    pos = self.radius * np.array(
                        [
                            np.cos(self._omega * time),
                            np.sin(self._omega * time),
                            0.0,
                        ]
                    )
                elif self.path_type == "figure_eight":
                    pos = self.radius * np.array(
                        [
                            np.sin(self._omega * time),
                            np.sin(2 * self._omega * time),
                            0.3 * np.cos(2 * self._omega * time) - 1.0,
                        ]
                    )
                else:
                    raise ValueError(f"Invalid path_type: {self.path_type}")
                self._true_positions[i, t] = pos

        self._noisy_positions = self._true_positions + np.random.normal(
            0, self.noise_std, self._true_positions.shape
        )

    def _compute_derivatives(self):
        self._velocities = np.gradient(self._noisy_positions, self.dt, axis=1)
        self._accelerations = np.gradient(self._velocities, self.dt, axis=1)

    def _apply_filter(self):
        if self.filter_type == "passthrough":
            self._filtered_positions = self._noisy_positions.copy()
            self._filtered_velocities = np.gradient(
                self._filtered_positions, self.dt, axis=1
            )
            return

        elif self.filter_type == "ekf":
            q_acc = self.filter_params.get("q_acc", 100.0)
            r_pos = self.filter_params.get("r_pos", 1.0)

            N = self.total_steps
            self._filtered_positions = np.zeros((self.num_envs, N, 3))
            self._filtered_velocities = np.zeros((self.num_envs, N, 3))

            F = np.array([[1, self.dt, 0.5 * self.dt**2], [0, 1, self.dt], [0, 0, 1]])
            Q = np.diag([0.0, 0.0, q_acc])
            H = np.array([[1, 0, 0]])
            R = np.array([[r_pos]])

            for env in range(self.num_envs):
                z = self._noisy_positions[env]
                x_est = np.zeros((3, N, 3))  # [pos, vel, acc] for each axis
                P = np.repeat(np.eye(3)[np.newaxis, :, :], 3, axis=0)  # One P per axis

                for dim in range(3):
                    x = np.zeros((3, N))
                    x[:, 0] = [z[0, dim], 0, 0]

                    for k in range(1, N):
                        # Predict
                        x_pred = F @ x[:, k - 1]
                        P_pred = F @ P[dim] @ F.T + Q

                        # Update
                        y = z[k, dim] - (H @ x_pred)
                        S = H @ P_pred @ H.T + R
                        K = P_pred @ H.T @ np.linalg.inv(S)
                        x[:, k] = x_pred + (K @ y).flatten()
                        P[dim] = (np.eye(3) - K @ H) @ P_pred

                    x_est[:, :, dim] = x

                self._filtered_positions[env] = x_est[0]
                self._filtered_velocities[env] = x_est[1]
            return

        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
