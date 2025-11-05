import numpy as np
from src.models.evaders.base import IEvader


class ClassicEvader(IEvader):
    def __init__(
        self,
        path_type: str = "straight",
        velocity_magnitude: float = 1.0,
        radius: float = 5.0,
        noise_std: float = 0.0,
        total_time: float = 30.0,
        dt: float = 0.01,
        filter_type: str = "passthrough",
        filter_params: dict = {},
    ):
        self.path_type = path_type
        self.velocity_magnitude = velocity_magnitude
        self.radius = radius
        self.omega = self.velocity_magnitude / self.radius if self.radius != 0 else 0.0
        self.noise_std = noise_std
        self.dt = dt
        self.times = np.arange(0.0, total_time + dt, dt)

        self.filter_type = filter_type
        self.filter_params = filter_params

        self._generate_trajectory()
        self._compute_derivatives()
        self._apply_filter()

        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def step(self, **kwargs) -> None:
        if self._idx < len(self.times) - 1:
            self._idx += 1

    def get_state(self) -> dict:
        return {
            "true_position": self._true_positions[self._idx].copy(),
            "noisy_position": self._noisy_positions[self._idx].copy(),
            "filtered_position": self._filtered_positions[self._idx].copy(),
            "filtered_velocity": self._filtered_velocities[self._idx].copy(),
            "velocity": self._velocities[self._idx].copy(),
            "acceleration": self._accelerations[self._idx].copy(),
        }

    def _generate_trajectory(self) -> None:
        self._true_positions = []
        self._noisy_positions = []
        for t in self.times:
            if self.path_type == "straight":
                pos = self._straight_line(t)
            elif self.path_type == "circular":
                pos = self._circular(t)
            elif self.path_type == "figure_eight":
                pos = self._figure_eight(t)
            else:
                raise ValueError(f"Unknown path_type='{self.path_type}'")

            noise = (
                np.random.normal(0, self.noise_std, size=3)
                if self.noise_std > 0
                else np.zeros(3)
            )

            self._true_positions.append(pos)
            self._noisy_positions.append(pos + noise)

        self._true_positions = np.stack(self._true_positions)
        self._noisy_positions = np.stack(self._noisy_positions)

    def _compute_derivatives(self) -> None:
        self._velocities = np.gradient(self._noisy_positions, self.dt, axis=0)
        self._accelerations = np.gradient(self._velocities, self.dt, axis=0)

    def _apply_filter(self) -> None:
        if self.filter_type == "passthrough":
            self._filtered_positions = self._noisy_positions.copy()
            self._filtered_velocities = np.gradient(
                self._filtered_positions, self.dt, axis=0
            )
            return

        if self.filter_type == "ekf":
            q_acc = self.filter_params.get("q_acc", 100.0)
            r_pos = self.filter_params.get("r_pos", 1.0)
            dt = self.dt
            z = self._noisy_positions

            N = len(z)
            x_est = np.zeros((N, 3))
            v_est = np.zeros((N, 3))

            # Loop over each axis independently (x, y, z)
            for dim in range(3):
                x_axis = np.zeros((3, N))  # [pos, vel, acc]
                P = np.eye(3)

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

                x_hat = np.zeros((3, N))
                x_hat[:, 0] = [z[0, dim], 0.0, 0.0]

                for k in range(1, N):
                    # Predict
                    x_pred = F @ x_axis[:, k - 1]
                    P_pred = F @ P @ F.T + Q

                    # Update
                    y = z[k, dim] - H @ x_pred
                    S = H @ P_pred @ H.T + R
                    K = P_pred @ H.T @ np.linalg.inv(S)
                    x_axis[:, k] = x_pred + (K * y).flatten()
                    P = (np.eye(3) - K @ H) @ P_pred

                    x_hat[:, k] = x_axis[:, k]

                x_est[:, dim] = x_hat[0]
                v_est[:, dim] = x_hat[1]

            self._filtered_positions = x_est
            self._filtered_velocities = v_est
            return

        raise ValueError(f"Unknown filter type '{self.filter_type}'")

    def _straight_line(self, t: float) -> np.ndarray:
        x = t * self.velocity_magnitude
        return np.array([x, 0.0, 0.0], dtype=float)

    def _circular(self, t: float) -> np.ndarray:
        x = self.radius * np.cos(self.omega * t)
        y = self.radius * np.sin(self.omega * t)
        return np.array([x, y, 0.0], dtype=float)

    def _figure_eight(self, t: float) -> np.ndarray:
        x = self.radius * np.sin(self.omega * t)
        y = self.radius * np.sin(2 * self.omega * t)
        z = self.radius * 0.3 * np.cos(2 * self.omega * t) - 1.0
        return np.array([x, y, z], dtype=float)
