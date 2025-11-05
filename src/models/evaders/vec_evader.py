import numpy as np
from src.models.evaders.base import IEvader


class VectorizedEvader(IEvader):
    """
    Batched evader dynamics for multi-environment simulation.
    Supports optional noisy observations and filtering (EKF or passthrough).
    """

    def __init__(self, num_envs: int, config: dict):
        self.num_envs = num_envs
        self.dt = config["DT"]
        self.max_accel = config["EVADER"]["MAX_ACCEL"]
        self.max_speed = config["EVADER"]["MAX_SPEED"]
        self.init_pos = np.array(config["EVADER"]["INIT_POS"], dtype=np.float32)
        self.init_vel = np.array(config["EVADER"]["INIT_VEL"], dtype=np.float32)

        self.noise_std = config["EVADER"]["NOISE_STD"]
        self.filter_type = config["EVADER"]["FILTER_TYPE"]
        self.filter_params = config["EVADER"]["FILTER_PARAMS"]

        if self.filter_type == "ekf":
            self.q_acc = self.filter_params["q_acc"]
            self.r_pos = self.filter_params["r_pos"]

        self.pos = np.zeros((num_envs, 3), dtype=np.float32)
        self.vel = np.zeros((num_envs, 3), dtype=np.float32)
        self._noisy_pos = np.zeros((num_envs, 3), dtype=np.float32)
        self._filtered_pos = np.zeros((num_envs, 3), dtype=np.float32)
        self._filtered_vel = np.zeros((num_envs, 3), dtype=np.float32)

        # EKF state: [pos, vel] = (6,)
        self._x_est = np.zeros((num_envs, 6), dtype=np.float32)
        self._P = np.tile(np.eye(6)[None, :, :], (num_envs, 1, 1))

        self.reset(np.ones(num_envs, dtype=bool))

    def reset(self, dones: np.ndarray) -> None:
        self.pos[dones] = self.init_pos
        self.vel[dones] = self.init_vel

        self._noisy_pos[dones] = self.pos[dones]

        self._filtered_pos[dones] = self.pos[dones]
        self._filtered_vel[dones] = self.vel[dones]

        self._x_est[dones, :3] = self.pos[dones]
        self._x_est[dones, 3:] = self.vel[dones]
        self._P[dones] = np.eye(6)

    def step(self, **kwargs) -> None:
        accel_cmd = kwargs.get("accel_cmd")
        if accel_cmd is None:
            raise ValueError("VectorizedEvader.step() requires 'accel_cmd'.")

        dt = kwargs.get("dt", self.dt)

        accel_cmd = np.clip(accel_cmd, -1, 1) * self.max_accel
        self.vel += accel_cmd * dt

        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_speed / (speed + 1e-8))
        self.vel *= scale

        self.pos += self.vel * dt

        # Apply noise
        noise = np.random.normal(0, self.noise_std, size=self.pos.shape)
        self._noisy_pos = self.pos + noise

        # Filter
        if self.filter_type == "passthrough":
            self._filtered_pos = self._noisy_pos.copy()
            self._filtered_vel = np.gradient(self._filtered_pos, dt, axis=0)
        elif self.filter_type == "ekf":
            self._apply_ekf_batch()
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

    def _apply_ekf_batch(self):
        dt = self.dt
        N = self.num_envs

        # State transition
        F = np.tile(np.eye(6), (N, 1, 1))
        F[:, :3, 3:] = np.eye(3) * dt

        Q = np.zeros((N, 6, 6), dtype=np.float32)
        Q[:, 3:, 3:] = np.eye(3) * self.q_acc

        H = np.zeros((N, 3, 6), dtype=np.float32)
        H[:, :, :3] = np.eye(3)

        R = np.eye(3) * self.r_pos
        R_batch = np.tile(R, (N, 1, 1))

        # Predict
        x_pred = (F @ self._x_est[:, :, None]).squeeze(-1)
        P_pred = F @ self._P @ F.transpose(0, 2, 1) + Q

        z = self._noisy_pos
        y = z - np.einsum("nij,nj->ni", H, x_pred)
        S = H @ P_pred @ H.transpose(0, 2, 1) + R_batch

        # Compute Kalman gain: K = P_pred @ H.T @ inv(S)
        PHt = np.einsum("nij,nkj->nik", P_pred, H)
        S_inv = np.linalg.inv(S)
        K = np.einsum("nij,njk->nik", PHt, S_inv)

        # Update state
        self._x_est = x_pred + np.einsum("nij,nj->ni", K, y)
        I = np.eye(6)
        KH = np.einsum("nij,njk->nik", K, H)
        self._P = np.einsum("nij,njk->nik", I - KH, P_pred)

        self._filtered_pos = self._x_est[:, :3]
        self._filtered_vel = self._x_est[:, 3:]

    def get_state(self) -> dict:
        return {
            "true_position": self.pos.copy(),
            "noisy_position": self._noisy_pos.copy(),
            "filtered_position": self._filtered_pos.copy(),
            "filtered_velocity": self._filtered_vel.copy(),
            "velocity": self.vel.copy(),
        }
