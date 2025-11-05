import numpy as np
from src.models.pursuers.base import IPursuer


class VecPursuer_Acc_1order(IPursuer):
    def __init__(self, num_envs: int, config: dict):
        self.num_envs = num_envs
        self.dt = config["DT"]
        p_cfg = config["PURSUER"]
        self.domain_rand_pct = p_cfg.get("domain_randomization_pct", {})

        self._load_nominal_config(p_cfg)
        self._init_parameter_arrays()

        # Noise parameters
        self.pos_noise_std = p_cfg["POSITION_NOISE_STD"]
        self.vel_noise_std = p_cfg["VELOCITY_NOISE_STD"]

        # Vectorized state variables
        self.pos = np.zeros((num_envs, 3), dtype=np.float32)
        self.vel = np.zeros((num_envs, 3), dtype=np.float32)
        self.accel = np.zeros((num_envs, 3), dtype=np.float32)
        self._last_cmd = np.zeros((num_envs, 3), dtype=np.float32)

        self.reset(np.ones(num_envs, dtype=bool))

    def _load_nominal_config(self, p_cfg):
        self.tau_nom = p_cfg["ACTUATOR_TAU"]
        self.max_accel_nom = p_cfg["MAX_ACCELERATION"]
        self.max_speed_nom = p_cfg["MAX_SPEED"]
        self.init_pos_nom = np.array(p_cfg["INITIAL_POS"], dtype=np.float32)
        self.init_vel_nom = np.array(p_cfg["INITIAL_VEL"], dtype=np.float32)
        self.init_radius_nom = p_cfg["INIT_RADIUS"]

    def _init_parameter_arrays(self):
        N = self.num_envs
        self.tau = np.full(N, self.tau_nom, dtype=np.float32)
        self.max_accel = np.full(N, self.max_accel_nom, dtype=np.float32)
        self.max_speed = np.full(N, self.max_speed_nom, dtype=np.float32)
        self.init_radius_arr = np.full(N, self.init_radius_nom, dtype=np.float32)

    def _get_randomized(self, nominal: float, key: str, size: int) -> np.ndarray:
        if key not in self.domain_rand_pct:
            raise KeyError(f"{key}")
        pct = self.domain_rand_pct.get(key, 0.0)
        return np.random.uniform(1 - pct, 1 + pct, size=size) * nominal

    def _randomize_physical_parameters(self, idx):
        n = len(idx)
        self.tau[idx] = self._get_randomized(self.tau_nom, "tau_actuator", n)
        self.max_accel[idx] = self._get_randomized(self.max_accel_nom, "max_accel", n)
        self.max_speed[idx] = self._get_randomized(self.max_speed_nom, "max_speed", n)
        self.init_radius_arr[idx] = self._get_randomized(
            self.init_radius_nom, "init_radius", n
        )

    def _sample_initial_positions(self, idx):
        n = len(idx)
        dirs = np.random.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = self.init_radius_arr[idx][:, None]
        return self.init_pos_nom + radii * dirs

    def _sample_initial_velocities(self, n):
        return np.random.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)

    def reset(self, dones: np.ndarray) -> None:
        idx = np.nonzero(dones)[0]
        if len(idx) == 0:
            return

        self._randomize_physical_parameters(idx)

        self.pos[idx] = self._sample_initial_positions(idx)
        self.vel[idx] = self._sample_initial_velocities(len(idx))

        self.accel[idx] = 0.0
        self._last_cmd[idx] = 0.0

    def step(self, guidance_state: dict) -> np.ndarray:
        self._acc_cmd = self.control_law.compute_acceleration(guidance_state)
        self._acc_cmd = np.clip(self._acc_cmd, -self.max_accel, self.max_accel)

        alpha = self.dt / (self.actuator_tau + 1e-6)
        self._applied_acc += alpha * (self._acc_cmd - self._applied_acc)

        self.vel += self._applied_acc * self.dt
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel *= self.max_speed / speed
        self.pos += self.vel * self.dt

        return self._applied_acc.copy()

    def step_learn(self, command) -> None:
        accel_cmd = command

        raw_cmd = np.clip(accel_cmd, -1, 1) * self.max_accel[:, None]
        self._last_cmd = raw_cmd.copy()

        alpha = self.dt / (self.tau + self.dt)[:, None]
        self.accel = (1 - alpha) * self.accel + alpha * raw_cmd

        self.vel += self.accel * self.dt
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_speed[:, None] / (speed + 1e-8))
        self.vel *= scale

        self.pos += self.vel * self.dt

    def get_state(self) -> dict:
        noisy_pos = self.pos + np.random.normal(
            0, self.pos_noise_std, size=self.pos.shape
        )
        noisy_vel = self.vel + np.random.normal(
            0, self.vel_noise_std, size=self.vel.shape
        )

        return {
            "true_position": self.pos.copy(),
            "noisy_position": noisy_pos,
            "velocity": self.vel.copy(),
            "noisy_velocity": noisy_vel,
            "acceleration": self.accel.copy(),
            "acc_measured": self.accel.copy(),
            "rates": np.zeros_like(self.pos),
            "attitude": np.zeros_like(self.pos),
            "attitude": np.zeros_like(self.pos),
            "T_force": np.zeros((self.num_envs, 1), dtype=np.float64),
            "omega_norm": np.zeros((self.num_envs, 4), dtype=np.float64),
            "omega": np.zeros((self.num_envs, 4), dtype=np.float64),
        }
