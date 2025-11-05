import numpy as np
from src.models.pursuers.base import IPursuer
from src.control_laws.control_laws import ControlLaw


class Acc_1order_Pursuer(IPursuer):
    """
    Point-mass pursuer with noisy sensors and delayed actuator.
    """

    def __init__(self, config: dict, control_law: ControlLaw):
        self.dt = config["DT"]
        self.control_law = control_law

        p_cfg = config["PURSUER"]

        # Nominal values
        self.max_accel = p_cfg["MAX_ACCELERATION"]
        self.max_speed = p_cfg["MAX_SPEED"]
        self.actuator_tau = p_cfg["ACTUATOR_TAU"]
        self.init_pos = np.array(p_cfg["INITIAL_POS"], dtype=np.float32)
        self.init_vel = np.array(p_cfg["INITIAL_VEL"], dtype=np.float32)

        self.pos_noise_std = p_cfg["POSITION_NOISE_STD"]
        self.vel_noise_std = p_cfg["VELOCITY_NOISE_STD"]
        self.init_radius = p_cfg["INIT_RADIUS"]

        # Dynamics state
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self._applied_acc = np.zeros(3, dtype=np.float32)
        self._acc_cmd = np.zeros(3, dtype=np.float32)

        # Init state
        self.reset()

    def reset(self) -> None:
        # Initialize position on a sphere
        rng = np.random.default_rng()
        rand_dir = rng.normal(size=3)
        rand_dir /= np.linalg.norm(rand_dir) + 1e-6
        self.pos = self.init_pos + self.init_radius * rand_dir

        self.vel = rng.uniform(-0.5, 0.5, size=3).astype(np.float32)

        self._applied_acc[:] = 0.0
        self._acc_cmd[:] = 0.0

    def step(self, command):
        acceleration_cmd = command * self.max_accel
        self._acc_cmd = acceleration_cmd.copy()

        alpha = self.dt / (self.actuator_tau + 1e-6)
        self._applied_acc += alpha * (self._acc_cmd - self._applied_acc)

        self.vel += self._applied_acc * self.dt
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel *= self.max_speed / speed
        self.pos += self.vel * self.dt

        return self._applied_acc.copy()

    def get_state(self) -> dict:
        noisy_pos = self.pos + np.random.normal(0, self.pos_noise_std, 3)
        noisy_vel = self.vel + np.random.normal(0, self.vel_noise_std, 3)

        return {
            "true_position": self.pos.copy(),
            "noisy_position": noisy_pos,
            "velocity": self.vel.copy(),
            "noisy_velocity": noisy_vel,
            "acceleration": self._applied_acc.copy(),
            "acc_command": self._acc_cmd.copy(),
            "acc_command_filtered": self._applied_acc.copy(),  # no filter used
            "acc_measured": self._applied_acc.copy(),
            # Dummy attitude
            "attitude": np.zeros(3, dtype=np.float32),
            "attitude_commanded": np.zeros(3, dtype=np.float32),
            # No angular control → zero rates
            "rates": np.zeros(3, dtype=np.float32),
            "rates_command": np.zeros(3, dtype=np.float32),
            # No thrust modeling
            "T_norm": 0.0,
            "T_force": 0.0,
            "T_command": 0.0,
            "omega": np.zeros(4, dtype=np.float32),
            "omega_norm": np.zeros(4, dtype=np.float32),
        }
