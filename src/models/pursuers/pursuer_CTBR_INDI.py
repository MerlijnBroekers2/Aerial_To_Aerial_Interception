import numpy as np
from sympy import Matrix, symbols, Array, cos, sin, tan, lambdify
from src.models.pursuers.base import IPursuer


class CTBR_INDI_Pursuer(IPursuer):
    def __init__(self, config: dict, control_law):
        self.dt = config["DT"]
        self._base_config = config["PURSUER"]
        self.control_law = control_law

        self._load_nominal_parameters()

        self.state = np.zeros(13, dtype=np.float64)
        self.initial_state = np.zeros_like(self.state)
        self.acc_measured = np.zeros(3, dtype=np.float64)
        self.p_command = 0.0
        self.q_command = 0.0
        self.r_command = 0.0
        self.T_command = 0.0
        self.rates_commanded = np.zeros(3, dtype=np.float64)

        self.reset()

    def _load_nominal_parameters(self):
        c = self._base_config
        self.g = c["gravity"]
        self.drag_params = c["drag"]

        ac = c["actuator_time_constants"]
        self.taup = ac["p"]
        self.tauq = ac["q"]
        self.taur = ac["r"]
        self.tauT = ac["T"]

        al = c["actuator_limits"]
        self.p_min, self.p_max = al["p"]
        self.q_min, self.q_max = al["q"]
        self.r_min, self.r_max = al["r"]
        self.T_min, self.T_max = al["T"]

        self.pos_noise_std = c["POSITION_NOISE_STD"]
        self.vel_noise_std = c["VELOCITY_NOISE_STD"]

        self.init_pos = np.array(c["INITIAL_POS"], dtype=np.float64)
        self.init_vel = np.array(c["INITIAL_VEL"], dtype=np.float64)
        self.init_att = np.array(c["INITIAL_ATTITUDE"], dtype=np.float64)
        self.init_rates = np.array(c["INITIAL_RATES"], dtype=np.float64)
        self.init_radius = c["INIT_RADIUS"]

    def _build_state_space_model(self):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, T_norm = symbols(
            "x y z v_x v_y v_z phi theta psi p q r T_norm"
        )
        state = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, T_norm]
        p_cmd, q_cmd, r_cmd, T_cmd = symbols("p_cmd q_cmd r_cmd T_cmd")

        kx = self.drag_params["kx_acc_ctbr"]
        ky = self.drag_params["ky_acc_ctbr"]

        Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
        Ry = Matrix(
            [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
        )
        Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
        R = Rz * Ry * Rx

        vb = R.T * Matrix([vx, vy, vz])
        Dx = -kx * vb[0]
        Dy = -ky * vb[1]

        d_x, d_y, d_z = vx, vy, vz
        T = (T_norm + 1) / 2 * (self.T_max - self.T_min) + self.T_min
        dv = Matrix([0, 0, +self.g]) + R * Matrix([Dx, Dy, -T])
        d_vx, d_vy, d_vz = dv

        d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        d_theta = q * cos(phi) - r * sin(phi)
        d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

        d_p = (
            ((p_cmd + 1) / 2 * (self.p_max - self.p_min) + self.p_min) - p
        ) / self.taup
        d_q = (
            ((q_cmd + 1) / 2 * (self.q_max - self.q_min) + self.q_min) - q
        ) / self.tauq
        d_r = (
            ((r_cmd + 1) / 2 * (self.r_max - self.r_min) + self.r_min) - r
        ) / self.taur
        d_Tn = (T_cmd - T_norm) / self.tauT

        return lambdify(
            (Array(state), Array([p_cmd, q_cmd, r_cmd, T_cmd])),
            Array(
                [
                    d_x,
                    d_y,
                    d_z,
                    d_vx,
                    d_vy,
                    d_vz,
                    d_phi,
                    d_theta,
                    d_psi,
                    d_p,
                    d_q,
                    d_r,
                    d_Tn,
                ]
            ),
            "numpy",
        )

    def step(self, command):
        control_vec = command
        self.p_command = control_vec[0] * self.p_max
        self.q_command = control_vec[1] * self.q_max
        self.r_command = control_vec[2] * self.r_max
        self.T_command = control_vec[3]

        self.rates_commanded = np.array(
            [self.p_command, self.q_command, self.r_command]
        )

        return self._apply_control(control_vec)

    def step_learn(self, control_vec):
        return self._apply_control(control_vec)

    def _apply_control(self, control_vec):
        deriv = np.array(self.f_func(self.state, control_vec)).flatten()
        self.state += deriv * self.dt
        self.acc_measured = deriv[3:6]
        return self.acc_measured.copy()

    def get_state(self):
        pos = self.state[0:3]
        vel = self.state[3:6]
        phi, theta, psi = self.state[6:9]
        p, q, r = self.state[9:12]
        T_norm = self.state[12]
        T_force = (T_norm + 1) / 2 * (self.T_max - self.T_min) + self.T_min

        return {
            "true_position": pos.copy(),
            "noisy_position": pos + np.random.normal(0, self.pos_noise_std, 3),
            "velocity": vel.copy(),
            "noisy_velocity": vel + np.random.normal(0, self.vel_noise_std, 3),
            "acceleration": self.acc_measured.copy(),
            "acc_command": np.zeros(3),  # Not used in CTBR
            "acc_command_filtered": np.zeros(3),  # Not used in CTBR
            "acc_measured": self.acc_measured.copy(),
            "attitude": np.array([phi, theta, psi]),
            "attitude_commanded": np.zeros(3),  # Not computed in CTBR
            "rates": np.array([p, q, r]),
            "rates_command": self.rates_commanded.copy(),
            "T_force": T_force.copy(),
            "T_norm": T_norm.copy(),
            "T_command": self.T_command,
            "omega": np.zeros(4),  # Not used in CTBR
            "omega_norm": np.zeros(3),  # Not used in CTBR
        }

    def _initialize_state(self):
        rng = np.random.default_rng()
        rand_dir = rng.normal(size=3)
        rand_dir /= np.linalg.norm(rand_dir) + 1e-6
        pos = self.init_pos + self.init_radius * rand_dir

        vel = rng.uniform(-0.5, 0.5, size=3)
        roll_pitch_range = np.pi / 4
        phi = rng.uniform(-roll_pitch_range, roll_pitch_range)
        theta = rng.uniform(-roll_pitch_range, roll_pitch_range)
        psi = 0.0
        pqr = rng.uniform(-2, 2, size=3)

        T_norm = rng.uniform(-1, 1)

        self.state[:] = np.concatenate([pos, vel, [phi, theta, psi], pqr, [T_norm]])
        self.initial_state[:] = self.state.copy()
        self.acc_measured[:] = 0.0

    def reset(self):
        self.f_func = self._build_state_space_model()
        self._initialize_state()
