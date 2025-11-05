import numpy as np
import copy
from sympy import symbols, Matrix, Array, cos, sin, tan, sqrt
from sympy import lambdify
from src.models.pursuers.base import IPursuer

EPSILON = 1e-6


class Motor_Pursuer(IPursuer):
    """
    Quadcopter pursuer model driven by normalized motor commands (u1..u4 in [-1,1]).
    Builds a state-space model from symbolic equations of motion.
    """

    def __init__(self, config: dict, control_law):
        self.dt = config["DT"]
        self._base_pconfig = config["PURSUER"]
        self.control_law = control_law

        self._load_nominal_parameters()

        # State vector: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4]
        self.state = np.zeros(16, dtype=np.float64)
        self.initial_state = np.zeros_like(self.state)

        # storage for last commands and measurements
        self.motor_cmd = np.zeros(4, dtype=np.float64)
        self.acc_measured = np.zeros(3, dtype=np.float64)
        self.rates = np.zeros(3, dtype=np.float64)

        self.reset()

    def _load_nominal_parameters(self):
        p = self._base_pconfig
        # Gravity
        self.g = p["gravity"]
        # Aerodynamic drag constants
        self.k_x = p["motor"]["k_x"]  # [units?]
        self.k_y = p["motor"]["k_y"]
        # Motor thrust coefficient
        self.k_w = p["motor"]["k_w"]  # [N/(rad/s)^2]
        # Moment coefficients around body axes
        self.k_p1 = p["motor"]["k_p1"]
        self.k_p2 = p["motor"]["k_p2"]
        self.k_p3 = p["motor"]["k_p3"]
        self.k_p4 = p["motor"]["k_p4"]
        self.k_q1 = p["motor"]["k_q1"]
        self.k_q2 = p["motor"]["k_q2"]
        self.k_q3 = p["motor"]["k_q3"]
        self.k_q4 = p["motor"]["k_q4"]
        self.k_r1 = p["motor"]["k_r1"]
        self.k_r2 = p["motor"]["k_r2"]
        self.k_r3 = p["motor"]["k_r3"]
        self.k_r4 = p["motor"]["k_r4"]
        self.k_r5 = p["motor"]["k_r5"]
        self.k_r6 = p["motor"]["k_r6"]
        self.k_r7 = p["motor"]["k_r7"]
        self.k_r8 = p["motor"]["k_r8"]
        # Motor delay and curve
        self.tau = p["motor"]["tau"]  # [s]
        self.k_blend = p["motor"]["curve_k"]  # dimensionless
        self.w_min = p["motor"]["w_min"]  # [rad/s]
        self.w_max = p["motor"]["w_max"]  # [rad/s]
        # pack parameters for symbolic model
        self.params = np.array(
            [
                self.k_x,
                self.k_y,
                self.k_w,
                self.k_p1,
                self.k_p2,
                self.k_p3,
                self.k_p4,
                self.k_q1,
                self.k_q2,
                self.k_q3,
                self.k_q4,
                self.k_r1,
                self.k_r2,
                self.k_r3,
                self.k_r4,
                self.k_r5,
                self.k_r6,
                self.k_r7,
                self.k_r8,
                self.tau,
                self.k_blend,
                self.w_min,
                self.w_max,
            ],
            dtype=np.float64,
        )
        # noise and init
        self.pos_noise_std = p["POSITION_NOISE_STD"]
        self.vel_noise_std = p["VELOCITY_NOISE_STD"]
        self.init_pos = np.array(p["INITIAL_POS"], dtype=np.float64)
        self.init_vel = np.array(p["INITIAL_VEL"], dtype=np.float64)
        self.init_att = np.array(p["INITIAL_ATTITUDE"], dtype=np.float64)
        self.init_rates = np.array(p["INITIAL_RATES"], dtype=np.float64)
        self.init_radius = p["INIT_RADIUS"]
        self.init_omega = np.array(p["INITIAL_OMEGA"], dtype=np.float64)

    def _build_state_space_model(self):
        # symbolic variables
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4 = symbols(
            "x y z v_x v_y v_z phi theta psi p q r w1 w2 w3 w4"
        )
        control = symbols("U_1 U_2 U_3 U_4")
        u1, u2, u3, u4 = control
        (
            k_x,
            k_y,
            k_w,
            k_p1,
            k_p2,
            k_p3,
            k_p4,
            k_q1,
            k_q2,
            k_q3,
            k_q4,
            k_r1,
            k_r2,
            k_r3,
            k_r4,
            k_r5,
            k_r6,
            k_r7,
            k_r8,
            tau,
            k,
            w_min,
            w_max,
        ) = symbols(
            "k_x k_y k_w k_p1 k_p2 k_p3 k_p4 k_q1 k_q2 k_q3 k_q4 k_r1 k_r2 k_r3 k_r4 k_r5 k_r6 k_r7 k_r8 tau k w_min w_max"
        )
        # rotation
        Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
        Ry = Matrix(
            [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
        )
        Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
        R = Rz * Ry * Rx
        # body vel
        vbx, vby, vbz = R.T @ Matrix([vx, vy, vz])
        # maps
        W1 = (w1 + 1) / 2 * (w_max - w_min) + w_min
        W2 = (w2 + 1) / 2 * (w_max - w_min) + w_min
        W3 = (w3 + 1) / 2 * (w_max - w_min) + w_min
        W4 = (w4 + 1) / 2 * (w_max - w_min) + w_min
        # commands scaled [0,1]
        U1 = (u1 + 1) / 2
        U2 = (u2 + 1) / 2
        U3 = (u3 + 1) / 2
        U4 = (u4 + 1) / 2
        # steady-state
        Wc1 = (w_max - w_min) * sqrt(k * U1**2 + (1 - k) * U1) + w_min
        Wc2 = (w_max - w_min) * sqrt(k * U2**2 + (1 - k) * U2) + w_min
        Wc3 = (w_max - w_min) * sqrt(k * U3**2 + (1 - k) * U3) + w_min
        Wc4 = (w_max - w_min) * sqrt(k * U4**2 + (1 - k) * U4) + w_min
        # rates
        d_W1 = (Wc1 - W1) / tau
        d_W2 = (Wc2 - W2) / tau
        d_W3 = (Wc3 - W3) / tau
        d_W4 = (Wc4 - W4) / tau
        # normalized
        d_w1 = d_W1 / (w_max - w_min) * 2
        d_w2 = d_W2 / (w_max - w_min) * 2
        d_w3 = d_W3 / (w_max - w_min) * 2
        d_w4 = d_W4 / (w_max - w_min) * 2
        # thrust and drag
        T = k_w * (W1**2 + W2**2 + W3**2 + W4**2)
        Dx = k_x * vbx * (W1 + W2 + W3 + W4)
        Dy = k_y * vby * (W1 + W2 + W3 + W4)
        # moments
        Mx = k_p1 * W1**2 + k_p2 * W2**2 + k_p3 * W3**2 + k_p4 * W4**2
        My = k_q1 * W1**2 + k_q2 * W2**2 + k_q3 * W3**2 + k_q4 * W4**2
        Mz = (
            k_r1 * W1
            + k_r2 * W2
            + k_r3 * W3
            + k_r4 * W4
            + k_r5 * d_W1  # in example all are d_W1
            + k_r6 * d_W2
            + k_r7 * d_W3
            + k_r8 * d_W4
        )

        # dynamics
        d_x = vx
        d_y = vy
        d_z = vz
        d_vx, d_vy, d_vz = Matrix([0, 0, self.g]) + R @ Matrix([Dx, Dy, T])
        d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        d_theta = q * cos(phi) - r * sin(phi)
        d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)
        d_p = Mx
        d_q = My
        d_r = Mz
        f = [
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
            d_w1,
            d_w2,
            d_w3,
            d_w4,
        ]
        state_syms = Array(
            [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4]
        )
        param_syms = Array(
            [
                k_x,
                k_y,
                k_w,
                k_p1,
                k_p2,
                k_p3,
                k_p4,
                k_q1,
                k_q2,
                k_q3,
                k_q4,
                k_r1,
                k_r2,
                k_r3,
                k_r4,
                k_r5,
                k_r6,
                k_r7,
                k_r8,
                tau,
                k,
                w_min,
                w_max,
            ]
        )
        return lambdify((state_syms, control, param_syms), Array(f), "numpy")

    def step(self, command):
        # Compute motor commands and apply state-space model
        self.motor_cmd = command
        deriv = np.array(
            self.f_model(self.state, self.motor_cmd, self.params)
        ).flatten()
        self.state += deriv * self.dt
        self.acc_measured = deriv[3:6]
        self.rates = deriv[9:12]
        return self.acc_measured.copy()

    def get_state(self):
        pos = self.state[0:3]
        vel = self.state[3:6]
        phi, theta, psi = self.state[6:9]
        p, q, r = self.state[9:12]
        omega_norm = self.state[12:16]
        omega = (omega_norm + 1) / 2 * (self.w_max - self.w_min) + self.w_min

        return {
            "true_position": pos.copy(),
            "noisy_position": pos + np.random.normal(0, self.pos_noise_std, 3),
            "velocity": vel.copy(),
            "noisy_velocity": vel + np.random.normal(0, self.vel_noise_std, 3),
            "acceleration": self.acc_measured.copy(),
            "acc_command": np.zeros(3),
            "acc_command_filtered": np.zeros(3),
            "acc_measured": self.acc_measured.copy(),
            "attitude": np.array([phi, theta, psi]),
            "attitude_commanded": np.zeros(3),
            "rates": np.array([p, q, r]),
            "rates_command": np.zeros(3),
            "T_force": np.zeros(1),
            "T_norm": np.zeros(1),
            "T_command": np.zeros(1),
            "omega": omega.copy(),
            "omega_norm": omega_norm.copy(),
            # Can later add thrust to state to also expose for plotting/analysis
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
        psi = rng.uniform(-np.pi, np.pi)
        pqr = rng.uniform(-2.0, 2.0, size=3)

        w_init = rng.uniform(-1, 1, size=4)

        self.state[:] = np.concatenate([pos, vel, [phi, theta, psi], pqr, w_init])

        self.initial_state[:] = self.state
        self.acc_measured[:] = 0.0
        self.rates = pqr

    def reset(self):
        self.f_model = self._build_state_space_model()
        self._initialize_state()
