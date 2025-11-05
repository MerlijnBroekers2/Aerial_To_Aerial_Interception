import numpy as np
from sympy import *


class VecMotorPursuer:
    def __init__(self, config: dict, num_envs: int):
        self.num_envs = num_envs
        self.dt = config["DT"]
        self.config = config["PURSUER"]
        self.domain_rand_pct = self.config["domain_randomization_pct"]

        self._load_nominal_parameters()
        self._init_parameter_arrays()
        self._build_state_space()

        # state = [x y z vx vy vz φ θ ψ p q r w1 w2 w3 w4]     shape: (N,16)
        self.state = np.zeros((num_envs, 16), dtype=np.float64)
        self.initial_state = np.zeros_like(self.state)

        self._last_acc = np.zeros((num_envs, 3), dtype=np.float64)
        self._last_rates = np.zeros((num_envs, 3), dtype=np.float64)
        self.motor_cmd = np.zeros((num_envs, 4), dtype=np.float64)

        self.reset(np.ones(num_envs, dtype=bool))

    def _load_nominal_parameters(self):
        p = self.config
        # Nominal values
        self.g_nom = p["gravity"]
        self.k_x_nom = p["motor"]["k_x"]
        self.k_y_nom = p["motor"]["k_y"]
        self.k_w_nom = p["motor"]["k_w"]
        self.k_p_nom = np.array(
            [p["motor"][f"k_p{i+1}"] for i in range(4)], dtype=np.float32
        )
        self.k_q_nom = np.array(
            [p["motor"][f"k_q{i+1}"] for i in range(4)], dtype=np.float32
        )
        self.k_r_nom = np.array(
            [p["motor"][f"k_r{i+1}"] for i in range(8)], dtype=np.float32
        )
        self.tau_nom = p["motor"]["tau"]
        self.k_blend_nom = p["motor"]["curve_k"]
        self.w_min_nom = p["motor"]["w_min"]
        self.w_max_nom = p["motor"]["w_max"]
        self.init_radius_nom = p["INIT_RADIUS"]

        # Other
        self.pos_noise_std = p["POSITION_NOISE_STD"]
        self.vel_noise_std = p["VELOCITY_NOISE_STD"]
        self.init_pos = np.array(p["INITIAL_POS"])
        self.init_vel = np.array(p["INITIAL_VEL"])
        self.init_att = np.array(p["INITIAL_ATTITUDE"])
        self.init_rates = np.array(p["INITIAL_RATES"])
        self.init_omega = np.array(p["INITIAL_OMEGA"])

    def _init_parameter_arrays(self):
        N = self.num_envs
        self.g_arr = np.full(N, self.g_nom, dtype=np.float32)
        self.k_x_arr = np.full(N, self.k_x_nom, dtype=np.float32)
        self.k_y_arr = np.full(N, self.k_y_nom, dtype=np.float32)
        self.k_w_arr = np.full(N, self.k_w_nom, dtype=np.float32)
        self.k_p_arr = np.tile(self.k_p_nom, (N, 1))
        self.k_q_arr = np.tile(self.k_q_nom, (N, 1))
        self.k_r_arr = np.tile(self.k_r_nom, (N, 1))
        self.tau_arr = np.full(N, self.tau_nom, dtype=np.float32)
        self.k_blend_arr = np.full(N, self.k_blend_nom, dtype=np.float32)
        self.w_min_arr = np.full(N, self.w_min_nom, dtype=np.float32)
        self.w_max_arr = np.full(N, self.w_max_nom, dtype=np.float32)
        self.init_radius_arr = np.full(N, self.init_radius_nom, dtype=np.float32)

    def _get_randomized(self, nominal, key, n):
        pct = self.domain_rand_pct.get(key, 0.0)
        return np.random.uniform(1 - pct, 1 + pct, size=n) * nominal

    def _randomize_physical_parameters(self, mask):
        idx = np.nonzero(mask)[0]
        n = len(idx)
        if n == 0:
            return

        self.g_arr[idx] = self._get_randomized(self.g_nom, "g", n)
        self.k_x_arr[idx] = self._get_randomized(self.k_x_nom, "k_x", n)
        self.k_y_arr[idx] = self._get_randomized(self.k_y_nom, "k_y", n)
        self.k_w_arr[idx] = self._get_randomized(self.k_w_nom, "k_w", n)
        self.tau_arr[idx] = self._get_randomized(self.tau_nom, "tau", n)
        self.k_blend_arr[idx] = np.clip(
            self._get_randomized(self.k_blend_nom, "curve_k", n), 0.0, 1.0
        )
        self.w_min_arr[idx] = self._get_randomized(self.w_min_nom, "w_min", n)
        self.w_max_arr[idx] = self._get_randomized(self.w_max_nom, "w_max", n)
        self.init_radius_arr[idx] = self._get_randomized(
            self.init_radius_nom, "init_radius", n
        )

        for i in range(4):
            self.k_p_arr[idx, i] = self._get_randomized(self.k_p_nom[i], f"k_p{i+1}", n)
            self.k_q_arr[idx, i] = self._get_randomized(self.k_q_nom[i], f"k_q{i+1}", n)
        for i in range(8):
            self.k_r_arr[idx, i] = self._get_randomized(self.k_r_nom[i], f"k_r{i+1}", n)

    def _pack_params(self) -> np.ndarray:
        """
        Return an array with **exactly the order used in the SymPy tuple
        `params`**.  Shape: (num_envs, 23).  No allocations in the hot loop.
        """
        return np.column_stack(
            [
                self.g_arr,
                self.k_x_arr,
                self.k_y_arr,
                self.k_w_arr,
                self.k_p_arr[:, 0],
                self.k_p_arr[:, 1],
                self.k_p_arr[:, 2],
                self.k_p_arr[:, 3],
                self.k_q_arr[:, 0],
                self.k_q_arr[:, 1],
                self.k_q_arr[:, 2],
                self.k_q_arr[:, 3],
                self.k_r_arr[:, 0],
                self.k_r_arr[:, 1],
                self.k_r_arr[:, 2],
                self.k_r_arr[:, 3],
                self.k_r_arr[:, 4],
                self.k_r_arr[:, 5],
                self.k_r_arr[:, 6],
                self.k_r_arr[:, 7],
                self.tau_arr,
                self.k_blend_arr,  #  ==  “k”   (prop-speed blend factor)
                self.w_min_arr,
                self.w_max_arr,
            ]
        )

    def _build_state_space(self):
        state = symbols("x y z v_x v_y v_z phi theta psi p q r w1 w2 w3 w4")
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4 = state
        control = symbols("U_1 U_2 U_3 U_4")  # normalized motor commands between [-1,1]
        u1, u2, u3, u4 = control

        params = symbols(
            "g, k_x, k_y, k_w, k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4, k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, tau, k, w_min, w_max"
        )
        (
            g,
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
        ) = params

        # Rotation matrix
        Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
        Ry = Matrix(
            [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
        )
        Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
        R = Rz * Ry * Rx

        # Body velocity
        vbx, vby, vbz = R.T @ Matrix([vx, vy, vz])

        # normalized motor speeds to rad/s
        W1 = (w1 + 1) / 2 * (w_max - w_min) + w_min
        W2 = (w2 + 1) / 2 * (w_max - w_min) + w_min
        W3 = (w3 + 1) / 2 * (w_max - w_min) + w_min
        W4 = (w4 + 1) / 2 * (w_max - w_min) + w_min

        # motor commands scaled to [0,1]
        U1 = (u1 + 1) / 2
        U2 = (u2 + 1) / 2
        U3 = (u3 + 1) / 2
        U4 = (u4 + 1) / 2

        # first order delay:
        # the steadystate rpm motor response to the motor command U is described by:
        # Wc = (w_max-w_min)*sqrt(k U**2 + (1-k)*U) + w_min
        Wc1 = (w_max - w_min) * sqrt(k * U1**2 + (1 - k) * U1) + w_min
        Wc2 = (w_max - w_min) * sqrt(k * U2**2 + (1 - k) * U2) + w_min
        Wc3 = (w_max - w_min) * sqrt(k * U3**2 + (1 - k) * U3) + w_min
        Wc4 = (w_max - w_min) * sqrt(k * U4**2 + (1 - k) * U4) + w_min

        # rad/s
        d_W1 = (Wc1 - W1) / tau
        d_W2 = (Wc2 - W2) / tau
        d_W3 = (Wc3 - W3) / tau
        d_W4 = (Wc4 - W4) / tau

        # normalized motor speeds d/dt[W - w_min_n)/(w_max_n-w_min_n)*2 - 1]
        d_w1 = d_W1 / (w_max - w_min) * 2
        d_w2 = d_W2 / (w_max - w_min) * 2
        d_w3 = d_W3 / (w_max - w_min) * 2
        d_w4 = d_W4 / (w_max - w_min) * 2

        # Thrust and Drag
        T = k_w * (W1**2 + W2**2 + W3**2 + W4**2)
        Dx = k_x * vbx * (W1 + W2 + W3 + W4)
        Dy = k_y * vby * (W1 + W2 + W3 + W4)

        # Moments
        Mx = k_p1 * W1**2 + k_p2 * W2**2 + k_p3 * W3**2 + k_p4 * W4**2
        My = k_q1 * W1**2 + k_q2 * W2**2 + k_q3 * W3**2 + k_q4 * W4**2
        Mz = (
            k_r1 * W1
            + k_r2 * W2
            + k_r3 * W3
            + k_r4 * W4
            + k_r5 * d_W1
            + k_r6 * d_W2
            + k_r7 * d_W3
            + k_r8 * d_W4
        )

        # Dynamics
        d_x = vx
        d_y = vy
        d_z = vz

        d_vx, d_vy, d_vz = Matrix([0, 0, g]) + R @ Matrix([Dx, Dy, T])

        d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        d_theta = q * cos(phi) - r * sin(phi)
        d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

        d_p = Mx
        d_q = My
        d_r = Mz

        # State space model
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

        # lambdify
        self.f_func = lambdify(
            (Array(state), Array(control), Array(params)), Array(f), "numpy"
        )

    def _compute_derivatives(
        self, state: np.ndarray, motor_cmd: np.ndarray
    ) -> np.ndarray:
        """
        Compute \dot{x} for every environment via the lambdified SymPy function.
        Parameters
        ----------
        state : (N,16)  current state
        motor_cmd : (N,4)  throttle commands in [-1,1]
        Returns
        -------
        (N,16)  state derivatives
        """
        u = np.clip(motor_cmd, -1.0, 1.0)

        # Prepare inputs in column-major shape expected by `f_func`
        #   state.T   -> (16, N)
        #   u.T       -> ( 4, N)
        #   params.T  -> (23, N)
        param_mat = self._pack_params()
        derivs = self.f_func(state.T, u.T, param_mat.T).T  #  (N,16)

        return derivs

    def step_learn(self, control_vec):
        control_vec = np.clip(control_vec, -1.0, 1.0)
        self.motor_cmd = control_vec
        derivs = self._compute_derivatives(self.state, control_vec)
        self.state += derivs * self.dt
        self._last_acc = derivs[:, 3:6]
        self._last_rates = derivs[:, 9:12]
        return self._last_acc.copy()

    def reset(self, mask: np.ndarray):
        idx = np.nonzero(mask)[0]
        n = len(idx)
        if n == 0:
            return

        self._randomize_physical_parameters(mask)
        dirs = np.random.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        pos = self.init_pos + self.init_radius_arr[idx][:, None] * dirs
        vel = np.random.uniform(-0.5, 0.5, size=(n, 3))
        limit = np.pi / 4
        phi = np.random.uniform(-limit, limit, size=(n, 1))
        theta = np.random.uniform(-limit, limit, size=(n, 1))
        psi = np.random.uniform(-np.pi, np.pi, size=(n, 1))
        pqr = np.random.uniform(-2, 2, size=(n, 3))
        w_init = np.random.uniform(-1, 1, size=(n, 4))
        new_state = np.concatenate([pos, vel, phi, theta, psi, pqr, w_init], axis=1)
        self.state[idx] = new_state
        self.initial_state[idx] = new_state
        self._last_acc[idx] = 0
        self._last_rates[idx] = pqr

    def get_state(self):
        pos = self.state[:, 0:3]
        vel = self.state[:, 3:6]
        att = self.state[:, 6:9]
        rates = self.state[:, 9:12]
        omega_norm = self.state[:, 12:16]
        omega = (omega_norm + 1) / 2 * (
            self.w_max_arr[:, None] - self.w_min_arr[:, None]
        ) + self.w_min_arr[:, None]

        noisy_pos = pos + np.random.normal(0, self.pos_noise_std, pos.shape)
        noisy_vel = vel + np.random.normal(0, self.vel_noise_std, vel.shape)

        zeros3 = np.zeros((self.num_envs, 3))
        zeros1 = np.zeros((self.num_envs, 1))

        return {
            "true_position": pos.copy(),
            "noisy_position": noisy_pos,
            "velocity": vel.copy(),
            "noisy_velocity": noisy_vel,
            "acceleration": self._last_acc.copy(),
            "acc_command": zeros3,
            "acc_command_filtered": zeros3,
            "acc_measured": self._last_acc.copy(),
            "attitude": att.copy(),
            "attitude_commanded": zeros3,
            "rates": rates.copy(),
            "rates_command": zeros3,
            "T_force": np.zeros((self.num_envs, 1)),
            "T_norm": omega_norm.copy(),
            "T_command": zeros1,
            "omega": omega.copy(),
            "omega_norm": omega_norm.copy(),
        }
