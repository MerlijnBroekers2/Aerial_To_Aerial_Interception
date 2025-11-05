import numpy as np
from sympy import Matrix, symbols, Array, cos, sin, tan, lambdify
from src.models.pursuers.base import IPursuer


class Acc_TVC_CTBR_INDI_Pursuer(IPursuer):
    """
    Unified 13-state quadrotor model supporting both:
    - step(): using acceleration command and INDI conversion to CTBR.
    - step_learn(): using CTBR commands directly (for RL).
    """

    def __init__(self, config: dict, control_law):
        self.dt = config["DT"]
        self.config = config["PURSUER"]
        self.control_law = control_law

        self.g = self.config["gravity"]
        self.gains = self.config["attitude_pd_gains"]
        self.drag_params = self.config["drag"]

        ac = self.config["actuator_time_constants"]
        self.taup, self.tauq, self.taur, self.tauT = ac["p"], ac["q"], ac["r"], ac["T"]

        al = self.config["actuator_limits"]
        self.p_min, self.p_max = al["p"]
        self.q_min, self.q_max = al["q"]
        self.r_min, self.r_max = al["r"]
        self.T_min, self.T_max = al["T"]

        self.pos_noise_std = self.config["POSITION_NOISE_STD"]
        self.vel_noise_std = self.config["VELOCITY_NOISE_STD"]

        # Store init parameters for reuse
        self.init_pos = np.array(self.config["INITIAL_POS"], dtype=np.float64)
        self.init_vel = np.array(self.config["INITIAL_VEL"], dtype=np.float64)
        self.init_att = np.array(self.config["INITIAL_ATTITUDE"], dtype=np.float64)
        self.init_rates = np.array(self.config["INITIAL_RATES"], dtype=np.float64)
        self.init_radius = self.config["INIT_RADIUS"]

        self.state = np.zeros(13, dtype=np.float64)
        self.initial_state = np.zeros_like(self.state)
        self._last_acc = np.zeros(3, dtype=np.float64)

        self.f_func = self._build_state_space_model()
        self.reset()

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
        dv = Matrix([0, 0, -self.g]) + R * Matrix([Dx, Dy, T])
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

    def step(self, guidance_state=None):
        """Step using acceleration-based guidance."""
        acceleration_cmd = self.control_law.compute_acceleration(guidance_state)

        F_des = np.array(acceleration_cmd) + np.array([0, 0, self.g])
        psi_des = self._compute_desired_yaw(guidance_state)

        control_vec = self._compute_control_vector(F_des, psi_des)
        return self._apply_control(control_vec)

    def step_learn(self, control_vec):
        """Step using direct CTBR command (normalized), for RL training."""
        return self._apply_control(control_vec)

    def _apply_control(self, control_vec):
        deriv = np.array(self.f_func(self.state, control_vec)).flatten()
        self.state += deriv * self.dt
        self._last_acc = deriv[3:6]
        return self._last_acc.copy()

    def get_state(self):
        pos = self.state[0:3]
        vel = self.state[3:6]
        return {
            "true_position": pos.copy(),
            "noisy_position": pos + np.random.normal(0, self.pos_noise_std, 3),
            "velocity": vel.copy(),
            "noisy_velocity": vel + np.random.normal(0, self.vel_noise_std, 3),
            "acceleration": self._last_acc.copy(),
            "attitude": self.state[6:9].copy(),
            "rates": self.state[9:12].copy(),
            "T": self.state[12].copy(),
        }

    def reset(self):
        rand_dir = np.random.normal(size=3)
        rand_dir /= np.linalg.norm(rand_dir)
        pos = self.init_pos + self.init_radius * rand_dir

        state = np.concatenate(
            [pos, self.init_vel, self.init_att, self.init_rates, [0.0]]
        )
        self.state[:] = state
        self.initial_state[:] = state
        self._last_acc[:] = 0.0

    def _compute_desired_yaw(self, guidance_state):
        if self.face_evader and guidance_state and "r" in guidance_state:
            r = guidance_state["r"]
            if np.linalg.norm(r) > 1e-5:
                return np.arctan2(r[1], r[0])
        return 0.0

    def _compute_control_vector(self, F_des, psi_des):
        phi, theta, T = self._decompose_thrust(F_des, psi_des)
        T = np.clip(T, self.T_min, self.T_max)  # Ensure it's within physical limits
        desired_angles = np.array([phi, theta, psi_des])
        current_angles = self.state[6:9]
        current_rates = self.state[9:12]

        cmd = self._compute_ctbr_command(
            desired_angles, current_angles, current_rates, T
        )

        T_clipped = np.clip(cmd[0], self.T_min, self.T_max)
        p_clipped = np.clip(cmd[1], self.p_min, self.p_max)
        q_clipped = np.clip(cmd[2], self.q_min, self.q_max)
        r_clipped = np.clip(cmd[3], self.r_min, self.r_max)

        return np.array(
            [
                self._normalize(p_clipped, self.p_min, self.p_max),
                self._normalize(q_clipped, self.q_min, self.q_max),
                self._normalize(r_clipped, self.r_min, self.r_max),
                self._normalize(T_clipped, self.T_min, self.T_max),
            ]
        )

    def _decompose_thrust(self, F, psi):
        T = np.linalg.norm(F)
        if T < 1e-6:
            return 0.0, 0.0, 0.0
        phi = np.arcsin((F[0] * np.sin(psi) - F[1] * np.cos(psi)) / T)
        theta = np.arctan2(F[0] * np.cos(psi) + F[1] * np.sin(psi), F[2])
        return phi, theta, T

    def _compute_ctbr_command(self, des_angles, cur_angles, cur_rates, T):
        angle_error = np.clip(des_angles - cur_angles, -np.pi / 4, np.pi / 4)
        rates_cmd = self.gains["kp"] * angle_error - self.gains["kd"] * cur_rates
        return np.concatenate(([T], rates_cmd))

    def _normalize(self, val, min_val, max_val):
        return 2 * (val - min_val) / (max_val - min_val) - 1
