import numpy as np
from sympy import Matrix, symbols, Array, cos, sin, lambdify
from src.models.pursuers.base import IPursuer


class Acc_TVC_INDI_Pursuer(IPursuer):
    def __init__(self, config: dict, control_law):
        self.dt = config["DT"]
        self.config = config["PURSUER"]
        self.control_law = control_law

        self.g = self.config["gravity"]
        self.face_evader = self.config["face_evader"]
        self.drag_params = self.config["drag"]

        self.tau_phi = self.config["actuator_time_constants"]["p"]
        self.tau_theta = self.config["actuator_time_constants"]["q"]
        self.tau_T = self.config["actuator_time_constants"]["T"]
        self.max_bank_angle = np.radians(self.config["actuator_limits"]["bank_angle"])

        self.pos_noise_std = self.config["POSITION_NOISE_STD"]
        self.vel_noise_std = self.config["VELOCITY_NOISE_STD"]

        # Store init parameters for reuse
        self.init_pos = np.array(self.config["INITIAL_POS"], dtype=np.float64)
        self.init_vel = np.array(self.config["INITIAL_VEL"], dtype=np.float64)
        self.init_att = np.array(self.config["INITIAL_ATTITUDE"], dtype=np.float64)
        self.init_rates = np.array(self.config["INITIAL_RATES"], dtype=np.float64)
        self.init_radius = self.config["INIT_RADIUS"]

        self.state = np.zeros(10, dtype=np.float64)
        self.initial_state = np.zeros_like(self.state)
        self._last_acc = np.zeros(3, dtype=np.float64)

        self.f_func = self._build_state_space_model()
        self.reset()

    def _build_state_space_model(self):
        x, y, z, vx, vy, vz, phi, theta, psi, thrust = symbols(
            "x y z v_x v_y v_z phi theta psi T"
        )

        state = [x, y, z, vx, vy, vz, phi, theta, psi, thrust]

        # Control inputs (commands)
        phi_cmd, theta_cmd, thrust_cmd = symbols("phi_cmd theta_cmd thrust_cmd")
        controls = [phi_cmd, theta_cmd, thrust_cmd]

        # Drag coefficients
        kx, ky = self.drag_params["kx_acc_ctbr"], self.drag_params["ky_acc_ctbr"]

        Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
        Ry = Matrix(
            [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
        )
        Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
        R = Rz * Ry * Rx

        # Velocity in body frame
        vb = R.T * Matrix([vx, vy, vz])
        Dx = -kx * vb[0]
        Dy = -ky * vb[1]

        # Translational dynamics
        d_pos = Matrix([vx, vy, vz])
        dv = Matrix([0, 0, -self.g]) + R * Matrix([Dx, Dy, thrust])
        d_vx, d_vy, d_vz = dv

        # First-order pitch, roll, thrust dynamics
        d_phi = (phi_cmd - phi) / self.tau_phi
        d_theta = (theta_cmd - theta) / self.tau_theta
        d_T = (thrust_cmd - thrust) / self.tau_T

        # Constant yaw for simplicity
        d_psi = 0

        # State derivative
        dx = Matrix(
            [d_pos[0], d_pos[1], d_pos[2], d_vx, d_vy, d_vz, d_phi, d_theta, d_psi, d_T]
        )

        f_func = lambdify((Array(state), Array(controls)), Array(dx), "numpy")
        return f_func

    def step(self, guidance_state=None):
        """Step using acceleration-based guidance."""
        acceleration_cmd = self.control_law.compute_acceleration(guidance_state)

        F_des = np.array(acceleration_cmd) + np.array([0, 0, self.g])
        psi_des = self._compute_desired_yaw(guidance_state)
        phi_cmd, theta_cmd, thrust_cmd = self._decompose_thrust(F_des, psi_des)
        control_vec = np.array([phi_cmd, theta_cmd, thrust_cmd])
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
            "rates": np.zeros(3),
            "T": self.state[9].copy(),
        }

    def reset(self):
        rand_dir = np.random.normal(size=3)
        rand_dir /= np.linalg.norm(rand_dir)
        pos = self.init_pos + self.init_radius * rand_dir

        state = np.concatenate([pos, self.init_vel, self.init_att, [0.0]])
        self.state[:] = state
        self.initial_state[:] = state
        self._last_acc[:] = 0.0

    def _compute_desired_yaw(self, guidance_state):
        if self.face_evader and guidance_state and "r" in guidance_state:
            r = guidance_state["r"]
            if np.linalg.norm(r) > 1e-5:
                return np.arctan2(r[1], r[0])
        return 0.0

    def _decompose_thrust(self, F, psi):
        T = np.linalg.norm(F)
        if T < 1e-6:
            return 0.0, 0.0, 0.0
        phi = np.arcsin((F[0] * np.sin(psi) - F[1] * np.cos(psi)) / T)
        theta = np.arctan2(F[0] * np.cos(psi) + F[1] * np.sin(psi), F[2])

        # # Clamp angles to ±20 degrees (in radians)
        # max_angle_rad = np.deg2rad(self.max_bank_angle)
        # phi = np.clip(phi, -max_angle_rad, max_angle_rad)
        # theta = np.clip(theta, -max_angle_rad, max_angle_rad)

        return phi, theta, T
