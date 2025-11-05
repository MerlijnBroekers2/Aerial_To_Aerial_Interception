import numpy as np
from sympy import Matrix, symbols, Array, cos, sin, tan, lambdify
from src.models.pursuers.base import IPursuer
import copy

from src.utils.helpers import ButterworthFilter

EPSILON = 1e-6


# class Acc_INDI_INDI_Pursuer(IPursuer):
#     def __init__(self, config: dict, control_law):
#         self.dt = config["DT"]
#         self._base_pconfig = copy.deepcopy(config["PURSUER"])
#         self.control_law = control_law

#         self._load_nominal_parameters()

#         self.state = np.zeros(10, dtype=np.float64)
#         self.initial_state = np.zeros_like(self.state)
#         self.acc_cmd_raw = np.zeros(3, dtype=np.float64)
#         self.acc_cmd_filtered = np.zeros(3, dtype=np.float64)
#         self.acc_measured = np.zeros(3, dtype=np.float64)
#         self.rates = np.zeros(3, dtype=np.float64)

#         self._last_phi_cmd = 0
#         self._last_theta_cmd = 0
#         self._last_thrust_cmd = 0

#         self.reset()

#     def _load_nominal_parameters(self):
#         p = self._base_pconfig

#         self.g = p["gravity"]
#         self.drag_params = {
#             "kx_acc_ctbr": p["drag"]["kx_acc_ctbr"],
#             "ky_acc_ctbr": p["drag"]["ky_acc_ctbr"],
#         }

#         self.tau_phi = p["actuator_time_constants"]["phi"]
#         self.tau_theta = p["actuator_time_constants"]["theta"]
#         self.tau_T = p["actuator_time_constants"]["T"]

#         self.bank_angle_deg = p["actuator_limits"]["bank_angle"]
#         self.max_bank_angle = np.radians(self.bank_angle_deg)

#         self.max_accel = p["MAX_ACCELERATION"]

#         self.T_min, self.T_max = p["actuator_limits"]["T"]

#         self.pos_noise_std = p["POSITION_NOISE_STD"]
#         self.vel_noise_std = p["VELOCITY_NOISE_STD"]

#         self.acc_filter = ButterworthFilter(
#             num_envs=1,
#             dim=3,
#             dt=self.dt,
#             cutoff_hz=p["BUTTER_ACC_FILTER_CUTOFF_HZ"],
#             order=2,
#         )

#         self.init_pos = np.array(p["INITIAL_POS"], dtype=np.float64)
#         self.init_vel = np.array(p["INITIAL_VEL"], dtype=np.float64)
#         self.init_att = np.array(p["INITIAL_ATTITUDE"], dtype=np.float64)
#         self.init_rates = np.array(p["INITIAL_RATES"], dtype=np.float64)
#         self.init_radius = p["INIT_RADIUS"]

#         self.delta_a_min = np.array(p["delta_a_limits"]["min"], dtype=np.float64)
#         self.delta_a_max = np.array(p["delta_a_limits"]["max"], dtype=np.float64)

#     def _build_state_space_model(self):
#         x, y, z, vx, vy, vz, phi, theta, psi, thrust = symbols(
#             "x y z v_x v_y v_z phi theta psi T"
#         )
#         state_syms = [x, y, z, vx, vy, vz, phi, theta, psi, thrust]

#         phi_cmd, theta_cmd, thrust_cmd = symbols("phi_cmd theta_cmd thrust_cmd")
#         control_syms = [phi_cmd, theta_cmd, thrust_cmd]

#         kx = self.drag_params["kx_acc_ctbr"]
#         ky = self.drag_params["ky_acc_ctbr"]

#         Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
#         Ry = Matrix(
#             [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
#         )
#         Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
#         R = Rz * Ry * Rx

#         vb = R.T * Matrix([vx, vy, vz])
#         Dx = -kx * vb[0]
#         Dy = -ky * vb[1]

#         d_pos = Matrix([vx, vy, vz])
#         dv = Matrix([0, 0, self.g]) + R * Matrix([Dx, Dy, -thrust])
#         d_vx, d_vy, d_vz = dv

#         d_phi = (phi_cmd - phi) / self.tau_phi
#         d_theta = (theta_cmd - theta) / self.tau_theta
#         d_T = (thrust_cmd - thrust) / self.tau_T

#         d_psi = 0

#         dx = Matrix(
#             [d_pos[0], d_pos[1], d_pos[2], d_vx, d_vy, d_vz, d_phi, d_theta, d_psi, d_T]
#         )

#         return lambdify((Array(state_syms), Array(control_syms)), Array(dx), "numpy")

#     def step(self, command):
#         cmd = np.asarray(command, dtype=np.float64)
#         cmd = np.clip(cmd, -1.0, 1.0)
#         norm = np.linalg.norm(cmd)
#         if norm > 1.0:
#             cmd = cmd / (norm + EPSILON)

#         acceleration_cmd = cmd * self.max_accel
#         self.acc_cmd_raw = acceleration_cmd.copy()

#         filtered_cmd = self.acc_filter.apply(acceleration_cmd[None, :])[0]
#         filt_norm = np.linalg.norm(filtered_cmd)
#         if filt_norm > self.max_accel:
#             filtered_cmd *= self.max_accel / (filt_norm + EPSILON)
#         self.acc_cmd_filtered = filtered_cmd.copy()

#         control_vec = self._indi_control_allocation(filtered_cmd)
#         return self._apply_control(control_vec)

#     def _apply_control(self, control_vec):
#         inner_loop_steps = 1
#         inner_dt = self.dt / inner_loop_steps

#         for _ in range(inner_loop_steps):
#             deriv = np.array(self.f_func(self.state, control_vec)).flatten()
#             self.state += deriv * inner_dt

#         self.acc_measured = deriv[3:6]
#         self.rates = deriv[6:9]
#         return self.acc_measured.copy()

#     def get_state(self):
#         pos = self.state[0:3]
#         vel = self.state[3:6]
#         phi, theta, psi = self.state[6:9]
#         T = self.state[9]
#         T_norm = (T - self.T_min) / (self.T_max - self.T_min) * 2 - 1

#         return {
#             "true_position": pos.copy(),
#             "noisy_position": pos + np.random.normal(0, self.pos_noise_std, 3),
#             "velocity": vel.copy(),
#             "noisy_velocity": vel + np.random.normal(0, self.vel_noise_std, 3),
#             "acceleration": self.acc_measured.copy(),
#             "acc_command": self.acc_cmd_raw.copy(),
#             "acc_command_filtered": self.acc_cmd_filtered.copy(),
#             "acc_measured": self.acc_measured.copy(),
#             "attitude": np.array([phi, theta, psi]),
#             "attitude_commanded": np.array(
#                 [self._last_phi_cmd, self._last_theta_cmd, 0.0]
#             ),
#             "rates": self.rates.copy(),
#             "rates_command": np.zeros_like(self.rates),
#             "T_norm": T_norm.copy(),
#             "T_force": T.copy(),
#             "T_command": self._last_thrust_cmd,
#             "omega": np.zeros(4),
#             "omega_norm": np.zeros(4),
#         }

#     def reset(self):
#         self.f_func = self._build_state_space_model()
#         self._initialize_state()

#     def _initialize_state(self):
#         rng = np.random.default_rng()
#         rand_dir = rng.normal(size=3)
#         rand_dir /= np.linalg.norm(rand_dir) + EPSILON
#         pos = self.init_pos + self.init_radius * rand_dir

#         vel = rng.uniform(-0.5, 0.5, size=3)

#         roll_pitch_limit = np.pi / 9
#         phi = rng.uniform(-roll_pitch_limit, roll_pitch_limit)
#         theta = rng.uniform(-roll_pitch_limit, roll_pitch_limit)
#         psi = 0.0

#         thrust_init = rng.uniform(0.0, self.T_max)

#         new_state = np.concatenate([pos, vel, [phi, theta, psi], [thrust_init]])
#         self.state[:] = new_state
#         self.initial_state[:] = new_state
#         self.acc_measured[:] = 0.0

#     def _indi_control_allocation(self, acc_des):
#         phi, theta, psi = self.state[6:9].copy()
#         accel_meas = self.acc_measured

#         Ga = self._compute_Ga_matrix(phi, theta)

#         try:
#             Ga_inv = np.linalg.inv(Ga)
#         except np.linalg.LinAlgError:
#             return phi, theta, self.state[9]

#         delta_a = acc_des - accel_meas
#         delta_a = np.clip(delta_a, self.delta_a_min, self.delta_a_max)

#         control_increment = Ga_inv @ delta_a

#         phi_cmd = phi + control_increment[1]
#         theta_cmd = theta + control_increment[0]
#         thrust_cmd = self.state[9] + control_increment[2] * (
#             -500
#         )  # NOTE! THIS IS THE SPECIFIC FORCE GAIN USED IN PPRZ!

#         phi_cmd = np.clip(phi_cmd, -self.max_bank_angle, self.max_bank_angle)
#         theta_cmd = np.clip(theta_cmd, -self.max_bank_angle, self.max_bank_angle)
#         thrust_cmd = np.clip(thrust_cmd, self.T_min, self.T_max)

#         self._last_phi_cmd = phi_cmd
#         self._last_theta_cmd = theta_cmd
#         self._last_thrust_cmd = thrust_cmd

#         return phi_cmd, theta_cmd, thrust_cmd

#     def _compute_Ga_matrix(self, phi, theta):
#         sphi = np.sin(phi)
#         cphi = np.cos(phi)
#         stheta = np.sin(theta)
#         ctheta = np.cos(theta)

#         T_est = -self.g

#         Ga = np.zeros((3, 3))
#         Ga[0, 0] = ctheta * cphi * T_est
#         Ga[0, 1] = -stheta * sphi * T_est
#         Ga[0, 2] = stheta * cphi

#         Ga[1, 0] = 0
#         Ga[1, 1] = -cphi * T_est
#         Ga[1, 2] = -sphi

#         Ga[2, 0] = -stheta * cphi * T_est
#         Ga[2, 1] = -ctheta * sphi * T_est
#         Ga[2, 2] = ctheta * cphi

#         return Ga


class Acc_INDI_INDI_Pursuer(IPursuer):
    """
    Acceleration INDI model with PD attitude loop -> rate commands -> 1st-order rate tracking.
    State: [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r, T]
    """

    def __init__(self, config: dict, control_law):
        self.dt = config["DT"]
        self._base_pconfig = copy.deepcopy(config["PURSUER"])
        self.control_law = control_law

        self._load_nominal_parameters()

        self.state = np.zeros(13, dtype=np.float64)
        self.initial_state = np.zeros_like(self.state)

        self.acc_cmd_raw = np.zeros(3, dtype=np.float64)
        self.acc_cmd_filtered = np.zeros(3, dtype=np.float64)
        self.acc_measured = np.zeros(3, dtype=np.float64)

        # last commanded values (for logging)
        self._last_phi_cmd = 0.0
        self._last_theta_cmd = 0.0
        self._last_T_cmd = 0.0
        self.rates_commanded = np.zeros(3, dtype=np.float64)

        self.reset()

    def _load_nominal_parameters(self):
        p = self._base_pconfig

        self.g = p["gravity"]
        self.drag_params = {
            "kx_acc_ctbr": p["drag"]["kx_acc_ctbr"],
            "ky_acc_ctbr": p["drag"]["ky_acc_ctbr"],
        }

        ac = p["actuator_time_constants"]
        self.tau_p = ac["p"]
        self.tau_q = ac["q"]
        self.tau_r = ac["r"]
        self.tau_T = ac["T"]

        apd = p["attitude_pd"]
        self.kp_phi = apd["kp"]["phi"]
        self.kp_theta = apd["kp"]["theta"]
        self.kp_psi = apd["kp"]["psi"]
        self.kd_phi = apd["kd"]["phi"]
        self.kd_theta = apd["kd"]["theta"]
        self.kd_psi = apd["kd"]["psi"]

        al = p["actuator_limits"]
        self.bank_angle_deg = al["bank_angle"]
        self.max_bank_angle = np.radians(self.bank_angle_deg)
        self.p_min, self.p_max = al["p"]
        self.q_min, self.q_max = al["q"]
        self.r_min, self.r_max = al["r"]
        self.T_min, self.T_max = al["T"]

        self.max_accel = p["MAX_ACCELERATION"]
        self.acc_filter = ButterworthFilter(
            num_envs=1,
            dim=3,
            dt=self.dt,
            cutoff_hz=p["BUTTER_ACC_FILTER_CUTOFF_HZ"],
            order=2,
        )

        self.init_pos = np.array(p["INITIAL_POS"], dtype=np.float64)
        self.init_vel = np.array(p["INITIAL_VEL"], dtype=np.float64)
        self.init_att = np.array(p["INITIAL_ATTITUDE"], dtype=np.float64)
        self.init_rates = np.array(p["INITIAL_RATES"], dtype=np.float64)
        self.init_radius = p["INIT_RADIUS"]

        self.pos_noise_std = p["POSITION_NOISE_STD"]
        self.vel_noise_std = p["VELOCITY_NOISE_STD"]

        self.delta_a_min = np.array(p["delta_a_limits"]["min"], dtype=np.float64)
        self.delta_a_max = np.array(p["delta_a_limits"]["max"], dtype=np.float64)

        # specific-force gain used for thrust increment (TAKEN TO BE SAME AS IN PPRZ)
        self.k_T_spec = -1.0

    def _build_state_space_model(self):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, T = symbols(
            "x y z v_x v_y v_z phi theta psi p q r T"
        )
        state_syms = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, T]

        p_cmd, q_cmd, r_cmd, T_cmd = symbols("p_cmd q_cmd r_cmd T_cmd")
        control_syms = [p_cmd, q_cmd, r_cmd, T_cmd]

        Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
        Ry = Matrix(
            [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
        )
        Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
        R = Rz * Ry * Rx

        kx = self.drag_params["kx_acc_ctbr"]
        ky = self.drag_params["ky_acc_ctbr"]
        vb = R.T * Matrix([vx, vy, vz])
        Dx = -kx * vb[0]
        Dy = -ky * vb[1]

        d_pos = Matrix([vx, vy, vz])
        dv = Matrix([0, 0, self.g]) + R * Matrix([Dx, Dy, -T])
        d_vx, d_vy, d_vz = dv

        d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        d_theta = q * cos(phi) - r * sin(phi)
        d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

        d_p = (p_cmd - p) / self.tau_p
        d_q = (q_cmd - q) / self.tau_q
        d_r = (r_cmd - r) / self.tau_r

        d_T = (T_cmd - T) / self.tau_T

        dx = Matrix(
            [
                d_pos[0],
                d_pos[1],
                d_pos[2],
                d_vx,
                d_vy,
                d_vz,
                d_phi,
                d_theta,
                d_psi,
                d_p,
                d_q,
                d_r,
                d_T,
            ]
        )

        return lambdify((Array(state_syms), Array(control_syms)), Array(dx), "numpy")

    def _compute_Ga_matrix(self, phi, theta):
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        stheta = np.sin(theta)
        ctheta = np.cos(theta)

        T_est = -self.g

        Ga = np.zeros((3, 3))
        Ga[0, 0] = ctheta * cphi * T_est
        Ga[0, 1] = -stheta * sphi * T_est
        Ga[0, 2] = stheta * cphi

        Ga[1, 0] = 0.0
        Ga[1, 1] = -cphi * T_est
        Ga[1, 2] = -sphi

        Ga[2, 0] = -stheta * cphi * T_est
        Ga[2, 1] = -ctheta * sphi * T_est
        Ga[2, 2] = ctheta * cphi
        return Ga

    def _indi_control_allocation(self, acc_des):
        phi, theta, psi = self.state[6:9].copy()
        accel_meas = self.acc_measured

        Ga = self._compute_Ga_matrix(phi, theta)
        try:
            Ga_inv = np.linalg.inv(Ga)
        except np.linalg.LinAlgError:
            return phi, theta, self.state[12]

        delta_a = acc_des - accel_meas
        delta_a = np.clip(delta_a, self.delta_a_min, self.delta_a_max)

        control_increment = Ga_inv @ delta_a
        phi_cmd = np.clip(
            phi + control_increment[1], -self.max_bank_angle, self.max_bank_angle
        )
        theta_cmd = np.clip(
            theta + control_increment[0], -self.max_bank_angle, self.max_bank_angle
        )
        T_cmd = np.clip(
            self.state[12] + self.k_T_spec * control_increment[2],
            self.T_min,
            self.T_max,
        )

        self._last_phi_cmd = phi_cmd
        self._last_theta_cmd = theta_cmd
        self._last_T_cmd = T_cmd
        return phi_cmd, theta_cmd, T_cmd

    def _attitude_pd_to_rate_cmds(self, phi_cmd, theta_cmd, psi_cmd=0.0):
        phi, theta, psi = self.state[6:9]
        p, q, r = self.state[9:12]

        p_cmd = self.kp_phi * (phi_cmd - phi) - self.kd_phi * p
        q_cmd = self.kp_theta * (theta_cmd - theta) - self.kd_theta * q
        r_cmd = self.kp_psi * (psi_cmd - psi) - self.kd_psi * r

        p_cmd = np.clip(p_cmd, self.p_min, self.p_max)
        q_cmd = np.clip(q_cmd, self.q_min, self.q_max)
        r_cmd = np.clip(r_cmd, self.r_min, self.r_max)

        self.rates_commanded[:] = [p_cmd, q_cmd, r_cmd]
        return p_cmd, q_cmd, r_cmd

    def step(self, command):
        # normalize accel command to respect acutator limits
        cmd = np.asarray(command, dtype=np.float64)
        cmd = np.clip(cmd, -1.0, 1.0)
        nrm = np.linalg.norm(cmd)
        if nrm > 1.0:
            cmd = cmd / (nrm + EPSILON)

        a_cmd = cmd * self.max_accel
        self.acc_cmd_raw = a_cmd.copy()

        a_filt = self.acc_filter.apply(a_cmd[None, :])[0]
        nrmf = np.linalg.norm(a_filt)
        if nrmf > self.max_accel:
            a_filt *= self.max_accel / (nrmf + EPSILON)
        self.acc_cmd_filtered = a_filt.copy()

        phi_sp, theta_sp, T_sp = self._indi_control_allocation(a_filt)

        p_cmd, q_cmd, r_cmd = self._attitude_pd_to_rate_cmds(
            phi_sp, theta_sp, psi_cmd=0.0
        )

        deriv = np.array(self.f_func(self.state, [p_cmd, q_cmd, r_cmd, T_sp])).flatten()
        self.state += deriv * self.dt

        self.acc_measured = deriv[3:6]
        return self.acc_measured.copy()

    def get_state(self):
        pos = self.state[0:3]
        vel = self.state[3:6]
        phi, theta, psi = self.state[6:9]
        p, q, r = self.state[9:12]
        T = self.state[12]

        return {
            "true_position": pos.copy(),
            "noisy_position": pos + np.random.normal(0, self.pos_noise_std, 3),
            "velocity": vel.copy(),
            "noisy_velocity": vel + np.random.normal(0, self.vel_noise_std, 3),
            "acceleration": self.acc_measured.copy(),
            "acc_command": self.acc_cmd_raw.copy(),
            "acc_command_filtered": self.acc_cmd_filtered.copy(),
            "acc_measured": self.acc_measured.copy(),
            "attitude": np.array([phi, theta, psi]),
            "attitude_commanded": np.array(
                [self._last_phi_cmd, self._last_theta_cmd, 0.0]
            ),
            "rates": np.array([p, q, r]),
            "rates_command": self.rates_commanded.copy(),
            "T_force": T.copy(),
            "T_norm": (T - self.T_min) / (self.T_max - self.T_min) * 2 - 1,
            "T_command": self._last_T_cmd,
            "omega": np.zeros(4),
            "omega_norm": np.zeros(4),
        }

    def _initialize_state(self):
        rng = np.random.default_rng()
        rand_dir = rng.normal(size=3)
        rand_dir /= np.linalg.norm(rand_dir) + EPSILON
        pos = self.init_pos + self.init_radius * rand_dir
        vel = rng.uniform(-0.5, 0.5, size=3)

        roll_pitch_limit = np.pi / 4
        phi = rng.uniform(-roll_pitch_limit, roll_pitch_limit)
        theta = rng.uniform(-roll_pitch_limit, roll_pitch_limit)
        psi = 0.0

        pqr = rng.uniform(-2.0, 2.0, size=3)

        T = rng.uniform(self.T_min, self.T_max)  # hover-ish

        self.state[:] = np.concatenate([pos, vel, [phi, theta, psi], pqr, [T]])
        self.initial_state[:] = self.state.copy()
        self.acc_measured[:] = 0.0

    def reset(self):
        self.f_func = self._build_state_space_model()
        self._initialize_state()
