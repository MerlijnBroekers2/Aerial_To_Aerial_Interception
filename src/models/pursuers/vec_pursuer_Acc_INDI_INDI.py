import numpy as np

from src.utils.helpers import ButterworthFilter


# class VecAcc_INDI_INDI_Pursuer:
#     def __init__(self, config: dict, num_envs: int):
#         self.num_envs = num_envs
#         self.dt = config["DT"]
#         self.config = config["PURSUER"]
#         self.domain_rand_pct = self.config.get("domain_randomization_pct", {})

#         self._load_nominal_config()
#         self._init_parameter_arrays()

#         self.pos_noise_std = self.config["POSITION_NOISE_STD"]
#         self.vel_noise_std = self.config["VELOCITY_NOISE_STD"]

#         self.init_pos = np.array(self.config["INITIAL_POS"], dtype=np.float64)
#         self.init_vel = np.array(self.config["INITIAL_VEL"], dtype=np.float64)
#         self.init_att = np.array(self.config["INITIAL_ATTITUDE"], dtype=np.float64)
#         cutoff_hz = self.config["BUTTER_ACC_FILTER_CUTOFF_HZ"]
#         self.acc_filter = ButterworthFilter(
#             num_envs=num_envs, dim=3, dt=self.dt, cutoff_hz=cutoff_hz, order=2
#         )

#         self.state = np.zeros((num_envs, 10), dtype=np.float64)
#         self.acc_measured = np.zeros((num_envs, 3), dtype=np.float64)
#         self.rates = np.zeros((num_envs, 3), dtype=np.float64)

#         self.reset(np.ones(num_envs, dtype=bool))

#     def _load_nominal_config(self):
#         c = self.config
#         self.g_nom = c["gravity"]
#         self.kx_nom = c["drag"]["kx_acc_ctbr"]
#         self.ky_nom = c["drag"]["ky_acc_ctbr"]
#         self.tau_phi_nom = c["actuator_time_constants"]["phi"]
#         self.tau_theta_nom = c["actuator_time_constants"]["theta"]
#         self.tau_T_nom = c["actuator_time_constants"]["T"]
#         self.bank_angle_nom = np.radians(c["actuator_limits"]["bank_angle"])
#         self.max_accel_nom = c["MAX_ACCELERATION"]
#         self.T_range_nom = c["actuator_limits"]["T"]
#         self.delta_a_min_nom = np.array(c["delta_a_limits"]["min"], dtype=np.float64)
#         self.delta_a_max_nom = np.array(c["delta_a_limits"]["max"], dtype=np.float64)
#         self.init_radius_nom = c["INIT_RADIUS"]

#     def _init_parameter_arrays(self):
#         N = self.num_envs
#         self.g_arr = np.full(N, self.g_nom)
#         self.drag_kx_arr = np.full(N, self.kx_nom)
#         self.drag_ky_arr = np.full(N, self.ky_nom)
#         self.tau_phi_arr = np.full(N, self.tau_phi_nom)
#         self.tau_theta_arr = np.full(N, self.tau_theta_nom)
#         self.tau_T_arr = np.full(N, self.tau_T_nom)
#         self.max_bank_angle_arr = np.full(N, self.bank_angle_nom)
#         self.max_accel_arr = np.full(N, self.max_accel_nom)
#         self.T_min_arr = np.full(N, self.T_range_nom[0])
#         self.T_max_arr = np.full(N, self.T_range_nom[1])
#         self.delta_a_min_arr = np.tile(self.delta_a_min_nom, (N, 1))
#         self.delta_a_max_arr = np.tile(self.delta_a_max_nom, (N, 1))
#         self.init_radius_arr = np.full(N, self.init_radius_nom)

#     def _get_randomized(self, nominal, key, n):
#         if key not in self.domain_rand_pct:
#             raise KeyError(f"{key}")
#         pct = self.domain_rand_pct.get(key)
#         if isinstance(nominal, (list, np.ndarray)) and np.array(nominal).ndim == 1:
#             nominal = np.array(nominal, dtype=np.float64)
#             factor = np.random.uniform(1 - pct, 1 + pct, size=(n, nominal.shape[0]))
#             return nominal * factor
#         else:
#             factor = np.random.uniform(1 - pct, 1 + pct, size=n)
#             return factor * nominal

#     def _randomize_physical_parameters(self, mask):
#         idx = np.nonzero(mask)[0]
#         n = len(idx)
#         if n == 0:
#             return

#         self.g_arr[idx] = self._get_randomized(self.g_nom, "g", n)
#         self.drag_kx_arr[idx] = self._get_randomized(self.kx_nom, "kx_acc_ctbr", n)
#         self.drag_ky_arr[idx] = self._get_randomized(self.ky_nom, "ky_acc_ctbr", n)
#         self.tau_phi_arr[idx] = self._get_randomized(self.tau_phi_nom, "tauphi", n)
#         self.tau_theta_arr[idx] = self._get_randomized(
#             self.tau_theta_nom, "tautheta", n
#         )
#         self.tau_T_arr[idx] = self._get_randomized(self.tau_T_nom, "tauT", n)
#         self.max_bank_angle_arr[idx] = self._get_randomized(
#             self.bank_angle_nom, "bank_angle", n
#         )
#         self.max_accel_arr[idx] = self._get_randomized(
#             self.max_accel_nom, "max_accel", n
#         )
#         self.T_max_arr[idx] = self._get_randomized(self.T_range_nom[1], "T_hi", n)
#         self.T_min_arr[idx] = 0.0
#         self.delta_a_min_arr[idx] = self._get_randomized(
#             self.delta_a_min_nom, "delta_a_min", n
#         )
#         self.delta_a_max_arr[idx] = self._get_randomized(
#             self.delta_a_max_nom, "delta_a_max", n
#         )
#         self.init_radius_arr[idx] = self._get_randomized(
#             self.init_radius_nom, "init_radius", n
#         )

#     def _sample_initial_positions(self, idx):
#         n = len(idx)
#         dirs = np.random.normal(size=(n, 3))
#         dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
#         radii = self.init_radius_arr[idx][:, None]
#         return self.init_pos + radii * dirs

#     def _sample_initial_velocities(self, n):
#         return np.random.uniform(-0.5, 0.5, size=(n, 3))

#     def _sample_initial_attitudes(self, n):
#         limit = np.pi / 4
#         roll = np.random.uniform(-limit, limit, size=(n, 1))
#         pitch = np.random.uniform(-limit, limit, size=(n, 1))
#         yaw = np.zeros((n, 1))
#         return np.concatenate([roll, pitch, yaw], axis=1)

#     def _sample_initial_thrusts(self, idx):
#         high = self.T_max_arr[idx]
#         return np.random.rand(len(idx), 1) * high[:, None]

#     def _initialize_states(self, idx):
#         n = len(idx)
#         positions = self._sample_initial_positions(idx)
#         velocities = self._sample_initial_velocities(n)
#         attitudes = self._sample_initial_attitudes(n)
#         thrusts = self._sample_initial_thrusts(idx)
#         self.state[idx] = np.concatenate(
#             [positions, velocities, attitudes, thrusts], axis=1
#         )

#     def reset(self, mask):
#         idx = np.nonzero(mask)[0]
#         if len(idx) == 0:
#             return
#         self._randomize_physical_parameters(mask)
#         self._initialize_states(idx)
#         self.acc_measured[idx] = 0.0

#     def _compute_Ga_matrix(self, phi_arr, theta_arr, T_est_arr):
#         sphi, cphi = np.sin(phi_arr), np.cos(phi_arr)
#         stheta, ctheta = np.sin(theta_arr), np.cos(theta_arr)
#         Ga = np.zeros((self.num_envs, 3, 3), dtype=np.float64)
#         Ga[:, 0, 0] = ctheta * cphi * T_est_arr
#         Ga[:, 0, 1] = -stheta * sphi * T_est_arr
#         Ga[:, 0, 2] = stheta * cphi
#         Ga[:, 1, 1] = -cphi * T_est_arr
#         Ga[:, 1, 2] = -sphi
#         Ga[:, 2, 0] = -stheta * cphi * T_est_arr
#         Ga[:, 2, 1] = -ctheta * sphi * T_est_arr
#         Ga[:, 2, 2] = ctheta * cphi
#         return Ga

#     def _compute_derivatives(self, state, control_vec):
#         vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
#         phi, theta, psi, T = state[:, 6], state[:, 7], state[:, 8], state[:, 9]
#         phi_cmd, theta_cmd, T_cmd = (
#             control_vec[:, 0],
#             control_vec[:, 1],
#             control_vec[:, 2],
#         )

#         sphi, cphi = np.sin(phi), np.cos(phi)
#         stheta, ctheta = np.sin(theta), np.cos(theta)
#         spsi, cpsi = np.sin(psi), np.cos(psi)

#         R11 = cpsi * ctheta
#         R12 = cpsi * stheta * sphi - spsi * cphi
#         R13 = cpsi * stheta * cphi + spsi * sphi
#         R21 = spsi * ctheta
#         R22 = spsi * stheta * sphi + cpsi * cphi
#         R23 = spsi * stheta * cphi - cpsi * sphi
#         R31 = -stheta
#         R32 = ctheta * sphi
#         R33 = ctheta * cphi

#         vb_x = R11 * vx + R21 * vy + R31 * vz
#         vb_y = R12 * vx + R22 * vy + R32 * vz
#         Dx = -self.drag_kx_arr * vb_x
#         Dy = -self.drag_ky_arr * vb_y

#         d_vx = R11 * Dx + R12 * Dy + R13 * (-T)
#         d_vy = R21 * Dx + R22 * Dy + R23 * (-T)
#         d_vz = R31 * Dx + R32 * Dy + R33 * (-T) + self.g_arr

#         d_phi = (phi_cmd - phi) / self.tau_phi_arr
#         d_theta = (theta_cmd - theta) / self.tau_theta_arr
#         d_T = (T_cmd - T) / self.tau_T_arr

#         return np.stack(
#             [vx, vy, vz, d_vx, d_vy, d_vz, d_phi, d_theta, np.zeros_like(psi), d_T],
#             axis=1,
#         )

#     def _indi_control_allocation(self, acc_des):
#         phi, theta, thrust = self.state[:, 6], self.state[:, 7], self.state[:, 9]
#         delta_a = np.clip(
#             acc_des - self.acc_measured,
#             self.delta_a_min_arr,
#             self.delta_a_max_arr,
#         )
#         Ga = self._compute_Ga_matrix(phi, theta, -self.g_arr)
#         Ga_inv = np.linalg.inv(Ga)
#         control_inc = np.einsum("nij,nj->ni", Ga_inv, delta_a)

#         phi_cmd = np.clip(
#             phi + control_inc[:, 1], -self.max_bank_angle_arr, self.max_bank_angle_arr
#         )
#         theta_cmd = np.clip(
#             theta + control_inc[:, 0], -self.max_bank_angle_arr, self.max_bank_angle_arr
#         )
#         thrust_cmd = np.clip(
#             thrust + control_inc[:, 2] * (-500), self.T_min_arr, self.T_max_arr
#         )  # INCREMENT USING SPECIFIC FORCE GAIN

#         return np.stack([phi_cmd, theta_cmd, thrust_cmd], axis=1)

#     def step_learn(self, actions):
#         """
#         Batched step with L2-norm caps using per-env max accel ONLY.
#         Requires:
#             actions.shape == (N, 3)
#             self.max_accel_arr.shape == (N,)
#         """
#         acts = np.asarray(actions, dtype=np.float64)
#         if acts.ndim != 2 or acts.shape[1] != 3:
#             raise ValueError(f"actions must be (N,3), got {acts.shape}")

#         N = acts.shape[0]

#         # ---- per-env max accel (strict) ----
#         max_env = np.asarray(self.max_accel_arr, dtype=np.float64)
#         if max_env.ndim != 1 or max_env.shape[0] != N:
#             raise ValueError(
#                 f"self.max_accel_arr must be shape (N,), got {max_env.shape} for N={N}"
#             )

#         # ---- 1) clip per-axis to [-1, 1] ----
#         acts = np.clip(acts, -1.0, 1.0)  # (N,3)

#         # ---- 2) L2 project to unit ball (only if norm > 1) ----
#         norms = np.linalg.norm(acts, axis=1)  # (N,)
#         scale = np.ones_like(norms)
#         over = norms > 1.0
#         if np.any(over):
#             scale[over] = 1.0 / (norms[over] + 1e-6)
#         acts_unit = acts * scale[:, None]  # (N,3), ||·||2 <= 1

#         # ---- 3) scale by per-env max accel ----
#         acc_cmd = acts_unit * max_env[:, None]  # (N,3), ||·||2 <= max_env
#         self.acc_cmd_raw = acc_cmd.copy()

#         # ---- 4) filter and re-cap L2 per env (filters can overshoot) ----
#         acc_filt = self.acc_filter.apply(acc_cmd)  # (N,3)
#         norms_f = np.linalg.norm(acc_filt, axis=1)  # (N,)
#         scale_f = np.ones_like(norms_f)
#         over_f = norms_f > max_env
#         if np.any(over_f):
#             scale_f[over_f] = max_env[over_f] / (norms_f[over_f] + 1e-6)
#         acc_filt = acc_filt * scale_f[:, None]
#         self.acc_cmd_filtered = acc_filt.copy()

#         # ---- 5) control allocation (batched) ----
#         control_vecs = self._indi_control_allocation(acc_filt)  # (N,3)

#         # ---- 6) integrate dynamics ----
#         derivs = self._compute_derivatives(self.state, control_vecs)  # (N, state_dim)
#         self.state += derivs * self.dt

#         # ---- 7) telemetry ----
#         self.acc_measured = derivs[:, 3:6].copy()
#         self.rates = derivs[:, 6:9].copy()

#         return self.acc_measured[0].copy()

#     def get_state(self):
#         pos = self.state[:, 0:3]
#         vel = self.state[:, 3:6]
#         att = self.state[:, 6:9]
#         T_force = self.state[:, 9]
#         noisy_pos = pos + np.random.normal(0, self.pos_noise_std, size=pos.shape)
#         noisy_vel = vel + np.random.normal(0, self.vel_noise_std, size=vel.shape)
#         return {
#             "true_position": pos.copy(),
#             "noisy_position": noisy_pos,
#             "velocity": vel.copy(),
#             "noisy_velocity": noisy_vel,
#             "acc_measured": self.acc_measured.copy(),
#             "attitude": att.copy(),
#             "rates": self.rates.copy(),
#             "omega_norm": np.zeros((self.num_envs, 4), dtype=np.float64),
#             "T_force": T_force.reshape(-1, 1),
#         }


class VecAcc_INDI_INDI_Pursuer:
    """
    Vectorized Acceleration INDI model with PD attitude loop -> rate commands -> 1st-order rate tracking.

    State per env (13):
      [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r, T]

    External action per env (3):
      normalized acceleration command in [-1,1]^3

    Notes
    -----
    - INDI maps filtered accel to (phi_sp, theta_sp, T_sp).
    - PD on attitude error outputs rate commands (p_cmd,q_cmd,r_cmd).
    - p,q,r follow first-order lags to those commands; Euler angles integrate current rates.
    - Thrust tracks T_sp via first-order lag.
    """

    def __init__(self, config: dict, num_envs: int):
        self.num_envs = num_envs
        self.dt = config["DT"]
        self.config = config["PURSUER"]
        self.domain_rand_pct = self.config.get("domain_randomization_pct", {})

        self._load_nominal_config()
        self._init_parameter_arrays()

        self.pos_noise_std = self.config["POSITION_NOISE_STD"]
        self.vel_noise_std = self.config["VELOCITY_NOISE_STD"]
        cutoff_hz = self.config["BUTTER_ACC_FILTER_CUTOFF_HZ"]
        self.acc_filter = ButterworthFilter(
            num_envs=num_envs, dim=3, dt=self.dt, cutoff_hz=cutoff_hz, order=2
        )

        self.state = np.zeros((num_envs, 13), dtype=np.float64)
        self.acc_measured = np.zeros((num_envs, 3), dtype=np.float64)
        self.rates = np.zeros((num_envs, 3), dtype=np.float64)
        self.acc_cmd_raw = np.zeros((num_envs, 3), dtype=np.float64)
        self.acc_cmd_filtered = np.zeros((num_envs, 3), dtype=np.float64)
        self.rates_cmd = np.zeros((num_envs, 3), dtype=np.float64)

        self.reset(np.ones(num_envs, dtype=bool))

    def _load_nominal_config(self):
        c = self.config
        self.g_nom = c["gravity"]
        self.kx_nom = c["drag"]["kx_acc_ctbr"]
        self.ky_nom = c["drag"]["ky_acc_ctbr"]

        self.tau_p_nom = c["actuator_time_constants"]["p"]
        self.tau_q_nom = c["actuator_time_constants"]["q"]
        self.tau_r_nom = c["actuator_time_constants"]["r"]
        self.tau_T_nom = c["actuator_time_constants"]["T"]

        apd = c["attitude_pd"]
        self.kp_nom = np.array(
            [apd["kp"]["phi"], apd["kp"]["theta"], apd["kp"]["psi"]], dtype=np.float64
        )
        self.kd_nom = np.array(
            [apd["kd"]["phi"], apd["kd"]["theta"], apd["kd"]["psi"]], dtype=np.float64
        )

        self.bank_angle_nom = np.radians(c["actuator_limits"]["bank_angle"])
        self.p_lim_nom = np.array(c["actuator_limits"]["p"], dtype=np.float64)
        self.q_lim_nom = np.array(c["actuator_limits"]["q"], dtype=np.float64)
        self.r_lim_nom = np.array(c["actuator_limits"]["r"], dtype=np.float64)
        self.T_range_nom = np.array(c["actuator_limits"]["T"], dtype=np.float64)

        self.max_accel_nom = c["MAX_ACCELERATION"]
        self.delta_a_min_nom = np.array(c["delta_a_limits"]["min"], dtype=np.float64)
        self.delta_a_max_nom = np.array(c["delta_a_limits"]["max"], dtype=np.float64)
        self.init_radius_nom = c["INIT_RADIUS"]

        # specific-force gain used for thrust increment in INDI TAKEN FROM PPRZ
        self.k_T_spec = -1.0

        self.init_pos = np.array(c["INITIAL_POS"], dtype=np.float64)
        self.init_vel = np.array(c["INITIAL_VEL"], dtype=np.float64)

    def _init_parameter_arrays(self):
        N = self.num_envs
        self.g_arr = np.full(N, self.g_nom)
        self.drag_kx_arr = np.full(N, self.kx_nom)
        self.drag_ky_arr = np.full(N, self.ky_nom)

        self.tau_p_arr = np.full(N, self.tau_p_nom)
        self.tau_q_arr = np.full(N, self.tau_q_nom)
        self.tau_r_arr = np.full(N, self.tau_r_nom)
        self.tau_T_arr = np.full(N, self.tau_T_nom)

        self.kp_arr = np.tile(self.kp_nom, (N, 1))  # (N,3) [phi,theta,psi]
        self.kd_arr = np.tile(self.kd_nom, (N, 1))

        self.max_bank_angle_arr = np.full(N, self.bank_angle_nom)
        self.p_min_arr = np.full(N, self.p_lim_nom[0])
        self.p_max_arr = np.full(N, self.p_lim_nom[1])
        self.q_min_arr = np.full(N, self.q_lim_nom[0])
        self.q_max_arr = np.full(N, self.q_lim_nom[1])
        self.r_min_arr = np.full(N, self.r_lim_nom[0])
        self.r_max_arr = np.full(N, self.r_lim_nom[1])
        self.T_min_arr = np.full(N, self.T_range_nom[0])
        self.T_max_arr = np.full(N, self.T_range_nom[1])

        self.max_accel_arr = np.full(N, self.max_accel_nom)
        self.delta_a_min_arr = np.tile(self.delta_a_min_nom, (N, 1))
        self.delta_a_max_arr = np.tile(self.delta_a_max_nom, (N, 1))

        self.init_radius_arr = np.full(N, self.init_radius_nom)

    def _get_randomized(self, nominal, key, n):
        if key not in self.domain_rand_pct:
            return (
                np.full(n, nominal)
                if np.isscalar(nominal)
                else np.tile(np.asarray(nominal), (n, 1))
            )
        pct = self.domain_rand_pct.get(key)
        if isinstance(nominal, (list, np.ndarray)) and np.array(nominal).ndim == 1:
            nominal = np.array(nominal, dtype=np.float64)
            factor = np.random.uniform(1 - pct, 1 + pct, size=(n, nominal.shape[0]))
            return nominal * factor
        else:
            factor = np.random.uniform(1 - pct, 1 + pct, size=n)
            return factor * nominal

    def _randomize_physical_parameters(self, mask):
        idx = np.nonzero(mask)[0]
        n = len(idx)
        if n == 0:
            return
        self.g_arr[idx] = self._get_randomized(self.g_nom, "g", n)
        self.drag_kx_arr[idx] = self._get_randomized(self.kx_nom, "kx_acc_ctbr", n)
        self.drag_ky_arr[idx] = self._get_randomized(self.ky_nom, "ky_acc_ctbr", n)

        self.tau_p_arr[idx] = self._get_randomized(self.tau_p_nom, "taup", n)
        self.tau_q_arr[idx] = self._get_randomized(self.tau_q_nom, "tauq", n)
        self.tau_r_arr[idx] = self._get_randomized(self.tau_r_nom, "taur", n)
        self.tau_T_arr[idx] = self._get_randomized(self.tau_T_nom, "tauT", n)

        self.max_bank_angle_arr[idx] = self._get_randomized(
            self.bank_angle_nom, "bank_angle", n
        )
        self.max_accel_arr[idx] = self._get_randomized(
            self.max_accel_nom, "max_accel", n
        )

        self.T_max_arr[idx] = self._get_randomized(self.T_range_nom[1], "T_hi", n)
        self.T_min_arr[idx] = 0.0

        self.delta_a_min_arr[idx] = self._get_randomized(
            self.delta_a_min_nom, "delta_a_min", n
        )
        self.delta_a_max_arr[idx] = self._get_randomized(
            self.delta_a_max_nom, "delta_a_max", n
        )
        self.init_radius_arr[idx] = self._get_randomized(
            self.init_radius_nom, "init_radius", n
        )

        # (Optional) randomize gains if desired:
        # self.kp_arr[idx] = self._get_randomized(self.kp_nom, "kp_att", n)
        # self.kd_arr[idx] = self._get_randomized(self.kd_nom, "kd_att", n)

    # -------- init state --------
    def _sample_initial_positions(self, idx):
        n = len(idx)
        dirs = np.random.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        radii = self.init_radius_arr[idx][:, None]
        return self.init_pos + radii * dirs

    def _sample_initial_velocities(self, n):
        return np.random.uniform(-0.5, 0.5, size=(n, 3))

    def _sample_initial_attitudes(self, n):
        limit = np.pi / 4
        roll = np.random.uniform(-limit, limit, size=(n, 1))
        pitch = np.random.uniform(-limit, limit, size=(n, 1))
        yaw = np.zeros((n, 1))
        return np.concatenate([roll, pitch, yaw], axis=1)

    def _sample_initial_rates(self, n):
        return np.random.uniform(-2.0, 2.0, size=(n, 3))

    def _sample_initial_thrusts(self, idx):
        high = self.T_max_arr[idx]
        return np.random.rand(len(idx), 1) * high[:, None]

    def _initialize_states(self, idx):
        n = len(idx)
        positions = self._sample_initial_positions(idx)
        velocities = self._sample_initial_velocities(n)
        attitudes = self._sample_initial_attitudes(n)
        rates = self._sample_initial_rates(n)
        thrusts = self._sample_initial_thrusts(idx)
        self.state[idx] = np.concatenate(
            [positions, velocities, attitudes, rates, thrusts], axis=1
        )

    def reset(self, mask):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return
        self._randomize_physical_parameters(mask)
        self._initialize_states(idx)
        self.acc_measured[idx] = 0.0
        self.rates[idx] = self.state[idx, 9:12]

    def _compute_Ga_matrix(self, phi_arr, theta_arr, T_est_arr):
        sphi, cphi = np.sin(phi_arr), np.cos(phi_arr)
        stheta, ctheta = np.sin(theta_arr), np.cos(theta_arr)
        Ga = np.zeros((phi_arr.shape[0], 3, 3), dtype=np.float64)
        Ga[:, 0, 0] = ctheta * cphi * T_est_arr
        Ga[:, 0, 1] = -stheta * sphi * T_est_arr
        Ga[:, 0, 2] = stheta * cphi
        Ga[:, 1, 1] = -cphi * T_est_arr
        Ga[:, 1, 2] = -sphi
        Ga[:, 2, 0] = -stheta * cphi * T_est_arr
        Ga[:, 2, 1] = -ctheta * sphi * T_est_arr
        Ga[:, 2, 2] = ctheta * cphi
        return Ga

    def _indi_control_allocation(self, acc_des):
        """
        Batched INDI: acc_des (N,3) -> [phi_sp, theta_sp, T_sp] (N,3)
        """
        phi, theta, thrust = self.state[:, 6], self.state[:, 7], self.state[:, 12]

        delta_a = np.clip(
            acc_des - self.acc_measured, self.delta_a_min_arr, self.delta_a_max_arr
        )

        Ga = self._compute_Ga_matrix(phi, theta, -self.g_arr)
        Ga_inv = np.linalg.inv(Ga)
        control_inc = np.einsum("nij,nj->ni", Ga_inv, delta_a)

        phi_sp = np.clip(
            phi + control_inc[:, 1], -self.max_bank_angle_arr, self.max_bank_angle_arr
        )
        theta_sp = np.clip(
            theta + control_inc[:, 0], -self.max_bank_angle_arr, self.max_bank_angle_arr
        )
        T_sp = np.clip(
            thrust + control_inc[:, 2] * self.k_T_spec, self.T_min_arr, self.T_max_arr
        )

        return np.stack([phi_sp, theta_sp, T_sp], axis=1)

    def _attitude_pd_to_rate_cmds(self, phi_sp, theta_sp, psi_sp=0.0):
        """
        Vectorized PD: attitude error -> rate commands.
        psi_sp can be scalar or per-env array; default 0 (yaw hold).
        """
        phi, theta, psi = self.state[:, 6], self.state[:, 7], self.state[:, 8]
        p, q, r = self.state[:, 9], self.state[:, 10], self.state[:, 11]

        kp_phi, kp_theta, kp_psi = (
            self.kp_arr[:, 0],
            self.kp_arr[:, 1],
            self.kp_arr[:, 2],
        )
        kd_phi, kd_theta, kd_psi = (
            self.kd_arr[:, 0],
            self.kd_arr[:, 1],
            self.kd_arr[:, 2],
        )

        p_cmd = kp_phi * (phi_sp - phi) - kd_phi * p
        q_cmd = kp_theta * (theta_sp - theta) - kd_theta * q
        psi_sp_arr = (
            psi_sp
            if isinstance(psi_sp, np.ndarray)
            else np.full(self.num_envs, psi_sp, dtype=np.float64)
        )
        r_cmd = kp_psi * (psi_sp_arr - psi) - kd_psi * r

        p_cmd = np.clip(p_cmd, self.p_min_arr, self.p_max_arr)
        q_cmd = np.clip(q_cmd, self.q_min_arr, self.q_max_arr)
        r_cmd = np.clip(r_cmd, self.r_min_arr, self.r_max_arr)

        self.rates_cmd[:] = np.stack([p_cmd, q_cmd, r_cmd], axis=1)
        return self.rates_cmd

    def _compute_derivatives(self, state, control_vec):
        """
        control_vec: (N,4) = [p_cmd, q_cmd, r_cmd, T_cmd]
        returns derivs: (N,13)
        """
        vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
        phi, theta, psi = state[:, 6], state[:, 7], state[:, 8]
        p, q, r = state[:, 9], state[:, 10], state[:, 11]
        T = state[:, 12]

        p_cmd, q_cmd, r_cmd, T_cmd = (
            control_vec[:, 0],
            control_vec[:, 1],
            control_vec[:, 2],
            control_vec[:, 3],
        )

        sphi, cphi = np.sin(phi), np.cos(phi)
        stheta, ctheta = np.sin(theta), np.cos(theta)
        spsi, cpsi = np.sin(psi), np.cos(psi)

        R11 = cpsi * ctheta
        R12 = cpsi * stheta * sphi - spsi * cphi
        R13 = cpsi * stheta * cphi + spsi * sphi
        R21 = spsi * ctheta
        R22 = spsi * stheta * sphi + cpsi * cphi
        R23 = spsi * stheta * cphi - cpsi * sphi
        R31 = -stheta
        R32 = ctheta * sphi
        R33 = ctheta * cphi

        vb_x = R11 * vx + R21 * vy + R31 * vz
        vb_y = R12 * vx + R22 * vy + R32 * vz
        Dx = -self.drag_kx_arr * vb_x
        Dy = -self.drag_ky_arr * vb_y

        d_vx = R11 * Dx + R12 * Dy + R13 * (-T)
        d_vy = R21 * Dx + R22 * Dy + R23 * (-T)
        d_vz = R31 * Dx + R32 * Dy + R33 * (-T) + self.g_arr

        d_phi = (
            p
            + q * sphi * (1.0 / np.maximum(ctheta, 1e-6)) * stheta
            + r * cphi * (1.0 / np.maximum(ctheta, 1e-6)) * stheta
        )
        d_phi = p + q * sphi * np.tan(theta) + r * cphi * np.tan(theta)
        d_theta = q * cphi - r * sphi
        d_psi = (q * sphi + r * cphi) / np.maximum(ctheta, 1e-6)

        d_p = (p_cmd - p) / self.tau_p_arr
        d_q = (q_cmd - q) / self.tau_q_arr
        d_r = (r_cmd - r) / self.tau_r_arr

        d_T = (T_cmd - T) / self.tau_T_arr

        d_x, d_y, d_z = vx, vy, vz

        return np.stack(
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
                d_T,
            ],
            axis=1,
        )

    # -------- public step --------
    def step_learn(self, actions):
        """
        Batched step. `actions` shape (N,3) in [-1,1].
        1) cap accel per env, 2) filter, 3) INDI -> [phi_sp,theta_sp,T_sp],
        4) PD -> [p_cmd,q_cmd,r_cmd], 5) 1st-order rate/thrust tracking, 6) integrate.
        """
        acts = np.asarray(actions, dtype=np.float64)
        if acts.ndim != 2 or acts.shape[1] != 3:
            raise ValueError(f"actions must be (N,3), got {acts.shape}")
        N = acts.shape[0]
        if N != self.num_envs:
            raise ValueError(f"N mismatch: actions {N}, model {self.num_envs}")

        acts = np.clip(acts, -1.0, 1.0)
        norms = np.linalg.norm(acts, axis=1)
        scale = np.ones_like(norms)
        over = norms > 1.0
        if np.any(over):
            scale[over] = 1.0 / (norms[over] + 1e-6)
        acts_unit = acts * scale[:, None]

        acc_cmd = acts_unit * self.max_accel_arr[:, None]
        self.acc_cmd_raw = acc_cmd.copy()

        acc_filt = self.acc_filter.apply(acc_cmd)
        norms_f = np.linalg.norm(acc_filt, axis=1)
        scale_f = np.ones_like(norms_f)
        over_f = norms_f > self.max_accel_arr
        if np.any(over_f):
            scale_f[over_f] = self.max_accel_arr[over_f] / (norms_f[over_f] + 1e-6)
        acc_filt *= scale_f[:, None]
        self.acc_cmd_filtered = acc_filt.copy()

        attT_sp = self._indi_control_allocation(acc_filt)
        phi_sp, theta_sp, T_sp = attT_sp[:, 0], attT_sp[:, 1], attT_sp[:, 2]

        rate_cmds = self._attitude_pd_to_rate_cmds(
            phi_sp, theta_sp, psi_sp=0.0
        )  # (N,3)

        control_vecs = np.concatenate([rate_cmds, T_sp[:, None]], axis=1)  # (N,4)
        derivs = self._compute_derivatives(self.state, control_vecs)
        self.state += derivs * self.dt

        self.acc_measured = derivs[:, 3:6].copy()
        self.rates = self.state[:, 9:12].copy()

        return self.acc_measured[0].copy()

    def get_state(self):
        pos = self.state[:, 0:3]
        vel = self.state[:, 3:6]
        att = self.state[:, 6:9]
        T_force = self.state[:, 12]
        noisy_pos = pos + np.random.normal(0, self.pos_noise_std, size=pos.shape)
        noisy_vel = vel + np.random.normal(0, self.vel_noise_std, size=vel.shape)
        return {
            "true_position": pos.copy(),
            "noisy_position": noisy_pos,
            "velocity": vel.copy(),
            "noisy_velocity": noisy_vel,
            "acc_measured": self.acc_measured.copy(),
            "attitude": att.copy(),
            "rates": self.rates.copy(),
            "rates_command": self.rates_cmd.copy(),
            "acc_command": self.acc_cmd_raw.copy(),
            "acc_command_filtered": self.acc_cmd_filtered.copy(),
            "omega_norm": np.zeros((self.num_envs, 4), dtype=np.float64),
            "T_force": T_force.reshape(-1, 1),
        }
