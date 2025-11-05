import numpy as np


class VecCTBR_INDI_Pursuer:
    def __init__(self, config: dict, num_envs: int):
        self.num_envs = num_envs
        self.dt = config["DT"]
        self.config = config["PURSUER"]
        self.domain_rand_pct = self.config["domain_randomization_pct"]

        # Save nominal config values for randomization
        self._load_nominal_config()
        self._init_parameter_arrays()

        # Noise std
        self.pos_noise_std = self.config["POSITION_NOISE_STD"]
        self.vel_noise_std = self.config["VELOCITY_NOISE_STD"]

        # Initial state settings
        self.init_pos = np.array(self.config["INITIAL_POS"], dtype=np.float64)
        self.init_vel = np.array(self.config["INITIAL_VEL"], dtype=np.float64)
        self.init_att = np.array(self.config["INITIAL_ATTITUDE"], dtype=np.float64)
        self.init_rates = np.array(self.config["INITIAL_RATES"], dtype=np.float64)

        self.state = np.zeros((num_envs, 13), dtype=np.float64)
        self.initial_state = np.zeros_like(self.state)
        self._last_acc = np.zeros((num_envs, 3), dtype=np.float64)

        self.reset(np.ones(num_envs, dtype=bool))

    def _load_nominal_config(self):
        c = self.config
        self.g_nom = c["gravity"]
        self.kx_nom = c["drag"]["kx_acc_ctbr"]
        self.ky_nom = c["drag"]["ky_acc_ctbr"]
        ac = c["actuator_time_constants"]
        self.taup_nom = ac["p"]
        self.tauq_nom = ac["q"]
        self.taur_nom = ac["r"]
        self.tauT_nom = ac["T"]
        al = c["actuator_limits"]
        self.p_range_nom = al["p"]
        self.q_range_nom = al["q"]
        self.r_range_nom = al["r"]
        self.T_range_nom = al["T"]
        self.init_radius_nom = c["INIT_RADIUS"]

    def _init_parameter_arrays(self):
        N = self.num_envs
        self.g_arr = np.full(N, self.g_nom, dtype=np.float64)
        self.init_radius_arr = np.full(N, self.init_radius_nom, dtype=np.float64)
        self.kx_arr = np.full(N, self.kx_nom, dtype=np.float64)
        self.ky_arr = np.full(N, self.ky_nom, dtype=np.float64)
        self.taup_arr = np.full(N, self.taup_nom, dtype=np.float64)
        self.tauq_arr = np.full(N, self.tauq_nom, dtype=np.float64)
        self.taur_arr = np.full(N, self.taur_nom, dtype=np.float64)
        self.tauT_arr = np.full(N, self.tauT_nom, dtype=np.float64)
        self.p_min_arr = np.full(N, self.p_range_nom[0], dtype=np.float64)
        self.p_max_arr = np.full(N, self.p_range_nom[1], dtype=np.float64)
        self.q_min_arr = np.full(N, self.q_range_nom[0], dtype=np.float64)
        self.q_max_arr = np.full(N, self.q_range_nom[1], dtype=np.float64)
        self.r_min_arr = np.full(N, self.r_range_nom[0], dtype=np.float64)
        self.r_max_arr = np.full(N, self.r_range_nom[1], dtype=np.float64)
        self.T_min_arr = np.full(N, self.T_range_nom[0], dtype=np.float64)
        self.T_max_arr = np.full(N, self.T_range_nom[1], dtype=np.float64)

        # --------------------------------------------------------------

    def set_init_radius(self, new_radius: float):
        """Update nominal and per-env radius (called by curriculum)."""
        self.init_radius_nom = new_radius
        self.init_radius_arr[:] = new_radius

    # optional: query
    def get_init_radius(self) -> float:
        return float(self.init_radius_nom)

    def _get_randomized(self, nominal, key, n):
        if key not in self.domain_rand_pct:
            raise KeyError(f"{key}")
        pct = self.domain_rand_pct.get(key, 0.0)
        return np.random.uniform(1 - pct, 1 + pct, size=n) * nominal

    def _randomize_physical_parameters(self, mask):
        idx = np.nonzero(mask)[0]
        n = idx.size
        if n == 0:
            return

        self.g_arr[idx] = self._get_randomized(self.g_nom, "g", n)
        self.kx_arr[idx] = self._get_randomized(self.kx_nom, "kx_acc_ctbr", n)
        self.ky_arr[idx] = self._get_randomized(self.ky_nom, "ky_acc_ctbr", n)
        self.taup_arr[idx] = self._get_randomized(self.taup_nom, "taup", n)
        self.tauq_arr[idx] = self._get_randomized(self.tauq_nom, "tauq", n)
        self.taur_arr[idx] = self._get_randomized(self.taur_nom, "taur", n)
        self.tauT_arr[idx] = self._get_randomized(self.tauT_nom, "tauT", n)
        self.init_radius_arr[idx] = self._get_randomized(
            self.init_radius_nom, "init_radius", n
        )

        def randomize_limit(lo_nom, hi_nom, lo_key, hi_key):
            lo = self._get_randomized(lo_nom, lo_key, n)
            hi = self._get_randomized(hi_nom, hi_key, n)
            return np.minimum(lo, hi), np.maximum(lo, hi)

        self.p_min_arr[idx], self.p_max_arr[idx] = randomize_limit(
            self.p_range_nom[0], self.p_range_nom[1], "p_lo", "p_hi"
        )
        self.q_min_arr[idx], self.q_max_arr[idx] = randomize_limit(
            self.q_range_nom[0], self.q_range_nom[1], "q_lo", "q_hi"
        )
        self.r_min_arr[idx], self.r_max_arr[idx] = randomize_limit(
            self.r_range_nom[0], self.r_range_nom[1], "r_lo", "r_hi"
        )
        self.T_min_arr[idx], self.T_max_arr[idx] = randomize_limit(
            self.T_range_nom[0], self.T_range_nom[1], "T_lo", "T_hi"
        )

    def _compute_derivatives(self, state, control_vec):
        vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
        phi, theta, psi = state[:, 6], state[:, 7], state[:, 8]
        p, q, r = state[:, 9], state[:, 10], state[:, 11]
        Tn = state[:, 12]

        p_cmd, q_cmd, r_cmd, T_cmd = (
            control_vec[:, 0],
            control_vec[:, 1],
            control_vec[:, 2],
            control_vec[:, 3],
        )

        sphi, cphi = np.sin(phi), np.cos(phi)
        stheta, ctheta = np.sin(theta), np.cos(theta)
        spsi, cpsi = np.sin(psi), np.cos(psi)
        tan_theta = stheta / np.maximum(ctheta, 1e-8)

        T = (Tn + 1.0) * 0.5 * (self.T_max_arr - self.T_min_arr) + self.T_min_arr

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
        Dx = -self.kx_arr * vb_x
        Dy = -self.ky_arr * vb_y

        d_vx = R11 * Dx + R12 * Dy + R13 * (-T)
        d_vy = R21 * Dx + R22 * Dy + R23 * (-T)
        d_vz = R31 * Dx + R32 * Dy + R33 * (-T) + self.g_arr

        d_phi = p + q * sphi * tan_theta + r * cphi * tan_theta
        d_theta = q * cphi - r * sphi
        d_psi = (q * sphi + r * cphi) / np.maximum(ctheta, 1e-8)

        p_cmd_real = (p_cmd + 1.0) * 0.5 * (
            self.p_max_arr - self.p_min_arr
        ) + self.p_min_arr
        q_cmd_real = (q_cmd + 1.0) * 0.5 * (
            self.q_max_arr - self.q_min_arr
        ) + self.q_min_arr
        r_cmd_real = (r_cmd + 1.0) * 0.5 * (
            self.r_max_arr - self.r_min_arr
        ) + self.r_min_arr

        d_p = (p_cmd_real - p) / self.taup_arr
        d_q = (q_cmd_real - q) / self.tauq_arr
        d_r = (r_cmd_real - r) / self.taur_arr
        d_Tn = (T_cmd - Tn) / self.tauT_arr

        return np.stack(
            [vx, vy, vz, d_vx, d_vy, d_vz, d_phi, d_theta, d_psi, d_p, d_q, d_r, d_Tn],
            axis=1,
        )

    def step_learn(self, control_vec):
        control_vec = np.clip(control_vec, -1.0, 1.0)
        derivs = self._compute_derivatives(self.state, control_vec)
        self.state += derivs * self.dt
        self._last_acc = derivs[:, 3:6].copy()
        return self._last_acc.copy()

    def reset(self, mask: np.ndarray):
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return
        self._randomize_physical_parameters(mask)
        positions = self._sample_initial_positions(idx)
        velocities = self._sample_initial_velocities(len(idx))
        attitudes = self._sample_initial_attitudes(len(idx))
        rates = self._sample_initial_rates(len(idx))
        T_norm = self._sample_initial_thrusts(len((idx)))
        new_states = np.concatenate(
            [positions, velocities, attitudes, rates, T_norm], axis=1
        )
        self.state[idx] = new_states
        self.initial_state[idx] = new_states
        self._last_acc[idx] = 0.0

    def _sample_initial_positions(self, idx):
        n = len(idx)
        dirs = np.random.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = self.init_radius_arr[idx][:, None]
        return self.init_pos + radii * dirs

    def _sample_initial_velocities(self, n: int) -> np.ndarray:
        return np.random.uniform(-0.5, 0.5, size=(n, 3))

    def _sample_initial_attitudes(self, n: int) -> np.ndarray:
        limit = np.pi / 4
        roll = np.random.uniform(-limit, limit, size=(n, 1))
        pitch = np.random.uniform(-limit, limit, size=(n, 1))
        yaw = np.zeros((n, 1))
        return np.concatenate([roll, pitch, yaw], axis=1)

    def _sample_initial_rates(self, n: int) -> np.ndarray:
        limit = 2
        return np.random.uniform(-limit, limit, size=(n, 3))

    def _sample_initial_thrusts(self, n):
        return np.random.uniform(-1, 1, size=(n, 1))

    # ------------------------------------------------------------------
    def get_state(self):
        # -------- core state ----------
        pos = self.state[:, 0:3]
        vel = self.state[:, 3:6]
        att = self.state[:, 6:9]
        rates = self.state[:, 9:12]
        T_norm = self.state[:, 12]  # [-1 … 1]

        # physical thrust (same scaling as scalar class)
        T_force = (
            0.5 * (T_norm[:, None] + 1.0) * (self.T_max_arr - self.T_min_arr)[:, None]
            + self.T_min_arr[:, None]
        )  # shape (N,1)

        # -------- noise ---------------
        noisy_pos = pos + np.random.normal(0, self.pos_noise_std, size=pos.shape)
        noisy_vel = vel + np.random.normal(0, self.vel_noise_std, size=vel.shape)

        # -------- assemble dict -------
        N = self.num_envs
        zeros4 = np.zeros((N, 4), dtype=np.float64)
        zeros3 = np.zeros((N, 3), dtype=np.float64)
        zeros1 = np.zeros((N, 1), dtype=np.float64)

        return {
            "true_position": pos.copy(),
            "noisy_position": noisy_pos,
            "velocity": vel.copy(),
            "noisy_velocity": noisy_vel,
            "acceleration": self._last_acc.copy(),  # same as scalar
            "acc_command": zeros3,  # not used
            "acc_command_filtered": zeros3,  # not used
            "acc_measured": self._last_acc.copy(),
            "attitude": att.copy(),
            "attitude_commanded": zeros3,  # not available
            "rates": rates.copy(),
            "rates_command": zeros3,  # not available
            "T_force": T_force.copy(),  # [N,1]  physical
            "T_norm": T_norm.copy(),  # [N,]
            "T_command": zeros1,  # not stored
            "omega_norm": zeros4,
        }
