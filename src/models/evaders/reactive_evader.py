import numpy as np
from stable_baselines3 import PPO
from src.models.evaders.base import IEvader


class RLEvader(IEvader):
    """
    Single-environment reactive evader driven by a pretrained PPO policy.

    Policy observation (NOISY, 9D):
        [ Δp_noisy, Δv_noisy, p_e_noisy ]
      where Δp_noisy = p_p(noisy) - p_e(noisy)
            Δv_noisy = v_p(noisy) - v_e(noisy)
      v_e(noisy) is derived by differentiating noisy position (no direct vel noise).

    Filtering:
      FILTER_TYPE: "passthrough" | "window_smoothing"
      FILTER_PARAMS:
        pos_window_samples: int
        vel_window_samples: int
        vel_from_filtered_pos: bool
    """

    def __init__(
        self, config: dict, model_path: str = None, deterministic: bool = True
    ):
        self.cfg = config
        ev = self.cfg["EVADER"]

        self.dt = float(self.cfg["DT"])
        self.max_accel = float(ev["MAX_ACCEL"])
        self.max_speed = float(ev["MAX_SPEED"])

        self.init_pos = np.asarray(ev["INIT_POS"], dtype=np.float32)
        self.init_vel = np.asarray(ev["INIT_VEL"], dtype=np.float32)

        # Position noise only
        self.noise_pos = float(ev["NOISE_STD_POS"])

        # Filtering (passthrough | window_smoothing)
        self.filter_type = ev.get("FILTER_TYPE", "passthrough")
        self.filter_params = ev.get("FILTER_PARAMS", {})
        if self.filter_type not in ("passthrough", "window_smoothing"):
            raise ValueError("FILTER_TYPE must be 'passthrough' or 'window_smoothing'")
        self.pos_window_samples = int(self.filter_params.get("pos_window_samples", 1))
        self.vel_window_samples = int(self.filter_params.get("vel_window_samples", 1))
        self.vel_from_filtered_pos_for_filter = bool(
            self.filter_params.get("vel_from_filtered_pos", True)
        )

        # Load pretrained PPO
        self.model_path = model_path or ev["RL_MODEL_PATH"]
        self.policy: PPO = PPO.load(self.model_path)
        self.deterministic = deterministic

        # Ground-truth state
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.acc_last = np.zeros(3, dtype=np.float32)

        # Noisy and filtered views
        self.noisy_pos = np.zeros(3, dtype=np.float32)
        self.noisy_vel = np.zeros(3, dtype=np.float32)  # derived from noisy pos
        self.filt_pos = np.zeros(3, dtype=np.float32)
        self.filt_vel = np.zeros(3, dtype=np.float32)

        # History for smoothing/differentiation
        self._hist_len = int(max(1, self.pos_window_samples, self.vel_window_samples))
        self._pos_hist = np.zeros((self._hist_len, 3), dtype=np.float32)  # noisy pos
        self._vel_hist = np.zeros(
            (self._hist_len, 3), dtype=np.float32
        )  # derived vel base
        self._hist_ptr = 0
        self._hist_count = 0

        # Previous positions for finite-difference
        self._noisy_pos_prev = np.zeros(3, dtype=np.float32)
        self._filt_pos_prev = np.zeros(3, dtype=np.float32)

        self.reset()

    # ---------- IEvader API ----------
    def reset(self) -> None:
        self.pos = self.init_pos.copy()
        self.vel = self.init_vel.copy()
        self.acc_last[:] = 0.0

        # Noisy position at reset
        if self.noise_pos > 0.0:
            self.noisy_pos = self.pos + np.random.normal(
                0.0, self.noise_pos, size=3
            ).astype(np.float32)
        else:
            self.noisy_pos = self.pos.copy()

        # Reset histories
        self._hist_ptr = 0
        self._hist_count = 0
        self._pos_hist[:, :] = 0.0
        self._vel_hist[:, :] = 0.0

        # Initialize previous positions and push first sample
        self._noisy_pos_prev = self.noisy_pos.copy()
        self._filt_pos_prev = self.noisy_pos.copy()
        self._push_pos_hist(self.noisy_pos)
        self._update_filtered()  # sets filt_pos

        # Initialize velocities to zero at reset
        self.noisy_vel[:] = 0.0
        self.filt_vel[:] = 0.0

    def step(self, pursuer_pos: np.ndarray, pursuer_vel: np.ndarray) -> None:
        """
        pursuer_pos: (3,)  -- expected measurement (noisy) of pursuer position
        pursuer_vel: (3,)  -- expected measurement (noisy) of pursuer velocity
        """
        p_p = np.asarray(pursuer_pos, dtype=np.float32).reshape(3)
        v_p = np.asarray(pursuer_vel, dtype=np.float32).reshape(3)

        # Build NOISY policy obs from current measurements
        dp = p_p - self.noisy_pos
        dv = v_p - self.noisy_vel
        obs = np.concatenate([dp, dv, self.noisy_pos], axis=0).astype(
            np.float32
        )  # (9,)

        # Policy -> accel command in [-1,1] -> scale to m/s^2
        actions, _ = self.policy.predict(obs, deterministic=self.deterministic)
        acc_cmd = (
            np.clip(np.asarray(actions, dtype=np.float32).reshape(3), -1.0, 1.0)
            * self.max_accel
        )
        self.acc_last = acc_cmd

        # Integrate true dynamics with speed cap
        self.vel += acc_cmd * self.dt
        speed = float(np.linalg.norm(self.vel))
        if speed > self.max_speed:
            self.vel *= self.max_speed / (speed + 1e-8)
        self.pos += self.vel * self.dt

        # Noisy position after integration (position noise only)
        if self.noise_pos > 0.0:
            self.noisy_pos = self.pos + np.random.normal(
                0.0, self.noise_pos, size=3
            ).astype(np.float32)
        else:
            self.noisy_pos = self.pos.copy()

        # Derive noisy velocity from noisy position (no smoothing)
        noisy_vel_base = (self.noisy_pos - self._noisy_pos_prev) / self.dt
        self.noisy_vel = noisy_vel_base.astype(np.float32)

        # Update histories and filtered signals
        self._push_pos_hist(self.noisy_pos)
        self._update_filtered()

        # Update prev positions
        self._noisy_pos_prev = self.noisy_pos.copy()
        self._filt_pos_prev = self.filt_pos.copy()

    def get_state(self) -> dict:
        return {
            "true_position": self.pos.copy(),
            "velocity": self.vel.copy(),
            "noisy_position": self.noisy_pos.copy(),
            "noisy_velocity": self.noisy_vel.copy(),  # derived from noisy pos
            "filtered_position": self.filt_pos.copy(),  # smoothed noisy pos
            "filtered_velocity": self.filt_vel.copy(),  # derived (and optionally smoothed)
        }

    # ---------- filtering helpers ----------
    def _push_pos_hist(self, noisy_pos):
        p = self._hist_ptr
        self._pos_hist[p, :] = noisy_pos
        self._hist_ptr = (p + 1) % self._hist_len
        self._hist_count = min(self._hist_len, self._hist_count + 1)

    def _window_mean_last(self, arr_seq, ptr, count, k):
        if k <= 1 or count <= 1:
            last_idx = (ptr - 1) % self._hist_len
            return arr_seq[last_idx, :].astype(np.float32)
        m = int(min(k, count))
        rolled = np.roll(arr_seq, -ptr, axis=0)[:count, :]
        tail = rolled[count - m : count, :]
        return tail.mean(axis=0).astype(np.float32)

    def _update_filtered(self):
        # Filtered position
        if self.filter_type == "passthrough":
            self.filt_pos = self._pos_hist[
                (self._hist_ptr - 1) % self._hist_len, :
            ].copy()
            vel_base = (self.filt_pos - self._filt_pos_prev) / self.dt
        else:
            kpos = max(1, self.pos_window_samples)
            self.filt_pos = self._window_mean_last(
                self._pos_hist, self._hist_ptr, self._hist_count, kpos
            )
            if self.vel_from_filtered_pos_for_filter:
                vel_base = (self.filt_pos - self._filt_pos_prev) / self.dt
            else:
                last_noisy = self._pos_hist[(self._hist_ptr - 1) % self._hist_len, :]
                prev_idx = (self._hist_ptr - 2) % self._hist_len
                prev_noisy = (
                    self._pos_hist[prev_idx, :] if self._hist_count > 1 else last_noisy
                )
                vel_base = (last_noisy - prev_noisy) / self.dt

        # Velocity smoothing (optional window on derived velocity)
        p = (self._hist_ptr - 1) % self._hist_len
        self._vel_hist[p, :] = vel_base.astype(np.float32)
        kvel = max(1, self.vel_window_samples)
        self.filt_vel = self._window_mean_last(
            self._vel_hist, self._hist_ptr, self._hist_count, kvel
        )
