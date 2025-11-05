import numpy as np
from src.models.evaders.base import IEvader


class VectorizedReactiveEvader(IEvader):
    """
    Batched evader with specific-acceleration command in world frame.

    Noise model:
      - Noise is added to POSITION ONLY.
      - "Noisy velocity" is derived by differentiating the noisy position.
      - "Filtered position" is a window-smoothed average of recent noisy positions
        (or passthrough), and "filtered velocity" is derived by differentiating
        that filtered position, then optionally window-smoothed as well.

    Expects in CONFIG["EVADER"]:
      MAX_ACCEL, MAX_SPEED, INIT_POS, INIT_VEL,
      NOISE_STD,
      FILTER_TYPE: "passthrough" | "window_smoothing"
      FILTER_PARAMS:
        pos_window_samples: int
        vel_window_samples: int
        vel_from_filtered_pos: bool
    """

    def __init__(self, num_envs: int, config: dict):
        self.N = int(num_envs)
        self.cfg = config
        ev = self.cfg["EVADER"]

        self.dt = float(self.cfg["DT"])
        self.max_accel = float(ev["MAX_ACCEL"])
        self.max_speed = float(ev["MAX_SPEED"])

        self.init_pos = np.array(ev["INIT_POS"], dtype=np.float32)
        self.init_vel = np.array(ev["INIT_VEL"], dtype=np.float32)

        # Position noise only
        self.noise_pos = float(ev["NOISE_STD"])

        # Filtering (passthrough | window_smoothing)
        self.filter_type = ev["FILTER_TYPE"]
        self.filter_params = ev.get("FILTER_PARAMS", {})
        if self.filter_type not in ("passthrough", "window_smoothing"):
            raise ValueError("FILTER_TYPE must be 'passthrough' or 'window_smoothing'")

        self.pos_window_samples = int(self.filter_params["pos_window_samples"])
        self.vel_window_samples = int(self.filter_params["vel_window_samples"])
        self.vel_from_filtered_pos_for_filter = bool(
            self.filter_params["vel_from_filtered_pos"]
        )

        # State buffers (ground truth)
        self.pos = np.zeros((self.N, 3), dtype=np.float32)
        self.vel = np.zeros((self.N, 3), dtype=np.float32)
        self.acc_last = np.zeros((self.N, 3), dtype=np.float32)

        # Measurement buffers (derived from noisy positions)
        self.noisy_pos = np.zeros((self.N, 3), dtype=np.float32)
        self.noisy_vel = np.zeros((self.N, 3), dtype=np.float32)  # diff(noisy_pos)

        # Filtered views
        self.filt_pos = np.zeros((self.N, 3), dtype=np.float32)
        self.filt_vel = np.zeros((self.N, 3), dtype=np.float32)

        # History for smoothing/differentiation
        hist_len = int(max(1, self.pos_window_samples, self.vel_window_samples))
        self._hist_len = hist_len
        self._pos_hist = np.zeros((self.N, hist_len, 3), dtype=np.float32)  # noisy pos
        self._vel_hist = np.zeros(
            (self.N, hist_len, 3), dtype=np.float32
        )  # derived vel (base) for smoothing
        self._hist_ptr = np.zeros(self.N, dtype=np.int32)  # next write index
        self._hist_count = np.zeros(self.N, dtype=np.int32)  # filled length

        # Previous samples for finite-difference
        self._noisy_pos_prev = np.zeros((self.N, 3), dtype=np.float32)
        self._filt_pos_prev = np.zeros((self.N, 3), dtype=np.float32)

        self.reset(np.ones(self.N, dtype=bool))

    # ---------- IEvader API ----------
    def reset(self, dones: np.ndarray) -> None:
        idx = np.nonzero(dones)[0]
        if idx.size == 0:
            return

        self.pos[idx] = self.init_pos
        self.vel[idx] = self.init_vel
        self.acc_last[idx] = 0.0

        # Noisy position (position noise only)
        if self.noise_pos > 0.0:
            self.noisy_pos[idx] = self.pos[idx] + np.random.normal(
                0.0, self.noise_pos, self.pos[idx].shape
            ).astype(np.float32)
        else:
            self.noisy_pos[idx] = self.pos[idx]

        # Reset histories
        self._hist_ptr[idx] = 0
        self._hist_count[idx] = 0
        self._pos_hist[idx, :, :] = 0.0
        self._vel_hist[idx, :, :] = 0.0

        # Initialize previous positions for finite-difference
        self._noisy_pos_prev[idx] = self.noisy_pos[idx]
        self._filt_pos_prev[idx] = self.noisy_pos[idx]  # filt_pos will start as noisy

        # Push first samples and compute filtered outputs
        self._push_pos_hist(idx, self.noisy_pos[idx])
        self._update_filtered(idx)  # sets filt_pos
        # Initialize velocities to zero at reset
        self.noisy_vel[idx] = 0.0
        self.filt_vel[idx] = 0.0

    def step(self, acc_cmd) -> None:
        """
        accel_cmd: (N,3) in [-1,1], mapped to [-MAX_ACCEL, MAX_ACCEL]
        """
        # Integrate true dynamics
        a = np.clip(acc_cmd, -1.0, 1.0).astype(np.float32) * self.max_accel
        self.acc_last = a
        self.vel += a * self.dt

        # Speed clamp
        speeds = np.linalg.norm(self.vel, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_speed / (speeds + 1e-8))
        self.vel *= scale

        self.pos += self.vel * self.dt

        # Noisy position (position noise only)
        if self.noise_pos > 0.0:
            self.noisy_pos = self.pos + np.random.normal(
                0.0, self.noise_pos, self.pos.shape
            ).astype(np.float32)
        else:
            self.noisy_pos = self.pos

        # Derive noisy velocity from noisy position (no smoothing)
        noisy_vel_base = (self.noisy_pos - self._noisy_pos_prev) / self.dt
        self.noisy_vel = noisy_vel_base.astype(np.float32)

        # Update histories and filters
        all_idx = np.arange(self.N)
        self._push_pos_hist(all_idx, self.noisy_pos)
        self._update_filtered(all_idx)  # updates filt_pos and filt_vel

        # Update prev positions for next step
        self._noisy_pos_prev = self.noisy_pos.copy()
        self._filt_pos_prev = self.filt_pos.copy()

    def get_state(self) -> dict:
        return {
            "true_position": self.pos.copy(),
            "velocity": self.vel.copy(),
            "noisy_position": self.noisy_pos.copy(),
            "noisy_velocity": self.noisy_vel.copy(),  # derived from noisy pos
            "filtered_position": self.filt_pos.copy(),  # smoothed noisy pos
            "filtered_velocity": self.filt_vel.copy(),  # derived from filtered pos (+ optional smoothing)
        }

    # ---------- filtering helpers ----------
    def _push_pos_hist(self, idx_batch, noisy_pos_batch):
        """
        Append latest NOISY positions into ring buffer for the given env indices.

        Supports either:
          - noisy_pos_batch shape == (N,3)  (full array; index by absolute env id), or
          - noisy_pos_batch shape == (len(idx_batch),3) aligned with idx_batch order.
        """
        if np.isscalar(idx_batch):
            idx_batch = np.array([int(idx_batch)], dtype=np.int32)
        else:
            idx_batch = np.asarray(idx_batch, dtype=np.int32)

        noisy_pos_batch = np.asarray(noisy_pos_batch, dtype=np.float32)

        full_batch = noisy_pos_batch.shape[0] == self.N

        if full_batch:
            for i in idx_batch:
                p = int(self._hist_ptr[i])
                self._pos_hist[i, p, :] = noisy_pos_batch[i]
                self._hist_ptr[i] = (p + 1) % self._hist_len
                self._hist_count[i] = min(self._hist_len, self._hist_count[i] + 1)
        else:
            for j, i in enumerate(idx_batch):
                p = int(self._hist_ptr[i])
                self._pos_hist[i, p, :] = noisy_pos_batch[j]
                self._hist_ptr[i] = (p + 1) % self._hist_len
                self._hist_count[i] = min(self._hist_len, self._hist_count[i] + 1)

    def _window_mean_last(self, arr_seq, ptr, count, k):
        """
        Mean of last k samples from ring buffer arr_seq: (L,3) for a single env.
        """
        if k <= 1 or count <= 1:
            # return most recent sample
            last_idx = (ptr - 1) % self._hist_len
            return arr_seq[last_idx : last_idx + 1, :].mean(axis=0)
        m = int(min(k, count))
        rolled = np.roll(arr_seq, -ptr, axis=0)[:count, :]
        tail = rolled[count - m : count, :]
        return tail.mean(axis=0)

    def _update_filtered(self, idx_batch):
        """
        Compute filtered position from window smoothing of NOISY positions,
        then derive filtered velocity from position differences and optionally
        smooth velocity with a window.
        """
        if np.isscalar(idx_batch):
            idx_batch = np.array([idx_batch], dtype=np.int32)

        for i in np.atleast_1d(idx_batch):
            if self.filter_type == "passthrough":
                # passthrough position
                self.filt_pos[i] = self._pos_hist[
                    i, (self._hist_ptr[i] - 1) % self._hist_len, :
                ]
                # derive vel from filtered pos (which equals noisy pos here)
                vel_base = (self.filt_pos[i] - self._filt_pos_prev[i]) / self.dt
                # optional vel smoothing (still allowed)
                kvel = max(1, self.vel_window_samples)
                # push base into vel history slot for consistent smoothing
                p = (self._hist_ptr[i] - 1) % self._hist_len
                self._vel_hist[i, p, :] = vel_base.astype(np.float32)
                self.filt_vel[i] = self._window_mean_last(
                    self._vel_hist[i], self._hist_ptr[i], self._hist_count[i], kvel
                ).astype(np.float32)
                continue

            # window_smoothing on positions
            kpos = max(1, self.pos_window_samples)
            self.filt_pos[i] = self._window_mean_last(
                self._pos_hist[i], self._hist_ptr[i], self._hist_count[i], kpos
            ).astype(np.float32)

            # base velocity from POSITION differences
            if self.vel_from_filtered_pos_for_filter:
                vel_base = (self.filt_pos[i] - self._filt_pos_prev[i]) / self.dt
            else:
                # derive from noisy pos instead
                last_noisy = self._pos_hist[
                    i, (self._hist_ptr[i] - 1) % self._hist_len, :
                ]
                prev_noisy_idx = (self._hist_ptr[i] - 2) % self._hist_len
                prev_noisy = (
                    self._pos_hist[i, prev_noisy_idx, :]
                    if self._hist_count[i] > 1
                    else last_noisy
                )
                vel_base = (last_noisy - prev_noisy) / self.dt

            # push base velocity into history and (optionally) smooth it
            p = (self._hist_ptr[i] - 1) % self._hist_len
            self._vel_hist[i, p, :] = vel_base.astype(np.float32)

            kvel = max(1, self.vel_window_samples)
            self.filt_vel[i] = self._window_mean_last(
                self._vel_hist[i], self._hist_ptr[i], self._hist_count[i], kvel
            ).astype(np.float32)
