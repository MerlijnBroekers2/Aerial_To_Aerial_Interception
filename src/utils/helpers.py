import numpy as np
import numpy as np
from scipy.signal import butter, lfilter_zi, lfilter


def split_vector(v):
    """Split N×3 vector into unit vector and magnitude"""
    mag = np.linalg.norm(v, axis=1, keepdims=True)
    safe_mag = np.where(mag < 1e-6, 1e-6, mag)  # avoid div by 0
    unit = v / safe_mag
    return unit, mag


class ButterworthFilter:
    def __init__(self, num_envs, dim, dt, cutoff_hz, order=2):
        """
        Vectorized Butterworth filter for batched signals.
        :param num_envs: Number of parallel environments
        :param dim: Number of features per signal (e.g. 3 for acceleration)
        :param dt: Time step duration
        :param cutoff_hz: Filter cutoff frequency in Hz
        :param order: Filter order (default 2)
        """
        self.num_envs = num_envs
        self.dim = dim
        self.dt = dt
        self.order = order

        fs = 1.0 / dt
        nyq = fs / 2.0
        wn = cutoff_hz / nyq

        self.b, self.a = butter(order, wn, btype="low", analog=False)

        # Initial condition per (env, dim, filter_state)
        zi = lfilter_zi(self.b, self.a).astype(np.float64)  # (order,)
        self.z = np.tile(zi[None, None, :], (num_envs, dim, 1))  # (N, D, S)

    def reset(self, mask=None):
        """
        Reset filter states. Resets all if mask is None, otherwise selective.
        """
        if mask is None:
            self.z.fill(0.0)
        else:
            idx = np.nonzero(mask)[0]
            self.z[idx] = 0.0

    def apply(self, signal):
        """
        Apply filter to batch of signals (vectorized).
        :param signal: np.ndarray shape (N, D)
        :return: filtered signal of shape (N, D)
        """
        # Reshape for lfilter (batch, time, features) → treat time as 1
        x = signal[:, :, None]  # (N, D, 1)
        y, zf = lfilter(self.b, self.a, x, axis=2, zi=self.z)
        self.z = zf  # update filter state
        return y[:, :, 0]  # (N, D)
