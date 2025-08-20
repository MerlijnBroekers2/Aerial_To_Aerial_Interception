import numpy as np
from src.utils.config import CONFIG


# --------------------------------------------------------------------- #
#  Distance shaping from Reinier's thesis
# --------------------------------------------------------------------- #
def _scaled_distance(d, d_INT, *, a=10**-1.5, b=3.0, c=100.0):
    d_rel = d - d_INT
    return ((-1.0 / (d_rel**2 + a)) + (d_rel + b) ** 2 - (b + 1) ** 2 + 1.0 / a) / c


def reward_reinier(dists, caught):
    T = CONFIG["TIME_LIMIT"]
    d_INT = CONFIG["CAPTURE_RADIUS"]
    k_dist = 10.0 / T
    k_time = 5.0 / T
    D_t = _scaled_distance(dists, d_INT)
    return (-k_dist * D_t - k_time + 5.0 * caught).astype(np.float32)


def reward_simple(dists, caught):
    T = CONFIG["TIME_LIMIT"]
    bound = CONFIG["ENV_BOUND"]
    capture_reward = -CONFIG["CAPTURE_PENALTY"]
    return np.where(caught, capture_reward, -dists / (T * bound)).astype(np.float32)


def reward_simple_smooth(dists, caught, action_delta):
    base = reward_simple(dists, caught)
    gamma = CONFIG["SMOOTHING_GAMMA"]
    smoothing_term = gamma * np.exp(-action_delta)
    return (base + smoothing_term).astype(np.float32)


def reward_effective_gain(dists, prev_dists, ev_step, caught):
    progress = prev_dists - dists
    effective_gain = progress - ev_step
    reward = effective_gain / CONFIG["ENV_BOUND"]
    reward[caught] += -CONFIG["CAPTURE_PENALTY"]
    return reward.astype(np.float32)


def reward_distance_gain(dists, prev_dists, caught):
    progress = prev_dists - dists
    reward = progress / CONFIG["ENV_BOUND"]
    reward[caught] += -CONFIG["CAPTURE_PENALTY"]
    return reward.astype(np.float32)


# --------------------------------------------------------------------- #
#  Unified API
# --------------------------------------------------------------------- #
def get_reward(
    dists,
    caught,
    reward_type,
    prev_dists=None,
    ev_step=None,
    action_delta=None,
):
    rt = reward_type.lower()

    if rt == "reinier":
        return reward_reinier(dists, caught)

    if rt == "reward_distance_gain":
        return reward_distance_gain(dists, prev_dists, caught)

    if rt == "simple_smooth":
        if action_delta is None:
            raise ValueError("action_delta must be provided for 'simple_smooth'")
        return reward_simple_smooth(dists, caught, action_delta)

    if rt == "effective_gain":
        if prev_dists is None or ev_step is None:
            raise ValueError(
                "prev_dists and ev_step must be provided for 'effective_gain'"
            )
        return reward_effective_gain(dists, prev_dists, ev_step, caught)

    raise ValueError(f"Unknown reward_type: {reward_type}")
