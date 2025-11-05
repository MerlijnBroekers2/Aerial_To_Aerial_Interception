import numpy as np


# --------------------------------------------------------------------- #
#  Distance shaping from Reinier's thesis
# --------------------------------------------------------------------- #
def _scaled_distance(d, d_INT, *, a=10**-1.5, b=3.0, c=100.0):
    d_rel = d - d_INT
    return ((-1.0 / (d_rel**2 + a)) + (d_rel + b) ** 2 - (b + 1) ** 2 + 1.0 / a) / c


def reward_reinier(dists, caught, *, T, d_INT):
    k_dist = 10.0 / T
    k_time = 5.0 / T
    D_t = _scaled_distance(dists, d_INT)
    return (-k_dist * D_t - k_time + 5.0 * caught).astype(np.float32)


def reward_simple(dists, caught, *, T, bound, capture_penalty):
    capture_reward = -capture_penalty
    return np.where(caught, capture_reward, -dists / (T * bound)).astype(np.float32)


def reward_effective_gain(
    dists, prev_dists, ev_step, caught, *, init_radius, capture_penalty
):
    progress = prev_dists - dists
    effective_gain = progress - ev_step
    reward = effective_gain  # ! FIX LATER THIS NEEDS TO BE RESET TO THE INITIAL DISTANCE BETWEEN THE PURSUER AND EVADER
    reward[caught] += -capture_penalty
    return reward.astype(np.float32)


def reward_distance_gain(dists, prev_dists, caught, *, env_bound, capture_penalty):
    progress = prev_dists - dists
    reward = progress / env_bound
    reward[caught] += -capture_penalty
    return reward.astype(np.float32)


def _parse_reward_type(rt_raw: str):
    rt = rt_raw.strip().lower()
    tokens = []
    for part in rt.replace("|", "+").split("+"):
        part = part.strip()
        if part:
            tokens.append(part)
    base = tokens[0]
    flags = set(tokens[1:])
    return base, flags


def _apply_common_terms(
    base,
    *,
    action_delta=None,
    flags=frozenset(),
    rates_norm=None,
    rate_penalty_coef=0.0,
    smoothing_gamma=0.0,
    action_dim=None,
    time_limit=None,
):
    # smoothing if "+smooth" present
    if "smooth" in flags:
        if action_delta is None or action_dim is None or time_limit is None:
            raise ValueError(
                "'+smooth' requires action_delta, action_dim, and time_limit."
            )

        # Max possible per-step ||Δa|| when actions are in [-1,1]^d is 2 * sqrt(d)
        max_delta = 2.0 * np.sqrt(float(action_dim))
        delta_norm01 = np.clip(action_delta / max_delta, 0.0, 1.0).astype(np.float32)

        # Bound episode penalty by smoothing_gamma
        per_step_weight = float(smoothing_gamma) / float(time_limit)
        base = base - per_step_weight * delta_norm01

    # rate penalty unless "+no_rate"
    if "no_rate" not in flags and rates_norm is not None and rate_penalty_coef:
        base = base - float(rate_penalty_coef) * rates_norm

    return base.astype(np.float32)


# --------------------------------------------------------------------- #
#  Unified API (now depends on the active env config)
# --------------------------------------------------------------------- #
def get_reward(
    dists,
    caught,
    reward_type,
    prev_dists=None,
    ev_step=None,
    action_delta=None,
    rates_norm=None,
    *,
    cfg: dict,
    action_dim: int,
):
    """
    Modifiers (append to reward_type):
      +smooth   -> subtract bounded smoothing penalty using cfg['SMOOTHING_GAMMA']
      +no_rate  -> disable subtraction of cfg['RATE_PENALTY'] * ||rates||

    Examples:
      'effective_gain+smooth'
      'reinier+no_rate'
      'reward_distance_gain+smooth+no_rate'
    """
    if cfg is None:
        raise ValueError("get_reward now requires cfg=<active config> and action_dim.")

    base_name, flags = _parse_reward_type(reward_type)

    # pull needed scalars from cfg
    T = float(cfg["TIME_LIMIT"])
    d_INT = float(cfg["CAPTURE_RADIUS"])
    env_bound = 5.0
    capture_penalty = float(cfg["CAPTURE_PENALTY"])
    init_radius = float(cfg["PURSUER"]["INIT_RADIUS"])
    rate_penalty_coef = float(cfg["RATE_PENALTY"])
    smoothing_gamma = float(cfg["SMOOTHING_GAMMA"])

    # --- base rewards ---
    if base_name == "reinier":
        base = reward_reinier(dists, caught, T=T, d_INT=d_INT)

    elif base_name == "reward_distance_gain":
        base = reward_distance_gain(
            dists,
            prev_dists,
            caught,
            env_bound=env_bound,
            capture_penalty=capture_penalty,
        )

    elif base_name == "effective_gain":
        if prev_dists is None or ev_step is None:
            raise ValueError(
                "prev_dists and ev_step must be provided for 'effective_gain'"
            )
        base = reward_effective_gain(
            dists,
            prev_dists,
            ev_step,
            caught,
            init_radius=init_radius,
            capture_penalty=capture_penalty,
        )

    elif base_name == "simple":
        base = reward_simple(
            dists, caught, T=T, bound=env_bound, capture_penalty=capture_penalty
        )

    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    # --- optional terms ---
    return _apply_common_terms(
        base,
        action_delta=action_delta,
        flags=flags,
        rates_norm=rates_norm,
        rate_penalty_coef=rate_penalty_coef,
        smoothing_gamma=smoothing_gamma,
        action_dim=action_dim,
        time_limit=T,
    )
