import numpy as np
from stable_baselines3 import PPO
from src.utils.observations import get_observation, get_observation_size


def make_rl_pursuer_action_provider(config: dict, num_envs: int):
    """
    Returns a callable: act(ev_state, pu_state) -> (N, ACTION_DIM),
    producing motor commands in [-1,1] for VecMotorPursuer.step_learn().
    Uses your existing OBSERVATIONS config and a trained PPO policy.
    """
    obs_base_dim, obs_dim = get_observation_size(config)
    adim = config["OBSERVATIONS"]["ACTION_DIM"]
    max_hist = config["OBSERVATIONS"]["MAX_HISTORY_STEPS"]

    # History buffers exactly like your VecEnv does
    obs_hist = np.zeros((num_envs, max_hist, obs_base_dim), dtype=np.float32)
    act_hist = np.zeros((num_envs, max_hist, adim), dtype=np.float32)

    # Load the trained pursuer policy
    ctrl_cfg = config["PURSUER"]["CONTROLLER"]
    assert ctrl_cfg["type"] == "rl", "Expecting RL controller for the pursuer."
    model = PPO.load(ctrl_cfg["policy_path"])
    print(
        f"[make_rl_pursuer_action_provider] loaded: {ctrl_cfg['policy_path']}  (N={num_envs})"
    )

    def act(ev_state: dict, pu_state: dict) -> np.ndarray:
        # Build the observation the model was trained with
        obs_now = get_observation(
            config=config,
            ev_state=ev_state,
            pu_state=pu_state,
            obs_history=obs_hist,
            act_history=act_hist,
        )  # (N, obs_dim)

        actions, _ = model.predict(obs_now, deterministic=True)
        actions = np.asarray(actions, dtype=np.float32).reshape(num_envs, adim)
        actions = np.clip(actions, -1.0, 1.0)

        # Update histories
        core = obs_now[:, :obs_base_dim]
        obs_hist[:] = np.roll(obs_hist, shift=-1, axis=1)
        act_hist[:] = np.roll(act_hist, shift=-1, axis=1)
        obs_hist[:, -1, :] = core
        act_hist[:, -1, :] = actions
        return actions

    return act
