import numpy as np
from stable_baselines3 import PPO


def make_rl_evader_action_provider(
    config: dict,
    num_envs: int,
    model_path: str | None = None,
    deterministic: bool = True,
):
    """
    Returns a callable: act(ev_state, pu_state) -> (N,3) in [-1,1].
    Obs layout matches your evader policy: [Δp, Δv, p_e]. NOT MODULAR DESIGN RN CHANGE LATER

    You will pass its output directly into VectorizedReactiveEvader.step(...),
    which internally scales to [-MAX_ACCEL, MAX_ACCEL].
    """
    if model_path is None:
        model_path = config["EVADER"]["RL_MODEL_PATH"]

    model = PPO.load(model_path)
    print(f"[make_rl_evader_action_provider] loaded: {model_path}  (N={num_envs})")

    def act(ev_state: dict, pu_state: dict) -> np.ndarray:
        dp = pu_state["true_position"] - ev_state["true_position"]
        dv = pu_state["velocity"] - ev_state["velocity"]
        obs = np.concatenate([dp, dv, ev_state["true_position"]], axis=1).astype(
            np.float32
        )  # (N,9)

        actions, _ = model.predict(obs, deterministic=deterministic)
        actions = np.asarray(actions, dtype=np.float32).reshape(num_envs, 3)
        return np.clip(actions, -1.0, 1.0)  # in [-1,1]

    return act
