"""
Train motor-pursuer PPO agents on multiple observation spaces.

Each run writes into:
    observation_testing/
        - models/<obs_mode>/{best_model.zip, ...}
        - logs/<obs_mode>/...
"""

import copy

from src.models.evaders.vec_moth_evader import VectorizedMothEvader
from src.models.pursuers.vec_pursuer_Motor import VecMotorPursuer
from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger, coerce_level

from scripts.training.utils.training_utils import train_one_config, TrainingConfig, make_pursuit_env_factory


def train_one(
    obs_mode: str,
    base_cfg: dict,
    total_ts: int = 25_000_000,
    n_envs: int = 100,
    seed: int = 0,
):
    """Train a single observation mode."""
    cfg = copy.deepcopy(base_cfg)
    cfg["OBSERVATIONS"]["OBS_MODE"] = obs_mode

    env_factory = make_pursuit_env_factory(
        pursuer_cls=VecMotorPursuer,
        evader_cls=VectorizedMothEvader,
        action_dim=4,
    )

    training_config = TrainingConfig(
        total_timesteps=total_ts,
        n_envs=n_envs,
        seed=seed,
        eval_freq=max(250_000 // n_envs, 1),
        checkpoint_freq=1_000_000,
        early_stopping={"max_no_improvement_evals": 40, "min_evals": 5},
    )

    train_one_config(
        env_factory=env_factory,
        train_cfg=cfg,
        output_tag=obs_mode,
        root_output_dir="observation_testing",
        training_config=training_config,
        logger_name="training.obs_space",
    )


# MAIN
# ----------------------------------------------------------------------------
# OBS_MODES = [
#     "rel_pos",
#     "rel_pos_body",
#     "pos",
#     "pos+vel",
#     "rel_pos+vel",
#     "rel_pos_vel_los_rate",
#     "rel_pos+vel_body",
#     "rel_pos_vel_los_rate_body",
#     "all",
#     "all_no_phi_rate",
# ]

OBS_MODES = ["rel_pos+vel_body", "all_body_no_phi_rate"]
SEED = 0

if __name__ == "__main__":
    setup_logging(coerce_level(CONFIG.get("LOG_LEVEL", "INFO")), name="training.obs_space")
    log = get_logger("training.obs_space")
    log.info("Start observation_space training sweep: modes=%s", OBS_MODES)

    for obs_mode in OBS_MODES:
        train_one(obs_mode, CONFIG, seed=SEED)

    print("\nAll observation modes trained!\n")
    print("To visualise:  tensorboard --logdir observation_testing/logs\n")
