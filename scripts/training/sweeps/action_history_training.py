"""
Train pursuer agents with varying action history lengths.
"""

import copy

from src.models.pursuers.vec_pursuer_CTBR_INDI import VecCTBR_INDI_Pursuer
from src.models.evaders.vec_moth_evader import VectorizedMothEvader
from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger, coerce_level

from scripts.training.utils.training_utils import train_one_config, TrainingConfig, make_pursuit_env_factory


def train_one(
    history_steps: int,
    base_cfg: dict,
    total_ts: int = 50_000_000,
    n_envs: int = 100,
    seed: int = 0,
):
    """Train with a specific action history length."""
    cfg = copy.deepcopy(base_cfg)
    cfg["OBSERVATIONS"]["OBS_MODE"] = "rel_pos+vel_body"
    cfg["OBSERVATIONS"]["INCLUDE_ACTION_HISTORY"] = True
    cfg["OBSERVATIONS"]["ACTION_HISTORY_STEPS"] = history_steps

    env_factory = make_pursuit_env_factory(
        pursuer_cls=VecCTBR_INDI_Pursuer,
        evader_cls=VectorizedMothEvader,
        action_dim=4,
    )

    training_config = TrainingConfig(
        total_timesteps=total_ts,
        n_envs=n_envs,
        seed=seed,
        eval_freq=max(250_000 // n_envs, 1),
        checkpoint_freq=1_000_000,
        early_stopping={"max_no_improvement_evals": 10, "min_evals": 5},
        policy_kwargs=CONFIG["POLICY_KWARGS"],
    )

    tag = f"history_{history_steps}"
    
    train_one_config(
        env_factory=env_factory,
        train_cfg=cfg,
        output_tag=tag,
        root_output_dir="action_history_testing_ctbr",
        training_config=training_config,
        logger_name="training.action_history",
    )


# MAIN
# ----------------------------------------------------------------------------
HISTORY_VALUES = [1, 2, 4, 8, 16]
SEED = 0

if __name__ == "__main__":
    setup_logging(
        coerce_level(CONFIG.get("LOG_LEVEL", "INFO")), name="training.action_history"
    )
    log = get_logger("training.action_history")
    log.info("Start action_history sweep: values=%s", HISTORY_VALUES)

    for history_steps in HISTORY_VALUES:
        train_one(history_steps, CONFIG, seed=SEED)

    print("\nAll action history values trained!\n")
    print("To visualise:  tensorboard --logdir action_history_testing_ctbr/logs\n")
