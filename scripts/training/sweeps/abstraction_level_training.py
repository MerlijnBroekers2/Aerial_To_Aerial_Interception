"""
Train PPO pursuer agents at different control‑abstraction levels
Pulls model parameters from the main CONFIG FILE
"""

from __future__ import annotations


from src.models.pursuers.vec_pursuer_Acc_1order import VecPursuer_Acc_1order
from src.models.pursuers.vec_pursuer_Motor import VecMotorPursuer
from src.models.pursuers.vec_pursuer_Acc_INDI_INDI import VecAcc_INDI_INDI_Pursuer
from src.models.pursuers.vec_pursuer_CTBR_INDI import VecCTBR_INDI_Pursuer
from src.models.evaders.vec_moth_evader import VectorizedMothEvader
from src.utils.config import CONFIG as BASE_CONFIG
from src.utils.logger import setup_logging, get_logger, coerce_level

from scripts.training.utils.training_utils import train_one_config, TrainingConfig, make_pursuit_env_factory, build_level_config

TOTAL_TIMESTEPS = 50_000_000  # per abstraction level
N_ENVS = 100  # parallel environments
SEED = 0  # RNG seed
LOGGER_NAME = "training.abstraction_levels"

LEVELS = {
    "motor": {
        "pursuer_cls": VecMotorPursuer,
        "action_dim": 4,
        "model_name": "motor",
        "opt_feats": ["attitude_mat", "rates", "omega_norm"],
    },
    "acc": {
        "pursuer_cls": VecAcc_INDI_INDI_Pursuer,
        "action_dim": 3,
        "model_name": "acc_indi",
        "opt_feats": [],
    },
    "ctbr": {
        "pursuer_cls": VecCTBR_INDI_Pursuer,
        "action_dim": 4,
        "model_name": "ctbr",
        "opt_feats": ["attitude_mat", "rates", "T_force"],
    },
    "acc_1order": {
        "pursuer_cls": VecPursuer_Acc_1order,
        "action_dim": 3,
        "model_name": "acc_1order",
        "opt_feats": [],
    },
}


def train_level(level_key: str):
    """Train a single abstraction level."""
    lv = LEVELS[level_key]
    cfg = build_level_config(BASE_CONFIG, lv)
    
    env_factory = make_pursuit_env_factory(
        pursuer_cls=lv["pursuer_cls"],
        evader_cls=VectorizedMothEvader,
        action_dim=lv["action_dim"],
    )
    
    training_config = TrainingConfig(
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS,
        seed=SEED,
        eval_freq=max(250_000 // N_ENVS, 1),
        checkpoint_freq=1_000_000 // N_ENVS,
    )
    
    train_one_config(
        env_factory=env_factory,
        train_cfg=cfg,
        output_tag=level_key,
        root_output_dir="abstraction_level_testing",
        training_config=training_config,
        logger_name=LOGGER_NAME,
    )
    print(f"[ {level_key.upper()} ] Finished")


if __name__ == "__main__":
    setup_logging(
        coerce_level(BASE_CONFIG.get("LOG_LEVEL", "INFO")),
        name=LOGGER_NAME,
    )
    log = get_logger(LOGGER_NAME)
    log.info(
        "Start abstraction_level training: TOTAL_TIMESTEPS=%d, N_ENVS=%d",
        TOTAL_TIMESTEPS,
        N_ENVS,
    )
    for lvl in ["acc", "ctbr", "motor"]:
        train_level(lvl)

    print("\nAll abstraction levels trained\n")
