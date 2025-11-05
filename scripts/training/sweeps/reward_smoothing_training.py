"""
Sweep reward smoothing for CTBR and Motor pursuers with the 'effective_gain' reward.

Combos:
- No smoothing, no rate penalty
- No smoothing, with rate penalty
- 0.1 smoothing, no rate penalty
- 0.2 smoothing, no rate penalty
- 0.4 smoothing, no rate penalty
- 1.0 smoothing, no rate penalty
- 2.0 smoothing, no rate penalty

This relies on your reward API:
  reward_type modifiers: "+smooth" and "+no_rate"
  SMOOTHING_GAMMA controls the smoothing strength.
"""

from __future__ import annotations
import copy

from src.models.evaders.vec_moth_evader import VectorizedMothEvader
from src.models.pursuers.vec_pursuer_Motor import VecMotorPursuer
from src.models.pursuers.vec_pursuer_CTBR_INDI import VecCTBR_INDI_Pursuer
from src.utils.config import CONFIG as BASE_CONFIG
from src.utils.logger import setup_logging, get_logger, coerce_level

from scripts.training.utils.training_utils import (
    train_one_config,
    TrainingConfig,
    make_pursuit_env_factory,
    build_level_config,
    cfg_with_reward,
    log_observation_config,
)

# EDIT THESE CONSTANTS
TOTAL_TIMESTEPS = 50_000_000
N_ENVS = 100
SEED = 0
ROOT_OUT = "trained_models/reward_smoothing_sweep_dr10"  # output root folder
# ----------------------------------------------------------------------------

LEVELS = {
    # action_dim is what the env expects from the pursuer policy
    "motor": {
        "pursuer_cls": VecMotorPursuer,
        "action_dim": 4,
        "model_name": "motor",
        "opt_feats": ["attitude_mat", "rates", "omega_norm"],
    },
    "ctbr": {
        "pursuer_cls": VecCTBR_INDI_Pursuer,
        "action_dim": 4,
        "model_name": "ctbr_indi",
        "opt_feats": ["attitude_mat", "rates", "T_force"],
    },
}

VARIANTS = [
    ("nosmooth_rate", "effective_gain", 0.0),
    ("nosmooth_no-rate", "effective_gain+no_rate", 0.0),
    ("smooth1p0_no-rate", "effective_gain+smooth+no_rate", 1.0),
    ("smooth2p0_no-rate", "effective_gain+smooth+no_rate", 2.0),
    ("smooth5p0_no-rate", "effective_gain+smooth+no_rate", 5.0),
    ("smooth10p0_no-rate", "effective_gain+smooth+no_rate", 10.0),
    ("smooth20p0_no-rate", "effective_gain+smooth+no_rate", 20.0),
    ("smooth30p0_no-rate", "effective_gain+smooth+no_rate", 30.0),
    ("smooth40p0_no-rate", "effective_gain+smooth+no_rate", 40.0),
    ("smooth50p0_no-rate", "effective_gain+smooth+no_rate", 50.0),
    ("smooth60p0_no-rate", "effective_gain+smooth+no_rate", 60.0),
    ("smooth70p0_no-rate", "effective_gain+smooth+no_rate", 70.0),
    ("smooth80p0_no-rate", "effective_gain+smooth+no_rate", 80.0),
    ("smooth90p0_no-rate", "effective_gain+smooth+no_rate", 90.0),
]


def train_level_variant(
    level_key: str, variant_tag: str, reward_type: str, smoothing_gamma: float
):
    """Train a single (level, reward variant) combination."""
    lv = LEVELS[level_key]
    base_cfg = build_level_config(BASE_CONFIG, lv)
    
    # IMPORTANT: actually force these off (setdefault would not override)
    base_cfg["OBSERVATIONS"]["INCLUDE_HISTORY"] = False
    base_cfg["OBSERVATIONS"]["INCLUDE_ACTION_HISTORY"] = False
    
    train_cfg = cfg_with_reward(base_cfg, reward_type, smoothing_gamma)
    eval_cfg = copy.deepcopy(train_cfg)

    log = get_logger("training.reward_smoothing")
    log_observation_config(
        train_cfg,
        "training.reward_smoothing",
        level=level_key,
        reward=train_cfg.get("reward_type"),
        smooth_gamma=train_cfg.get("SMOOTHING_GAMMA"),
    )

    # Make the output path unambiguously reflect the actual gamma
    gamma_suffix = f"g{str(smoothing_gamma).replace('.', 'p')}"
    tag = f"{level_key}/{variant_tag}_{gamma_suffix}"

    env_factory = make_pursuit_env_factory(
        pursuer_cls=lv["pursuer_cls"],
        evader_cls=VectorizedMothEvader,
        action_dim=lv["action_dim"],
    )

    training_config = TrainingConfig(
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS,
        seed=SEED,
        eval_freq=max(500_000 // N_ENVS, 1),
        n_eval_episodes=50,
        checkpoint_freq=1_000_000 // N_ENVS,
        early_stopping={"max_no_improvement_evals": 60, "min_evals": 5},
        use_action_smoothing_stats=True,
        use_episode_termination_stats=True,
        info_keywords=("action_l1_sum", "action_tv_sum"),
    )

    print(
        f"\n[ {level_key.upper()} | {variant_tag} | gamma={smoothing_gamma} ] Training for {TOTAL_TIMESTEPS:,} timesteps …"
    )
    
    train_one_config(
        env_factory=env_factory,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        output_tag=tag,
        root_output_dir=ROOT_OUT,
        training_config=training_config,
        logger_name="training.reward_smoothing",
    )
    
    print(
        f"[ {level_key.upper()} | {variant_tag} | gamma={smoothing_gamma} ] Finished"
    )


# ----------------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    setup_logging(coerce_level(BASE_CONFIG.get("LOG_LEVEL", "INFO")), name="training.reward_smoothing")
    log = get_logger("training.reward_smoothing")
    log.info("Start reward_smoothing sweep: TOTAL_TIMESTEPS=%d, N_ENVS=%d", TOTAL_TIMESTEPS, N_ENVS)
    os.makedirs(ROOT_OUT, exist_ok=True)

    level_order = ["ctbr"]  # "ctbr"  # CTBR and Motor only
    for lvl in level_order:
        for tag, rt, gamma in VARIANTS:
            train_level_variant(lvl, tag, rt, gamma)

    print("\nAll levels × smoothing variants trained!\n")
