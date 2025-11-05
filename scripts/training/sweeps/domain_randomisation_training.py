"""
Train PPO pursuer agents across abstraction levels and multiple domain-randomization (DR) levels.

Usage:
    python domain_randomisation_training.py

Notes:
- DR is applied ONLY for training by setting PURSUER.domain_randomization_pct[*] = DR% / 100.
- Evaluation (in EvalCallback) uses the SAME config (thus same DR settings) for on-the-fly selection;
  if you want eval without DR, set a separate 'eval_cfg' inside train_level() with DR = 0.0.
- Artifacts are organized under: abstraction_level_dr/<level>/dr<XX>/{models,logs}.
"""

from __future__ import annotations

import copy

from src.models.pursuers.vec_pursuer_Acc_1order import VecPursuer_Acc_1order
from src.models.pursuers.vec_pursuer_Motor import VecMotorPursuer
from src.models.pursuers.vec_pursuer_Acc_INDI_INDI import VecAcc_INDI_INDI_Pursuer
from src.models.pursuers.vec_pursuer_CTBR_INDI import VecCTBR_INDI_Pursuer
from src.models.evaders.vec_moth_evader import VectorizedMothEvader
from src.utils.config import CONFIG as BASE_CONFIG
from src.utils.logger import setup_logging, get_logger, coerce_level

from scripts.training.utils.training_utils import train_one_config, TrainingConfig, make_pursuit_env_factory, build_level_config, apply_dr_percent, log_observation_config

# EDIT THESE CONSTANTS
TOTAL_TIMESTEPS = 25_000_000  # per (level, DR) run
N_ENVS = 100  # parallel environments
SEED = 0
DR_LEVELS = [00, 10, 20, 30]  # percent; change as you like
ROOT_OUT = "abstraction_level_dr"  # output root folder
# ----------------------------------------------------------------------------

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
        "opt_feats": ["attitude_mat", "rates", "T_force"],
    },
    "ctbr": {
        "pursuer_cls": VecCTBR_INDI_Pursuer,
        "action_dim": 4,
        "model_name": "ctbr_indi",
        "opt_feats": ["attitude_mat", "rates", "T_force"],
    },
}


def train_level_with_dr(level_key: str, dr_percent: int):
    """Train a single (level, DR) combination."""
    lv = LEVELS[level_key]
    
    # Base config for this level (no DR yet)
    base_cfg = build_level_config(BASE_CONFIG, lv)
    
    # Training config with DR
    train_cfg = apply_dr_percent(base_cfg, dr_percent)
    
    # For now we evaluate under the same DR as training for consistency:
    # OPTIONAL: If you prefer evaluation *without* DR, uncomment:
    # eval_cfg = apply_dr_percent(base_cfg, 0)
    eval_cfg = copy.deepcopy(train_cfg)
    
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
        early_stopping={"max_no_improvement_evals": 40, "min_evals": 5},
    )
    
    tag = f"{level_key}/dr{dr_percent:02d}"
    
    log = get_logger("training.dr_sweep")
    log_observation_config(train_cfg, "training.dr_sweep", level=level_key, DR=f"{dr_percent}%")
    
    print(f"\n[ {level_key.upper()} | DR={dr_percent}% ] Training for {TOTAL_TIMESTEPS:,} timesteps …")
    
    train_one_config(
        env_factory=env_factory,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        output_tag=tag,
        root_output_dir=ROOT_OUT,
        training_config=training_config,
        logger_name="training.dr_sweep",
    )
    
    print(f"[ {level_key.upper()} | DR={dr_percent}% ] Finished")


if __name__ == "__main__":
    setup_logging(
        coerce_level(BASE_CONFIG.get("LOG_LEVEL", "INFO")), name="training.dr_sweep"
    )
    log = get_logger("training.dr_sweep")
    log.info(
        "Start DR sweep: TOTAL_TIMESTEPS=%d, N_ENVS=%d, DR_LEVELS=%s",
        TOTAL_TIMESTEPS,
        N_ENVS,
        DR_LEVELS,
    )
    
    level_order = ("motor", "ctbr", "acc")
    
    for lvl in level_order:
        for dr in DR_LEVELS:
            train_level_with_dr(lvl, dr)
    
    print("\nAll abstraction levels × DR levels trained!\n")
