import os
import time
import copy
import subprocess
from functools import partial

from src.models.evaders.vec_pliska_evader import VectorizedPliskaEvader
from src.models.pursuers.vec_pursuer_Motor import VecMotorPursuer
from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger, coerce_level

from scripts.training.utils.training_utils import train_one_config, TrainingConfig, make_pursuit_env_factory


def train_one(
    obs_mode: str,
    history_steps: int,
    base_cfg: dict,
    seed=0,
    total_ts=15_000_000,
    n_envs=100,
):
    """Train with a specific observation mode and history length."""
    cfg = copy.deepcopy(base_cfg)
    cfg["OBSERVATIONS"]["OBS_MODE"] = obs_mode
    if history_steps < 1:
        cfg["OBSERVATIONS"]["INCLUDE_HISTORY"] = False
    else:
        cfg["OBSERVATIONS"]["INCLUDE_HISTORY"] = True
    cfg["OBSERVATIONS"]["HISTORY_STEPS"] = history_steps
    cfg["OBSERVATIONS"]["MAX_HISTORY_STEPS"] = history_steps + 1

    env_factory = make_pursuit_env_factory(
        pursuer_cls=VecMotorPursuer,
        evader_cls=VectorizedPliskaEvader,
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

    tag = f"{obs_mode}_h{history_steps}"
    
    train_one_config(
        env_factory=env_factory,
        train_cfg=cfg,
        output_tag=tag,
        root_output_dir="Meas_GT_obs_history_testing_sig0.02_smooth",
        training_config=training_config,
        logger_name="training.obs_history",
    )


def run_train(obs_mode, h, s):
    train_one(obs_mode, h, CONFIG, seed=s)


def launch_parallel(obs_mode: str, history_values=[0], max_parallel=1, seed=0):
    from multiprocessing import Process

    max_parallel = max_parallel or cpu_count()
    jobs = [(h, seed) for h in history_values]
    running = []

    while jobs or running:
        while jobs and len(running) < max_parallel:
            h, s = jobs.pop(0)
            print(f"[launcher] starting {obs_mode} h={h}")
            p = Process(target=run_train, args=(obs_mode, h, s))
            p.start()
            running.append((p, h))

        still_running = []
        for p, h in running:
            p.join(timeout=0.1)
            if p.is_alive():
                still_running.append((p, h))
            else:
                print(f"[launcher] {obs_mode}_h{h} finished")
        running = still_running
        time.sleep(1)

    print("\nAll runs finished.")


if __name__ == "__main__":
    setup_logging(coerce_level(CONFIG.get("LOG_LEVEL", "INFO")), name="training.obs_history")
    log = get_logger("training.obs_history")
    log.info("Start observation_history sweep: modes=%s", ["rel_pos+vel_body"])
    OBSERVATION_MODES = ["rel_pos+vel_body"]
    MAX_PARALLEL = 1
    SEED = 0

    for obs in OBSERVATION_MODES:
        launch_parallel(obs_mode=obs, max_parallel=MAX_PARALLEL, seed=SEED)
