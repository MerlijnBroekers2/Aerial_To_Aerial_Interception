import os
from datetime import datetime
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import VecMonitor

from scripts.training.single_env import CheckpointCallback
from src.models.pursuers.vec_pursuer_CTBR_INDI import VecCTBR_INDI_Pursuer
from src.simulation.env_evasion_pursuit import EvaderVsPursuerVecEnv
from src.models.evaders.vec_reactive_evader import VectorizedReactiveEvader
from src.models.pursuers.vec_pursuer_Motor import VecMotorPursuer
from src.models.pursuers.pursuer_action_provider import make_rl_pursuer_action_provider
from src.utils.config import CONFIG
from stable_baselines3.common.callbacks import BaseCallback
from src.utils.logger import setup_logging, get_logger, coerce_level


def resolve_pursuer_cls_and_action_dim(cfg):
    model = str(cfg["PURSUER"]["MODEL"]).lower()
    if model in ("motor", "motor_rl"):
        return VecMotorPursuer, 4
    if model in ("ctbr", "ctbr_indi", "ctbr-indi"):
        return VecCTBR_INDI_Pursuer, 4
    raise ValueError(f"Unknown PURSUER.MODEL or model not suported='{model}'")


class RolloutTerminationStatsCallback(BaseCallback):
    """
    Logs per-rollout termination breakdown during TRAINING:
      rollout/percent_caught, rollout/percent_timed_out, rollout/percent_other
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.verbose = verbose
        self._reset()

    def _reset(self):
        self.total = 0
        self.caught = 0
        self.timed = 0
        self.oob = 0
        self.other = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.total += 1
            reason = info.get("term_reason", "other")
            if reason == "caught":
                self.caught += 1
            elif reason == "timed_out":
                self.timed += 1
            elif reason == "out_of_bounds":
                self.oob += 1
            else:
                self.other += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.total > 0:
            self.logger.record(
                "rollout/percent_caught", 100.0 * self.caught / self.total
            )
            self.logger.record(
                "rollout/percent_timed_out", 100.0 * self.timed / self.total
            )
            self.logger.record("rollout/percent_other", 100.0 * self.other / self.total)
            self.logger.record("rollout/oob", 100.0 * self.oob / self.total)

            if self.verbose:
                print(
                    f"[Train] episodes={self.total} | "
                    f"caught={self.caught/self.total:6.2%} | "
                    f"timed={self.timed/self.total:6.2%} | "
                    f"oob={self.oob/self.total:6.2%} | "
                    f"other={self.other/self.total:6.2%}"
                )
        self._reset()


def train_evader_reactive_and_save(
    config,
    total_timesteps=25_000_000,
    train_envs=100,
    n_steps=1000,
    batch_size=5000,
    resume_from_path=None,
):
    log = get_logger("training.reactive_evader")
    obs = config.get("OBSERVATIONS", {})
    log.info(
        "OBS config: mode=%s, action_dim=%s, hist=%s(h=%s), act_hist=%s(h=%s), opt=%s",
        obs.get("OBS_MODE"),
        obs.get("ACTION_DIM"),
        obs.get("INCLUDE_HISTORY"),
        obs.get("HISTORY_STEPS"),
        obs.get("INCLUDE_ACTION_HISTORY"),
        obs.get("ACTION_HISTORY_STEPS"),
        obs.get("OPTIONAL_FEATURES"),
    )
    # --- dirs & logging names -------------------------------------------------
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"EvaderReactive_MotorVsRLPursuer_{timestamp}"
    save_dir = os.path.join("final_models_evader", run_name)
    log_dir = os.path.join("logs_evader", run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- pursuer action providers (separate histories for train/eval) --------
    act_pu_train = make_rl_pursuer_action_provider(config, train_envs)
    act_pu_eval = make_rl_pursuer_action_provider(config, 1)

    pursuer_cls, _ = resolve_pursuer_cls_and_action_dim(config)

    # --- env factories (same API shape as your current trainer) --------------
    TrainEnvFactory = partial(
        EvaderVsPursuerVecEnv,
        evader_cls=VectorizedReactiveEvader,
        pursuer_cls=pursuer_cls,
        action_provider=act_pu_train,
        evader_action_dim=3,
    )
    EvalEnvFactory = partial(
        EvaderVsPursuerVecEnv,
        evader_cls=VectorizedReactiveEvader,
        pursuer_cls=pursuer_cls,
        action_provider=act_pu_eval,
        evader_action_dim=3,
    )

    train_env = VecMonitor(TrainEnvFactory(num_envs=train_envs, config=config))
    eval_env = VecMonitor(EvalEnvFactory(num_envs=1, config=config))

    # --- model (same hyperparams you use elsewhere) --------------------------
    policy_kwargs = config["POLICY_KWARGS"]
    if resume_from_path:
        print(f"Resuming from: {resume_from_path}")
        model = PPO.load(resume_from_path, env=train_env, tensorboard_log=log_dir)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            tensorboard_log=log_dir,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.999,
            policy_kwargs=policy_kwargs,
        )

    # --- callbacks ------------------------------------------------------------
    desired_eval_freq = 10_000_000
    eval_freq = max(desired_eval_freq // train_envs, 1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=1_000_000, save_path=save_dir, prefix="ppo_checkpoint", verbose=1
    )

    print(f"[EvaderReactive] Training PPO for {total_timesteps:,} steps → {save_dir}")
    rollout_stats_cb = RolloutTerminationStatsCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            eval_cb,
            checkpoint_cb,
            ProgressBarCallback(),
            rollout_stats_cb,
        ],
    )

    final_path = os.path.join(save_dir, "ppo_final_evader_reactive")
    model.save(final_path)
    print(f"[EvaderReactive] Final model saved to {final_path}")


if __name__ == "__main__":
    setup_logging(coerce_level(CONFIG.get("LOG_LEVEL", "INFO")), name="training.reactive_evader")
    log = get_logger("training.reactive_evader")
    log.info(
        "Start reactive_evader_training: pursuer=%s, DT=%.4f",
        CONFIG["PURSUER"].get("MODEL"),
        CONFIG.get("DT"),
    )
    train_evader_reactive_and_save(
        CONFIG,
        total_timesteps=25_000_000,  # same as your pursuer runs
        train_envs=100,  # same
        n_steps=1000,  # same
        batch_size=5000,  # same
        resume_from_path=None,
    )
