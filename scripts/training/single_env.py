import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    ProgressBarCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import VecMonitor

from functools import partial
from src.models.evaders.vec_pliska_evader import VectorizedPliskaEvader
from src.models.pursuers.vec_pursuer_Acc_1order import VecPursuer_Acc_1order
from src.models.pursuers.vec_pursuer_Acc_INDI_INDI import VecAcc_INDI_INDI_Pursuer
from src.models.pursuers.vec_pursuer_CTBR_INDI import VecCTBR_INDI_Pursuer
from src.simulation.env_pursuit_evasion import PursuitVecEnv
from src.models.evaders.vec_moth_evader import VectorizedMothEvader
from src.models.pursuers.vec_pursuer_Motor import VecMotorPursuer
from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger, coerce_level

from src.utils.observations import save_observation_layout_csv
from src.utils.run_prefix import build_run_prefix


def resolve_pursuer_cls_and_action_dim(cfg):
    model = str(cfg["PURSUER"]["MODEL"]).lower()
    if model in ("motor", "motor_rl"):
        return VecMotorPursuer, 4
    if model in ("ctbr", "ctbr_indi", "ctbr-indi"):
        return VecCTBR_INDI_Pursuer, 4
    if model in ("acc_indi_indi", "acc_indi", "acc-indi-indi", "indi"):
        return VecAcc_INDI_INDI_Pursuer, 3
    if model in ("acc_1order", "firstorder", "1order"):
        return VecPursuer_Acc_1order, 3
    raise ValueError(f"Unknown PURSUER.MODEL='{model}'")


class WaitRewardStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._reset()

    def _reset(self):
        self.n_eps = 0
        self.sum_wait = 0.0
        self.sum_wait_per_step = 0.0
        self.n_eps_with_len = 0  # <- new

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for done, info in zip(dones, infos):
            if not done:
                continue
            w = float(info.get("wait_reward_sum", 0.0))
            if (w != 0.0) or ("wait_reward_sum" in info):
                self.n_eps += 1
                self.sum_wait += w
                ep_len = None
                if isinstance(info.get("episode"), dict):
                    ep_len = info["episode"].get("l", None)
                if ep_len and ep_len > 0:
                    self.sum_wait_per_step += w / ep_len
                    self.n_eps_with_len += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.n_eps > 0:
            self.logger.record(
                "rollout/wait_reward_sum_mean", self.sum_wait / self.n_eps
            )
        if self.n_eps_with_len > 0:
            self.logger.record(
                "rollout/wait_reward_per_step_mean",
                self.sum_wait_per_step / self.n_eps_with_len,
            )
        self._reset()


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, prefix="", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.prefix = prefix
        self.last_save = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_save) >= self.save_freq:
            model_filename = os.path.join(
                self.save_path, f"{self.prefix}_step_{self.num_timesteps}.zip"
            )
            self.model.save(model_filename)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved model at step {self.num_timesteps}")
            self.last_save = self.num_timesteps
        return True


class ActionSmoothingStatsCallback(BaseCallback):
    """
    Reads 'action_l1_sum' and 'action_tv_sum' that your env puts into infos
    on terminal steps, aggregates them over the rollout, and logs to TB.
    Also logs per-step (normalized by episode length when available).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.verbose = verbose
        self._reset()

    def _reset(self):
        self.n_eps = 0
        self.sum_l1 = 0.0
        self.sum_tv = 0.0
        self.sum_l1_per_step = 0.0
        self.sum_tv_per_step = 0.0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for done, info in zip(dones, infos):
            if not done:
                continue

            l1 = float(info.get("action_l1_sum", 0.0))
            tv = float(info.get("action_tv_sum", 0.0))

            if (
                (l1 != 0.0)
                or (tv != 0.0)
                or ("action_l1_sum" in info)
                or ("action_tv_sum" in info)
            ):
                self.n_eps += 1
                self.sum_l1 += l1
                self.sum_tv += tv

                # VecMonitor injects 'episode': {'r': ..., 'l': ..., 't': ...}
                ep_len = None
                if isinstance(info.get("episode"), dict):
                    ep_len = info["episode"].get("l", None)

                if ep_len and ep_len > 0:
                    self.sum_l1_per_step += l1 / ep_len
                    self.sum_tv_per_step += tv / ep_len
        return True

    def _on_rollout_end(self) -> None:
        if self.n_eps > 0:
            self.logger.record("rollout/action_l1_sum_mean", self.sum_l1 / self.n_eps)
            self.logger.record("rollout/action_tv_sum_mean", self.sum_tv / self.n_eps)
            self.logger.record(
                "rollout/action_l1_per_step_mean", self.sum_l1_per_step / self.n_eps
            )
            self.logger.record(
                "rollout/action_tv_per_step_mean", self.sum_tv_per_step / self.n_eps
            )

            if self.verbose:
                print(
                    f"[Rollout] n_eps={self.n_eps} | "
                    f"L1_sum_mean={self.sum_l1/self.n_eps:.3f} | "
                    f"TV_sum_mean={self.sum_tv/self.n_eps:.3f} | "
                    f"L1/step_mean={self.sum_l1_per_step/self.n_eps:.6f} | "
                    f"TV/step_mean={self.sum_tv_per_step/self.n_eps:.6f}"
                )
        self._reset()


class EpisodeTerminationStatsCallback(BaseCallback):
    """
    Tracks termination causes (capture, timeout, OOB position, OOB spiral),
    and logs per-rollout episode breakdown.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.verbose = verbose
        self.reset_counters()

    def reset_counters(self):
        self.total_episodes = 0
        self.n_caught = 0
        self.n_timeout = 0
        self.n_oob_pos = 0
        self.n_oob_spiral = 0
        self.n_other = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for done, info in zip(dones, infos):
            if done:
                self.total_episodes += 1
                reason = info.get("term_reason", "other")
                match reason:
                    case "caught":
                        self.n_caught += 1
                    case "timed_out":
                        self.n_timeout += 1
                    case "out_of_bounds_pos":
                        self.n_oob_pos += 1
                    case "out_of_control_spiral":
                        self.n_oob_spiral += 1
                    case _:
                        self.n_other += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.total_episodes:
            self.logger.record(
                "ep/percent_caught", self.n_caught / self.total_episodes * 100
            )
            self.logger.record(
                "ep/percent_timed_out", self.n_timeout / self.total_episodes * 100
            )
            self.logger.record(
                "ep/percent_oob_pos", self.n_oob_pos / self.total_episodes * 100
            )
            self.logger.record(
                "ep/percent_oob_spiral", self.n_oob_spiral / self.total_episodes * 100
            )
            self.logger.record(
                "ep/percent_other", self.n_other / self.total_episodes * 100
            )

            if self.verbose:
                print(
                    f"[Rollout] episodes={self.total_episodes:5d} | "
                    f"caught={self.n_caught/self.total_episodes:6.2%} | "
                    f"timeout={self.n_timeout/self.total_episodes:6.2%} | "
                    f"OOB-pos={self.n_oob_pos/self.total_episodes:6.2%} | "
                    f"OOB-spiral={self.n_oob_spiral/self.total_episodes:6.2%}"
                )

        self.reset_counters()


def train_and_save(
    env_class,
    config,
    total_timesteps=100_000_000,
    train_envs=100,
    n_steps=1000,
    batch_size=5000,
    resume_from_path=None,
):
    log = get_logger("training.single_env")
    obs = config.get("OBSERVATIONS", {})
    log.info(
        "OBS config: mode=%s, action_dim=%s, include_hist=%s(h=%s), include_act_hist=%s(h=%s), opt_feats=%s",
        obs.get("OBS_MODE"),
        obs.get("ACTION_DIM"),
        obs.get("INCLUDE_HISTORY"),
        obs.get("HISTORY_STEPS"),
        obs.get("INCLUDE_ACTION_HISTORY"),
        obs.get("ACTION_HISTORY_STEPS"),
        obs.get("OPTIONAL_FEATURES"),
    )
    if "reward_type" in config or "SMOOTHING_GAMMA" in config:
        log.info(
            "Reward: type=%s, smoothing_gamma=%s",
            config.get("reward_type"),
            config.get("SMOOTHING_GAMMA"),
        )
    run_prefix = build_run_prefix(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join("final_models", f"{run_prefix}_{timestamp}")
    log_dir = os.path.join("logs", f"{run_prefix}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_observation_layout_csv(
        config, os.path.join(save_dir, "observation_layout.csv")
    )

    train_env = VecMonitor(
        env_class(num_envs=train_envs, config=config),
        info_keywords=("action_l1_sum", "action_tv_sum", "wait_reward_sum"),
    )
    eval_env = VecMonitor(
        env_class(num_envs=1, config=config),
        info_keywords=("action_l1_sum", "action_tv_sum"),
    )

    policy_kwargs = config["POLICY_KWARGS"]

    if resume_from_path is not None:
        print(f"Resuming training from: {resume_from_path}")
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

    desired_eval_freq = 1_000_000  # logical timestep granularity
    adjusted_eval_freq = max(desired_eval_freq // train_envs, 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=adjusted_eval_freq,
        n_eval_episodes=1,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path=save_dir,
        prefix="ppo_checkpoint",
        verbose=1,
    )

    print(f"Training PPO for {total_timesteps:,} steps. Saving to: {save_dir}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            eval_callback,
            checkpoint_callback,
            ProgressBarCallback(),
            EpisodeTerminationStatsCallback(),
            ActionSmoothingStatsCallback(),
            WaitRewardStatsCallback(),
        ],
    )

    final_path = os.path.join(save_dir, f"ppo_final_{run_prefix}")
    model.save(final_path)
    print(f"Final model saved to {final_path}")


pursuer_cls, action_dim = resolve_pursuer_cls_and_action_dim(CONFIG)

MothEnv = partial(
    PursuitVecEnv,
    evader_cls=VectorizedMothEvader,
    pursuer_cls=pursuer_cls,
    action_dim=action_dim,
)
PliskaEnv = partial(
    PursuitVecEnv,
    evader_cls=VectorizedPliskaEvader,
    pursuer_cls=pursuer_cls,
    action_dim=action_dim,
)

if __name__ == "__main__":
    setup_logging(
        coerce_level(CONFIG.get("LOG_LEVEL", "INFO")), name="training.single_env"
    )
    log = get_logger("training.single_env")
    log.info(
        "Start single_env: pursuer=%s, evader=%s, DT=%.4f",
        CONFIG["PURSUER"].get("MODEL"),
        CONFIG["EVADER"].get("MODEL"),
        CONFIG.get("DT"),
    )
    train_and_save(
        env_class=MothEnv,
        config=CONFIG,
        resume_from_path=None,
    )
