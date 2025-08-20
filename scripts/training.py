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
from src.evaders.vec_pliska_evader import VectorizedPliskaEvader
from src.simulation.env_pursuit_evasion import PursuitVecEnv
from src.evaders.vec_moth_evader import VectorizedMothEvader
from src.pursuers.vec_pursuer_Motor import VecMotorPursuer
from src.utils.config import CONFIG

from src.utils.observations import save_observation_layout_csv
from src.utils.run_prefix import build_run_prefix


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

    run_prefix = build_run_prefix(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join("final_models", f"{run_prefix}_{timestamp}")
    log_dir = os.path.join("logs", f"{run_prefix}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_observation_layout_csv(
        config, os.path.join(save_dir, "observation_layout.csv")
    )

    train_env = VecMonitor(env_class(num_envs=train_envs, config=config))
    eval_env = VecMonitor(env_class(num_envs=1, config=config))

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

    desired_eval_freq = 25_000_000  # logical timestep granularity
    adjusted_eval_freq = max(desired_eval_freq // train_envs, 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=adjusted_eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000_000,
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
        ],
    )

    final_path = os.path.join(save_dir, f"ppo_final_{run_prefix}")
    model.save(final_path)
    print(f"Final model saved to {final_path}")


MotorVsMothEnv = partial(
    PursuitVecEnv,
    evader_cls=VectorizedMothEvader,
    pursuer_cls=VecMotorPursuer,
    action_dim=4,
)

MotorVsPliskaEnv = partial(
    PursuitVecEnv,
    evader_cls=VectorizedPliskaEvader,
    pursuer_cls=VecMotorPursuer,
    action_dim=4,
)

if CONFIG["EVADER"]["MODEL"] == "moth":
    train_and_save(
        env_class=MotorVsMothEnv,
        config=CONFIG,
        resume_from_path=None,
    )
if CONFIG["EVADER"]["MODEL"] == "pliska":
    train_and_save(
        env_class=MotorVsPliskaEnv,
        config=CONFIG,
        resume_from_path=None,
    )
else:
    assert f"model type not compatible for training"
