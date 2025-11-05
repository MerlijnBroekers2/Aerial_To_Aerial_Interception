"""
Training utilities and framework for running configuration sweeps.
This module consolidates shared functionality across different training loops.
"""

import os
import copy
from typing import Callable, Dict, Any, Optional, List, Tuple
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import (
    EvalCallback,
    ProgressBarCallback,
    StopTrainingOnNoModelImprovement,
)

from src.utils.observations import save_observation_layout_csv
from src.utils.logger import get_logger


# CALLBACKS
# ----------------------------------------------------------------------------

class CheckpointEverySteps(BaseCallback):
    """Save `checkpoint_step_XXXXX.zip` every *freq* real env steps."""

    def __init__(self, freq: int, folder: str, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            fname = os.path.join(
                self.folder, f"checkpoint_step_{self.num_timesteps}.zip"
            )
            self.model.save(fname)
        return True


class CheckpointCallback(BaseCallback):
    """Save checkpoint with a prefix every *save_freq* steps."""

    def __init__(self, save_freq: int, save_path: str, prefix: str = "ppo_checkpoint", verbose: int = 0):
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
    Reads 'action_l1_sum' and 'action_tv_sum' from env infos on terminal steps,
    aggregates them over the rollout, and logs to TB.
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


class RolloutTerminationStatsCallback(BaseCallback):
    """
    Logs per-rollout termination breakdown during TRAINING.
    Simplified version that tracks caught, timed_out, and other.
    """

    def __init__(self, prefix="rollout", verbose=0):
        super().__init__(verbose)
        self.prefix = prefix
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
            elif reason == "out_of_bounds" or reason == "out_of_bounds_pos":
                self.oob += 1
            else:
                self.other += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.total > 0:
            self.logger.record(
                f"{self.prefix}/percent_caught", 100.0 * self.caught / self.total
            )
            self.logger.record(
                f"{self.prefix}/percent_timed_out", 100.0 * self.timed / self.total
            )
            self.logger.record(
                f"{self.prefix}/percent_other", 100.0 * self.other / self.total
            )
            self.logger.record(f"{self.prefix}/oob", 100.0 * self.oob / self.total)

            if self.verbose:
                print(
                    f"[Train] episodes={self.total} | "
                    f"caught={self.caught/self.total:6.2%} | "
                    f"timed={self.timed/self.total:6.2%} | "
                    f"oob={self.oob/self.total:6.2%} | "
                    f"other={self.other/self.total:6.2%}"
                )
        self._reset()


class WaitRewardStatsCallback(BaseCallback):
    """Tracks wait reward statistics from episode infos."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._reset()

    def _reset(self):
        self.n_eps = 0
        self.sum_wait = 0.0
        self.sum_wait_per_step = 0.0
        self.n_eps_with_len = 0

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


# ENVIRONMENT FACTORY UTILITIES
# ----------------------------------------------------------------------------

def make_pursuit_env_factory(pursuer_cls, evader_cls, action_dim):
    """Create an environment factory for pursuit-evasion training."""
    from src.simulation.env_pursuit_evasion import PursuitVecEnv
    return partial(
        PursuitVecEnv,
        evader_cls=evader_cls,
        pursuer_cls=pursuer_cls,
        action_dim=action_dim,
    )


# CONFIG UTILITIES
# ----------------------------------------------------------------------------

def apply_dr_percent(cfg: dict, dr_percent: float) -> dict:
    """
    Set every key in PURSUER.domain_randomization_pct to dr_percent/100.0.
    Returns a deep copy.
    """
    cfg = copy.deepcopy(cfg)
    dr_fraction = float(dr_percent) / 100.0
    dr_dict = cfg["PURSUER"].get("domain_randomization_pct", {})
    for k in list(dr_dict.keys()):
        try:
            dr_dict[k] = float(dr_fraction)
        except Exception:
            pass
    cfg["PURSUER"]["domain_randomization_pct"] = dr_dict
    return cfg


def build_level_config(base_cfg: dict, level_info: dict) -> dict:
    """
    Build a config for a specific abstraction level.
    
    Args:
        base_cfg: Base configuration dictionary
        level_info: Dictionary with keys: opt_feats, action_dim, model_name
    
    Returns:
        Deep-copied config patched for the level
    """
    cfg = copy.deepcopy(base_cfg)
    obs = cfg["OBSERVATIONS"]
    obs["OPTIONAL_FEATURES"] = list(level_info.get("opt_feats", []))
    obs["ACTION_DIM"] = level_info.get("action_dim", 4)
    cfg["PURSUER"]["MODEL"] = level_info.get("model_name", "motor")
    
    # Ensure history flags are off unless explicitly needed
    obs.setdefault("INCLUDE_HISTORY", False)
    obs.setdefault("INCLUDE_ACTION_HISTORY", False)
    
    return cfg


def cfg_with_reward(cfg: dict, reward_type: str, smoothing_gamma: float) -> dict:
    """Apply reward string modifiers and smoothing gamma to a config copy."""
    cfg = copy.deepcopy(cfg)
    cfg["reward_type"] = reward_type
    cfg["SMOOTHING_GAMMA"] = float(smoothing_gamma)
    return cfg


# LOGGING UTILITIES
# ----------------------------------------------------------------------------

def log_observation_config(cfg: dict, logger_name: str = "training", **extra_log_kwargs):
    """Log observation configuration in a consistent format."""
    log = get_logger(logger_name)
    obs = cfg.get("OBSERVATIONS", {})
    log.info(
        "OBS: mode=%s, action_dim=%s, hist=%s(h=%s), act_hist=%s(h=%s), opt=%s%s",
        obs.get("OBS_MODE"),
        obs.get("ACTION_DIM"),
        obs.get("INCLUDE_HISTORY"),
        obs.get("HISTORY_STEPS"),
        obs.get("INCLUDE_ACTION_HISTORY"),
        obs.get("ACTION_HISTORY_STEPS"),
        obs.get("OPTIONAL_FEATURES"),
        "".join([f", {k}={v}" for k, v in extra_log_kwargs.items()]),
    )


# TRAINING CONFIGURATION
# ----------------------------------------------------------------------------

class TrainingConfig:
    """Configuration object for a training run."""
    
    def __init__(
        self,
        total_timesteps: int = 25_000_000,
        n_envs: int = 100,
        n_eval_envs: int = 4,
        seed: int = 0,
        eval_freq: Optional[int] = None,
        n_eval_episodes: int = 10,
        checkpoint_freq: Optional[int] = None,
        early_stopping: Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict] = None,
        use_action_smoothing_stats: bool = False,
        use_episode_termination_stats: bool = False,
        use_rollout_termination_stats: bool = False,
        use_wait_reward_stats: bool = False,
        info_keywords: Optional[Tuple[str, ...]] = None,
    ):
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.n_eval_envs = n_eval_envs
        self.seed = seed
        self.eval_freq = eval_freq or max(250_000 // n_envs, 1)
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_freq = checkpoint_freq or (1_000_000 // n_envs)
        self.early_stopping = early_stopping or {}
        self.policy_kwargs = policy_kwargs or dict(net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]))
        self.use_action_smoothing_stats = use_action_smoothing_stats
        self.use_episode_termination_stats = use_episode_termination_stats
        self.use_rollout_termination_stats = use_rollout_termination_stats
        self.use_wait_reward_stats = use_wait_reward_stats
        self.info_keywords = info_keywords or ()


# MAIN TRAINING FUNCTION
# ----------------------------------------------------------------------------

def train_one_config(
    env_factory: Callable,
    train_cfg: Dict[str, Any],
    eval_cfg: Optional[Dict[str, Any]] = None,
    output_tag: str = "run",
    root_output_dir: str = "training_output",
    training_config: Optional[TrainingConfig] = None,
    resume_from_path: Optional[str] = None,
    logger_name: str = "training",
    extra_callbacks: Optional[List] = None,
) -> str:
    """
    Train a PPO agent with the given configuration.
    
    Args:
        env_factory: Callable that creates an environment when called with (num_envs, config)
        train_cfg: Configuration dictionary for training
        eval_cfg: Configuration dictionary for evaluation (defaults to train_cfg)
        output_tag: Tag for organizing output directories
        root_output_dir: Root directory for outputs
        training_config: TrainingConfig object with hyperparameters
        resume_from_path: Path to resume training from (optional)
        logger_name: Logger name for logging
        extra_callbacks: Additional callbacks to include
        
    Returns:
        Path to the final model
    """
    if training_config is None:
        training_config = TrainingConfig()
    
    if eval_cfg is None:
        eval_cfg = copy.deepcopy(train_cfg)
    
    # Create output directories
    modeldir = os.path.join(root_output_dir, output_tag, "models")
    logdir = os.path.join(root_output_dir, output_tag, "logs")
    os.makedirs(modeldir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    
    # Save observation layout
    save_observation_layout_csv(train_cfg, os.path.join(modeldir, "obs_layout.csv"))
    
    # Log configuration
    log = get_logger(logger_name)
    log_observation_config(train_cfg, logger_name)
    
    # Create environments
    train_env = VecMonitor(
        env_factory(num_envs=training_config.n_envs, config=train_cfg),
        info_keywords=training_config.info_keywords,
    )
    eval_env = VecMonitor(
        env_factory(num_envs=training_config.n_eval_envs, config=eval_cfg),
        info_keywords=training_config.info_keywords,
    )
    
    # Create or load model
    if resume_from_path:
        log.info(f"Resuming from: {resume_from_path}")
        model = PPO.load(resume_from_path, env=train_env, tensorboard_log=logdir)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=1000,
            batch_size=5000,
            n_epochs=10,
            gamma=0.999,
            tensorboard_log=logdir,
            verbose=0,
            seed=training_config.seed,
            policy_kwargs=training_config.policy_kwargs,
        )
    
    # Build callbacks
    callbacks = []
    
    # Early stopping callback
    if training_config.early_stopping:
        plateau_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=training_config.early_stopping.get("max_no_improvement_evals", 10),
            min_evals=training_config.early_stopping.get("min_evals", 5),
            verbose=1,
        )
    else:
        plateau_cb = None
    
    # Eval callback
    callbacks.append(
        EvalCallback(
            eval_env,
            best_model_save_path=modeldir,
            log_path=logdir,
            eval_freq=training_config.eval_freq,
            n_eval_episodes=training_config.n_eval_episodes,
            deterministic=True,
            callback_after_eval=plateau_cb,
        )
    )
    
    # Checkpoint callback
    callbacks.append(
        CheckpointEverySteps(
            freq=training_config.checkpoint_freq,
            folder=modeldir,
        )
    )
    
    # Progress bar
    callbacks.append(ProgressBarCallback())
    
    # Optional stats callbacks
    if training_config.use_action_smoothing_stats:
        callbacks.append(ActionSmoothingStatsCallback())
    
    if training_config.use_episode_termination_stats:
        callbacks.append(EpisodeTerminationStatsCallback())
    
    if training_config.use_rollout_termination_stats:
        callbacks.append(RolloutTerminationStatsCallback())
    
    if training_config.use_wait_reward_stats:
        callbacks.append(WaitRewardStatsCallback())
    
    # Extra callbacks
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    
    # Train
    log.info(f"Training for {training_config.total_timesteps:,} timesteps")
    model.learn(total_timesteps=training_config.total_timesteps, callback=callbacks)
    
    # Save final model
    final_path = os.path.join(modeldir, "ppo_final")
    model.save(final_path)
    log.info(f"Training complete. Model saved to {final_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return final_path


# SWEEP RUNNER
# ----------------------------------------------------------------------------

def run_configuration_sweep(
    configs: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]],
    env_factory: Callable,
    root_output_dir: str,
    training_config: Optional[TrainingConfig] = None,
    logger_name: str = "training",
    extra_callbacks: Optional[List] = None,
):
    """
    Run training for multiple configurations.
    
    Args:
        configs: List of tuples (tag, train_cfg, eval_cfg) for each run
        env_factory: Environment factory function
        root_output_dir: Root directory for outputs
        training_config: TrainingConfig object
        logger_name: Logger name
        extra_callbacks: Additional callbacks
    """
    log = get_logger(logger_name)
    os.makedirs(root_output_dir, exist_ok=True)
    
    for tag, train_cfg, eval_cfg in configs:
        log.info(f"Starting training for: {tag}")
        try:
            train_one_config(
                env_factory=env_factory,
                train_cfg=train_cfg,
                eval_cfg=eval_cfg,
                output_tag=tag,
                root_output_dir=root_output_dir,
                training_config=training_config,
                logger_name=logger_name,
                extra_callbacks=extra_callbacks,
            )
            log.info(f"Completed training for: {tag}")
        except Exception as e:
            log.error(f"Error training {tag}: {e}", exc_info=True)
    
    log.info("All configurations completed")

