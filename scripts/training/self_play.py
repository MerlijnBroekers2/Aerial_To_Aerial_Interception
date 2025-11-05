import os
import copy
from datetime import datetime
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    ProgressBarCallback,
    BaseCallback,
)

from scripts.training.single_env import (
    ActionSmoothingStatsCallback,
    CheckpointCallback,
    EpisodeTerminationStatsCallback,
)
from src.models.evaders.evader_action_provider import make_rl_evader_action_provider
from src.models.pursuers.vec_pursuer_CTBR_INDI import VecCTBR_INDI_Pursuer
from src.simulation.env_evasion_pursuit import EvaderVsPursuerVecEnv
from src.simulation.env_pursuit_evasion import PursuitVecEnv  # <-- pursuer trains here
from src.models.evaders.vec_reactive_evader import VectorizedReactiveEvader
from src.models.pursuers.vec_pursuer_Motor import VecMotorPursuer
from src.models.pursuers.pursuer_action_provider import make_rl_pursuer_action_provider
from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger, coerce_level


def resolve_pursuer_cls_and_action_dim(cfg):
    model = str(cfg["PURSUER"]["MODEL"]).lower()
    if model in ("motor", "motor_rl"):
        return VecMotorPursuer, 4
    if model in ("ctbr", "ctbr_indi", "ctbr-indi"):
        return VecCTBR_INDI_Pursuer, 4
    raise ValueError(f"Unknown PURSUER.MODEL='{model}'")


# --- simple % termination logger during training ---
class RolloutTerminationStatsCallback(BaseCallback):
    def __init__(self, prefix="rollout", verbose=0):
        super().__init__(verbose)
        self.prefix = prefix
        self._reset()

    def _reset(self):
        self.total = self.caught = self.timed = self.other = 0

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
        self._reset()


def train_evader_phase(
    config,
    frozen_pursuer_model_path,
    out_dir,
    total_timesteps,
    train_envs,
    n_steps,
    batch_size,
    resume_evader_from_path=None,  # NEW (optional)
):
    log = get_logger("training.self_play")
    # --- copy config and point pursuer policy at the frozen path ---
    cfg = copy.deepcopy(config)
    cfg["PURSUER"]["CONTROLLER"]["policy_path"] = frozen_pursuer_model_path

    print(f"[EvaderPhase] Frozen pursuer model: {frozen_pursuer_model_path}")
    obs = cfg.get("OBSERVATIONS", {})
    log.info(
        "EvaderPhase OBS: mode=%s, action_dim=%s, hist=%s(h=%s), act_hist=%s(h=%s), opt=%s",
        obs.get("OBS_MODE"),
        obs.get("ACTION_DIM"),
        obs.get("INCLUDE_HISTORY"),
        obs.get("HISTORY_STEPS"),
        obs.get("INCLUDE_ACTION_HISTORY"),
        obs.get("ACTION_HISTORY_STEPS"),
        obs.get("OPTIONAL_FEATURES"),
    )
    if resume_evader_from_path:
        print(f"[EvaderPhase] RESUME evader policy from: {resume_evader_from_path}")
    else:
        print("[EvaderPhase] Training evader FROM SCRATCH")

    # frozen pursuer action provider (train/eval separate histories)
    act_pu_train = make_rl_pursuer_action_provider(cfg, train_envs)
    act_pu_eval = make_rl_pursuer_action_provider(cfg, 1)

    pursuer_cls, _ = resolve_pursuer_cls_and_action_dim(cfg)

    TrainEnv = partial(
        EvaderVsPursuerVecEnv,
        evader_cls=VectorizedReactiveEvader,
        pursuer_cls=pursuer_cls,
        action_provider=act_pu_train,
        evader_action_dim=3,
    )
    EvalEnv = partial(
        EvaderVsPursuerVecEnv,
        evader_cls=VectorizedReactiveEvader,
        pursuer_cls=pursuer_cls,
        action_provider=act_pu_eval,
        evader_action_dim=3,
    )

    train_env = VecMonitor(TrainEnv(num_envs=train_envs, config=cfg))
    eval_env = VecMonitor(EvalEnv(num_envs=1, config=cfg))

    # ---- create or resume evader PPO ----
    if resume_evader_from_path:
        model = PPO.load(
            resume_evader_from_path,
            env=train_env,
            tensorboard_log=os.path.join(out_dir, "tb"),
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            tensorboard_log=os.path.join(out_dir, "tb"),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.999,
            policy_kwargs=cfg["POLICY_KWARGS"],
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=max(1_000_000 // train_envs, 1),
        n_eval_episodes=1,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=1_000_000, save_path=out_dir, prefix="evader_ckpt", verbose=1
    )
    ro_cb = RolloutTerminationStatsCallback(prefix="evader/rollout")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, ckpt_cb, ProgressBarCallback(), ro_cb],
    )

    # Prefer best_model for passing to next round; else use final
    best_path = os.path.join(out_dir, "best_model.zip")
    final_path = os.path.join(out_dir, "evader_final.zip")
    model.save(final_path)

    chosen = best_path if os.path.exists(best_path) else final_path
    print(
        f"[EvaderPhase] Done. best={os.path.exists(best_path)}  Using for next round: {chosen}"
    )
    return chosen  # <— return the path we will *use* next


def train_pursuer_phase(
    config,
    frozen_evader_model_path,
    out_dir,
    total_timesteps,
    train_envs,
    n_steps,
    batch_size,
    resume_pursuer_from_path=None,  # NEW (this is the key fix)
):
    log = get_logger("training.self_play")
    # --- copy config so EVADER is RL and points to the frozen evader path ---
    cfg = copy.deepcopy(config)
    cfg["EVADER"]["MODEL"] = "rl"  # <-- env should run an RL evader internally
    cfg["EVADER"]["RL_MODEL_PATH"] = frozen_evader_model_path

    print(f"[PursuerPhase] Frozen evader model: {frozen_evader_model_path}")
    obs = cfg.get("OBSERVATIONS", {})
    log.info(
        "PursuerPhase OBS: mode=%s, action_dim=%s, hist=%s(h=%s), act_hist=%s(h=%s), opt=%s",
        obs.get("OBS_MODE"),
        obs.get("ACTION_DIM"),
        obs.get("INCLUDE_HISTORY"),
        obs.get("HISTORY_STEPS"),
        obs.get("INCLUDE_ACTION_HISTORY"),
        obs.get("ACTION_HISTORY_STEPS"),
        obs.get("OPTIONAL_FEATURES"),
    )
    if resume_pursuer_from_path:
        print(f"[PursuerPhase] RESUME pursuer policy from: {resume_pursuer_from_path}")
    else:
        print("[PursuerPhase] Training pursuer FROM SCRATCH")

    pursuer_cls, action_dim = resolve_pursuer_cls_and_action_dim(cfg)

    ev_act_train = make_rl_evader_action_provider(
        cfg,
        num_envs=train_envs,
        model_path=frozen_evader_model_path,
        deterministic=True,
    )
    ev_act_eval = make_rl_evader_action_provider(
        cfg, num_envs=1, model_path=frozen_evader_model_path, deterministic=True
    )

    TrainEnv = partial(
        PursuitVecEnv,
        evader_cls=VectorizedReactiveEvader,
        pursuer_cls=pursuer_cls,
        action_dim=action_dim,
        evader_action_provider=ev_act_train,
    )
    EvalEnv = partial(
        PursuitVecEnv,
        evader_cls=VectorizedReactiveEvader,
        pursuer_cls=pursuer_cls,
        action_dim=action_dim,
        evader_action_provider=ev_act_eval,
    )

    train_env = VecMonitor(
        TrainEnv(num_envs=train_envs, config=cfg),
        info_keywords=("action_l1_sum", "action_tv_sum"),
    )
    eval_env = VecMonitor(
        EvalEnv(num_envs=1, config=cfg),
        info_keywords=("action_l1_sum", "action_tv_sum"),
    )

    # ---- create or resume pursuer PPO ----
    if resume_pursuer_from_path:
        model = PPO.load(
            resume_pursuer_from_path,
            env=train_env,
            tensorboard_log=os.path.join(out_dir, "tb"),
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            tensorboard_log=os.path.join(out_dir, "tb"),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.999,
            policy_kwargs=cfg["POLICY_KWARGS"],
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=max(1_000_000 // train_envs, 1),
        n_eval_episodes=1,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=1_000_000, save_path=out_dir, prefix="pursuer_ckpt", verbose=1
    )
    ro_cb = RolloutTerminationStatsCallback(prefix="pursuer/rollout")
    ep_cb = EpisodeTerminationStatsCallback(verbose=0)
    smooth_cb = ActionSmoothingStatsCallback(verbose=0)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, ckpt_cb, ProgressBarCallback(), ro_cb, ep_cb, smooth_cb],
    )

    # Prefer best_model for passing to next round; else use final
    best_path = os.path.join(out_dir, "best_model.zip")
    final_path = os.path.join(out_dir, "pursuer_final.zip")
    model.save(final_path)

    chosen = best_path if os.path.exists(best_path) else final_path
    print(
        f"[PursuerPhase] Done. best={os.path.exists(best_path)}  Using for next round: {chosen}"
    )
    return chosen  # <— return the path we will *use* next


def alternating_self_play(
    config,
    rounds=4,
    evader_steps_per_round=5_000_000,
    pursuer_steps_per_round=5_000_000,
    train_envs=100,
    n_steps=1000,
    batch_size=5000,
    init_pursuer_policy_path=None,
):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    root = os.path.join("final_models_selfplay", f"ALT_{timestamp}")
    os.makedirs(root, exist_ok=True)

    if init_pursuer_policy_path is None:
        init_pursuer_policy_path = config["PURSUER"]["CONTROLLER"]["policy_path"]

    # this is the *current* pursuer model that we’ll keep improving
    current_pursuer = init_pursuer_policy_path
    current_evader = None  # None until first evader round finishes

    print(f"[SelfPlay] Initial pursuer seed: {current_pursuer}")

    for k in range(rounds):
        print(f"\n=== Round {k+1}/{rounds} — EVADER trains vs frozen PURSUER ===")
        ev_dir = os.path.join(root, f"round_{k+1}_evader")
        os.makedirs(ev_dir, exist_ok=True)

        # Evader trains against the *frozen* pursuer model (current_pursuer)
        current_evader = train_evader_phase(
            config=config,
            frozen_pursuer_model_path=current_pursuer,
            out_dir=ev_dir,
            total_timesteps=evader_steps_per_round,
            train_envs=train_envs,
            n_steps=n_steps,
            batch_size=batch_size,
            resume_evader_from_path=None,  # or pass a path to continue evader too
        )
        print(f"[SelfPlay] New evader model for next phase: {current_evader}")

        print(f"\n=== Round {k+1}/{rounds} — PURSUER trains vs frozen EVADER ===")
        pu_dir = os.path.join(root, f"round_{k+1}_pursuer")
        os.makedirs(pu_dir, exist_ok=True)

        # Pursuer continues training from the *previous* pursuer model
        current_pursuer = train_pursuer_phase(
            config=config,
            frozen_evader_model_path=current_evader,
            out_dir=pu_dir,
            total_timesteps=pursuer_steps_per_round,
            train_envs=train_envs,
            n_steps=n_steps,
            batch_size=batch_size,
            resume_pursuer_from_path=current_pursuer,  # <— CRUCIAL
        )
        print(f"[SelfPlay] New pursuer model for next round: {current_pursuer}")

    print(
        f"\n[Alternating] Done.\n"
        f"Final evader:  {current_evader}\n"
        f"Final pursuer: {current_pursuer}"
    )


if __name__ == "__main__":
    setup_logging(coerce_level(CONFIG.get("LOG_LEVEL", "INFO")), name="training.self_play")
    log = get_logger("training.self_play")
    log.info(
        "Start self_play: pursuer=%s, DT=%.4f",
        CONFIG["PURSUER"].get("MODEL"),
        CONFIG.get("DT"),
    )
    alternating_self_play(
        CONFIG,
        rounds=4,
        evader_steps_per_round=15_000_000,
        pursuer_steps_per_round=15_000_000,
        train_envs=100,
        n_steps=1000,
        batch_size=5000,
        init_pursuer_policy_path=CONFIG["PURSUER"]["CONTROLLER"]["policy_path"],
    )
