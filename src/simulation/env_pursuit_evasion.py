import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from src.utils.observations import get_observation, get_observation_size
from src.utils.rewards import get_reward


class PursuitVecEnv(VecEnv):
    """
    Generic N-parallel evader-vs-pursuer environment.

    Parameters
    ----------
    num_envs      : int
    config        : CONFIG dict
    evader_cls    : class   – must implement  (num_envs, config) constructor
    pursuer_cls   : class   – must implement  (num_envs, config) constructor
    action_dim    : int     – size of pursuer action vector
    """

    def __init__(
        self, num_envs: int, config: dict, evader_cls, pursuer_cls, action_dim: int
    ):
        # ---- sizes ---------------------------------------------------
        self.config = config
        self.base_dim, self.obs_dim = get_observation_size(config)
        self.action_dim = action_dim

        super().__init__(
            num_envs=num_envs,
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
            ),
        )

        # ---- constants ----------------------------------------------
        self.num_envs = num_envs
        self.dt = config["DT"]
        self.time_limit = config["TIME_LIMIT"]
        self.capture_radius = config["CAPTURE_RADIUS"]
        self.reward_type = config["reward_type"]
        self.max_bound = config["ENV_BOUND"]
        self.oob_penalty = config["OUT_OF_BOUNDS_PENALTY"]
        self.rate_penalty = config["RATE_PENALTY"]
        self.max_history_steps = config["OBSERVATIONS"]["MAX_HISTORY_STEPS"]

        # ---- running buffers ----------------------------------------
        self.prev_dist = np.zeros(num_envs, np.float32)
        self.prev_ev_pos = np.zeros((num_envs, 3), np.float32)
        self.prev_actions = np.zeros((num_envs, action_dim), np.float32)
        self._actions = np.zeros((num_envs, action_dim), np.float32)
        self.timestep = np.zeros(num_envs, np.int32)

        self.obs_history = np.zeros(
            (num_envs, self.max_history_steps, self.base_dim), np.float32
        )
        self.act_history = np.zeros(
            (num_envs, self.max_history_steps, self.action_dim), np.float32
        )

        # ---- instantiate agents -------------------------------------
        self.evader = evader_cls(num_envs, config)
        self.pursuers = pursuer_cls(num_envs=num_envs, config=config)

        # placeholders filled in reset()
        self.obs = None
        self.state = None

    # =================================================================
    # VecEnv interface
    # =================================================================
    def reset(self, dones=None):
        if dones is None:
            dones = np.ones(self.num_envs, dtype=bool)

        self.evader.reset(dones)
        self.pursuers.reset(dones)
        self.timestep[dones] = 0

        self._update_state()
        self.obs = self._build_obs()

        ev_pos = self.evader.get_state()["true_position"][dones]
        self.prev_ev_pos[dones] = ev_pos
        self.obs_history[dones] = 0.0
        self.act_history[dones] = 0.0

        return self.obs

    def step_async(self, actions):
        assert actions.shape == (self.num_envs, self.action_dim)
        self._actions[:] = actions

    def step_wait(self):
        # ---- history -------------------------------------------------
        self._store_history()

        # ---- advance agents -----------------------------------------
        self.pursuers.step_learn(self._actions)
        self.evader.step()
        self.timestep += 1

        # ---- reward & done ------------------------------------------
        ev, pu = self.evader.get_state(), self.pursuers.get_state()
        rewards, dones = self._reward_and_done(ev, pu)
        infos = self._infos(dones)

        # ---- bookkeeping & reset finished envs ----------------------
        self._track(ev, pu)
        self.reset(dones)

        # ---- next obs -----------------------------------------------
        self.obs = self._build_obs()
        self._update_state()

        return self.obs, rewards, dones, infos

    def close(self):
        pass

    def get_images(self):
        return []

    def _get_event_loop(self): ...
    def get_attr(self, *_, **__):
        return [None] * self.num_envs

    def set_attr(self, *_, **__):
        pass

    def env_method(self, *_, **__):
        pass

    def env_is_wrapped(self, *_, **__):
        return [False] * self.num_envs

    # =================================================================
    # helpers
    # =================================================================
    def _update_state(self):
        """Small convenience buffer (not strictly required)."""
        ev, pu = self.evader.get_state(), self.pursuers.get_state()
        if self.state is None:
            self.state = np.zeros((self.num_envs, 12), np.float32)
        self.state[:, 0:3] = ev["true_position"]
        self.state[:, 3:6] = ev["velocity"]
        self.state[:, 6:9] = pu["true_position"]
        self.state[:, 9:12] = pu["velocity"]

    def _build_obs(self):
        ev, pu = self.evader.get_state(), self.pursuers.get_state()
        return get_observation(
            config=self.config,
            ev_state=ev,
            pu_state=pu,
            obs_history=self.obs_history,
            act_history=self.act_history,
        )

    def _store_history(self):
        core = self.obs[:, : self.base_dim]
        self.obs_history = np.roll(self.obs_history, shift=-1, axis=1)
        self.obs_history[:, -1, :] = core

        self.act_history = np.roll(self.act_history, shift=-1, axis=1)
        self.act_history[:, -1, :] = self._actions

    # -----------------------------------------------------------------
    def _reward_and_done(self, ev, pu):
        dists = np.linalg.norm(ev["true_position"] - pu["true_position"], axis=1)
        caught = dists < self.capture_radius
        timed = self.timestep >= self.time_limit

        oob_pos = np.any(np.abs(pu["true_position"]) > self.max_bound, axis=1)
        oob_spiral = np.any(np.abs(pu["rates"]) > 1000, axis=1)
        dones = caught | timed | oob_pos | oob_spiral

        ev_step = np.linalg.norm(ev["true_position"] - self.prev_ev_pos, axis=1)
        action_delta = np.linalg.norm(self._actions - self.prev_actions, axis=1)

        rewards = get_reward(
            dists=dists,
            caught=caught,
            reward_type=self.reward_type,
            prev_dists=self.prev_dist,
            ev_step=ev_step,
            action_delta=action_delta,
        )
        rewards -= self.rate_penalty * np.linalg.norm(pu["rates"], axis=1)
        rewards[oob_pos | oob_spiral] -= self.oob_penalty

        # remember for infos()
        self._flags = dict(
            caught=caught,
            timed_out=timed,
            out_of_bounds_pos=oob_pos,
            out_of_control_spiral=oob_spiral,
        )
        return rewards, dones

    def _infos(self, dones):
        infos = []
        for i in range(self.num_envs):
            info = {}
            if dones[i]:
                info["terminal_observation"] = self.obs[i].copy()
                info["TimeLimit.truncated"] = not (self.timestep[i] < self.time_limit)
                for k in self._flags:
                    if self._flags[k][i]:
                        info["term_reason"] = k
                        break
            infos.append(info)
        return infos

    def _track(self, ev, pu):
        self.prev_dist = np.linalg.norm(
            ev["true_position"] - pu["true_position"], axis=1
        )
        self.prev_ev_pos = ev["true_position"]
        self.prev_actions = self._actions
