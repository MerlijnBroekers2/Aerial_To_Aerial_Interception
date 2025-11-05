import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from src.models.evaders.evader_action_provider import make_rl_evader_action_provider
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
        self,
        num_envs: int,
        config: dict,
        evader_cls,
        pursuer_cls,
        action_dim: int,
        evader_action_provider=None,
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
        self.oob_penalty = config["OUT_OF_BOUNDS_PENALTY"]
        self.rate_penalty = config["RATE_PENALTY"]
        self.max_history_steps = config["OBSERVATIONS"]["MAX_HISTORY_STEPS"]
        self.wait_reward_base = 0.001
        self.wait_reward_max = 0.01
        self.wait_reward_decay_beta = 2
        self.cum_wait_reward = np.zeros(num_envs, np.float32)

        # ---- boundary shaping knobs ----
        self.bounds_min, self.bounds_max = self._parse_bounds(config)
        self.boundary_margin = float(config["PURSUER"]["BOUNDARIES"]["BOUNDARY_MARGIN"])
        self.boundary_weight = float(
            config["PURSUER"]["BOUNDARIES"]["BOUNDARY_PENALTY_WEIGHT"]
        )
        self.boundary_mode = str(
            config["PURSUER"]["BOUNDARIES"]["BOUNDARY_MODE"].lower()
        )

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

        self.cum_action_l1 = np.zeros(num_envs, np.float32)
        self.cum_action_tv = np.zeros(num_envs, np.float32)

        # ---- (optional) frustum planes ---------------------------------
        planes_cfg = config["PURSUER"]["BOUNDARIES"]["PLANES"]
        if planes_cfg:
            P0 = np.array([p["p0"] for p in planes_cfg], dtype=np.float32)  # (6,3)
            N = np.array([p["n"] for p in planes_cfg], dtype=np.float32)  # (6,3)
            # normalize normals so signed distance is in meters
            n_norm = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
            N_hat = N / n_norm
            self._planes_p0 = P0
            self._planes_n = N_hat
        else:
            self._planes_p0 = None
            self._planes_n = None

        # ---- instantiate agents -------------------------------------
        self.evader = evader_cls(num_envs, config)
        self.pursuers = pursuer_cls(num_envs=num_envs, config=config)

        self.ev_mode = str(config["EVADER"]["MODEL"])

        self.ev_mode = str(config["EVADER"]["MODEL"])
        if self.ev_mode == "rl":
            # If caller supplies a frozen provider, use it; otherwise build from cfg
            self._evader_act = evader_action_provider or make_rl_evader_action_provider(
                config, num_envs
            )
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

        # --- Seed “previous” buffers with current state ---
        ev_pos = self.evader.get_state()["true_position"][dones]
        pu_pos = self.pursuers.get_state()["true_position"][dones]

        # set prev distance to current dist so first delta is ~0
        self.prev_dist[dones] = np.linalg.norm(ev_pos - pu_pos, axis=1)

        # you already had this:
        self.prev_ev_pos[dones] = ev_pos

        # histories & actions
        self.obs_history[dones] = 0.0
        self.act_history[dones] = 0.0
        self.cum_action_l1[dones] = 0.0
        self.cum_action_tv[dones] = 0.0
        self.prev_actions[dones] = 0.0
        self.cum_wait_reward[dones] = 0.0

        return self.obs

    def step_async(self, actions):
        assert actions.shape == (self.num_envs, self.action_dim)
        self._actions[:] = actions

    def step_wait(self):
        # ---- history -------------------------------------------------
        self._store_history()

        ev, pu = self.evader.get_state(), self.pursuers.get_state()

        # ---- advance agents -----------------------------------------
        self.pursuers.step_learn(self._actions)

        if self.ev_mode == "rl":
            ev_actions = self._evader_act(ev_state=ev, pu_state=pu)  # (N,3) in [-1,1]
            # VectorizedReactiveEvader.step(acc_cmd) style
            self.evader.step(ev_actions)
        else:
            self.evader.step()

        self.timestep += 1

        ev, pu = (
            self.evader.get_state(),
            self.pursuers.get_state(),
        )  # fetch state after steping for reward assignment

        # ---- reward & done ------------------------------------------
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
    def _signed_dists_to_planes(self, p: np.ndarray) -> np.ndarray:
        """
        p: (N,3) points
        returns s: (N,6) signed distances to each plane (>=0 is inside).
        Requires self._planes_p0/_planes_n to be set.
        """
        # (N,1,3) - (1,6,3) -> (N,6,3)
        diff = p[:, None, :] - self._planes_p0[None, :, :]
        # (N,6,3) · (1,6,3) -> sum over last axis -> (N,6)
        s = np.sum(diff * self._planes_n[None, :, :], axis=2)
        return s

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

        ev_step = np.linalg.norm(ev["true_position"] - self.prev_ev_pos, axis=1)
        action_delta = np.linalg.norm(self._actions - self.prev_actions, axis=1)
        rates_norm = np.linalg.norm(pu["rates"], axis=1)

        # --- Evader "inside" check uses planes if available -------------
        if self._planes_n is not None:
            s_ev = self._signed_dists_to_planes(ev["true_position"])  # (N,6)
            # gate at margin like your old box code (inside the inner frustum)
            ev_inside = np.all(s_ev >= self.boundary_margin, axis=1)
        else:
            ev_p = ev["true_position"]
            ev_inside = np.all(
                (ev_p >= self.bounds_min[None, :] + self.boundary_margin)
                & (ev_p <= self.bounds_max[None, :] - self.boundary_margin),
                axis=1,
            )

        # Base interception reward (as before) ...
        base_R = get_reward(
            dists=dists,
            caught=caught,
            reward_type=self.reward_type,
            prev_dists=self.prev_dist,
            ev_step=ev_step,
            action_delta=action_delta,
            rates_norm=rates_norm,
            cfg=self.config,
            action_dim=self.action_dim,
        )

        # ... but it is only active if the evader is INSIDE
        rewards = np.where(ev_inside, base_R, 0.0).astype(np.float32)

        act_mag_sq = np.sum(self._actions * self._actions, axis=1)  # (N,)
        wait_reward = (
            self.wait_reward_base
            + self.wait_reward_max * np.exp(-self.wait_reward_decay_beta * act_mag_sq)
        ).astype(np.float32)

        wait_component = (~ev_inside).astype(np.float32) * wait_reward  # (N,)
        rewards += wait_component

        # accumulate episode total of waiting reward for logging
        self.cum_wait_reward += wait_component

        # --- existing boundary shaping & hard penalties -----------------
        force_field_shaping, oob_pos = self._boundary_penalty_and_oob(
            pu["true_position"]
        )
        oob_spiral = np.any(np.abs(pu["rates"]) > 100, axis=1)

        rewards -= force_field_shaping
        rewards[oob_pos | oob_spiral] -= self.oob_penalty

        self._flags = dict(
            caught=caught,
            timed_out=timed,
            out_of_bounds_pos=oob_pos,
            out_of_control_spiral=oob_spiral,
            evader_inside=ev_inside,
        )

        dones = caught | timed | oob_pos | oob_spiral
        return rewards, dones

    def _boundary_penalty_and_oob(self, p: np.ndarray):
        """
        Returns:
          penalty: (N,) float32  soft shaping penalty near boundary
          oob:     (N,) bool     hard OOB flag
        """
        if self._planes_n is not None:
            return self._plane_penalty_and_oob(p)
        else:
            return self._box_penalty_and_oob(p)  # your existing logic moved here

    def _plane_penalty_and_oob(self, p: np.ndarray):
        """
        Plane (frustum) shaping:
          - Inside if all s_i >= 0
          - Soft penalty when 0 <= s_i < margin
          - OOB if any s_i < 0
        """
        s = self._signed_dists_to_planes(p)  # (N,6)
        margin = float(self.boundary_margin)

        # soft ramp only for points INSIDE the frustum (s>=0)
        inside = s >= 0.0  # (N,6)
        if margin > 0.0:
            ramp = np.clip((margin - s) / margin, 0.0, 1.0)  # (N,6)
            ramp *= inside.astype(ramp.dtype)  # zero outside
        else:
            ramp = (s <= 0.0).astype(np.float32)  # degenerate case like before

        # aggregate across planes like your box-mode
        if self.boundary_mode == "max":
            agg = np.max(ramp, axis=1)  # (N,)
        else:
            agg = np.sum(ramp, axis=1)  # (N,)

        penalty = (float(self.boundary_weight) * agg).astype(np.float32)

        # OOB: any plane negative
        oob = np.any(s < 0.0, axis=1)

        return penalty, oob

    def _box_penalty_and_oob(self, p: np.ndarray):
        # --- your current axis-aligned implementation moved here unchanged ---
        d_to_min = p - self.bounds_min[None, :]  # (N,3)
        d_to_max = self.bounds_max[None, :] - p  # (N,3)
        d_near = np.minimum(d_to_min, d_to_max)
        m = self.boundary_margin
        if m <= 0:
            ramp = (d_near <= 0).astype(np.float32)
        else:
            ramp = np.clip((m - d_near) / m, 0.0, 1.0)
        if self.boundary_mode == "max":
            agg = np.max(ramp, axis=1)
        else:
            agg = np.sum(ramp, axis=1)
        penalty = self.boundary_weight * agg
        oob = np.any(
            (p < self.bounds_min[None, :]) | (p > self.bounds_max[None, :]), axis=1
        )
        return penalty.astype(np.float32), oob

    def _infos(self, dones):
        self.cum_action_l1 += np.sum(np.abs(self._actions), axis=1)
        self.cum_action_tv += np.linalg.norm(self._actions - self.prev_actions, axis=1)

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
                info["action_l1_sum"] = float(self.cum_action_l1[i])
                info["action_tv_sum"] = float(self.cum_action_tv[i])
                info["wait_reward_sum"] = float(self.cum_wait_reward[i])

            infos.append(info)
        return infos

    def _track(self, ev, pu):
        self.prev_dist = np.linalg.norm(
            ev["true_position"] - pu["true_position"], axis=1
        )
        self.prev_ev_pos = ev["true_position"]
        self.prev_actions = self._actions.copy()

    def _parse_bounds(self, cfg):
        b = cfg["PURSUER"]["BOUNDARIES"]["ENV_BOUNDS"]
        xs = tuple(b["x"])
        ys = tuple(b["y"])
        zs = tuple(b["z"])
        mins = [min(xs), min(ys), min(zs)]
        maxs = [max(xs), max(ys), max(zs)]
        return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)
