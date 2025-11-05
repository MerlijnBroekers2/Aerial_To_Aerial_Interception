import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class EvaderVsPursuerVecEnv(VecEnv):
    """
    Learning agent: EVADER (accel commands). Opponent: VecMotorPursuer driven
    by a provided action_provider callable (e.g., PPO policy).

    action_provider(ev_state, pu_state) -> (N,action_dim) in [-1,1]
    """

    def __init__(
        self,
        num_envs,
        config,
        evader_cls,
        pursuer_cls,
        action_provider,
        evader_action_dim=3,
    ):
        self.cfg = config
        self.N = num_envs
        self.dt = float(self.cfg["DT"])
        self.T = int(self.cfg["TIME_LIMIT"])
        self.R = float(self.cfg["CAPTURE_RADIUS"])
        self.cap_pen = float(self.cfg["CAPTURE_PENALTY"])
        self.oob_penalty = config["OUT_OF_BOUNDS_PENALTY"]

        # ---- boundary shaping knobs ----
        self.bounds_min, self.bounds_max = self._parse_bounds(config)
        self.boundary_margin = float(config["EVADER"]["BOUNDARIES"]["BOUNDARY_MARGIN"])
        self.boundary_weight = float(
            config["EVADER"]["BOUNDARIES"]["BOUNDARY_PENALTY_WEIGHT"]
        )
        self.boundary_mode = str(
            config["EVADER"]["BOUNDARIES"]["BOUNDARY_MODE"].lower()
        )

        # actors
        self.evader = evader_cls(num_envs, self.cfg)
        self.pursuer = pursuer_cls(config=self.cfg, num_envs=num_envs)
        self.act_pu = action_provider

        # Obs: [ Δp, Δv, p_e ] = 3 + 3 + 3 = 9
        self.obs_dim = 9
        self.adim = evader_action_dim

        super().__init__(
            num_envs=num_envs,
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.adim,), dtype=np.float32
            ),
        )

        self.timestep = np.zeros(self.N, np.int32)
        self.prev_dist = np.zeros(self.N, np.float32)
        self.prev_pu_pos = np.zeros((self.N, 3), np.float32)
        self._actions = np.zeros((self.N, self.adim), np.float32)
        self._flags = {}
        self.obs = None

    # -------- VecEnv API --------
    def reset(self, dones=None):
        if dones is None:
            dones = np.ones(self.N, dtype=bool)
        self.evader.reset(dones)
        self.pursuer.reset(dones)
        self.timestep[dones] = 0

        ev, pu = self.evader.get_state(), self.pursuer.get_state()
        d = np.linalg.norm(ev["true_position"] - pu["true_position"], axis=1).astype(
            np.float32
        )
        self.prev_dist[dones] = d[dones]
        self.prev_pu_pos[dones] = pu["true_position"][dones]

        self.obs = self._build_evader_obs(ev, pu)
        return self.obs

    def step_async(self, actions):
        assert actions.shape == (self.N, self.adim)
        self._actions[:] = actions

    def step_wait(self):
        ev, pu = self.evader.get_state(), self.pursuer.get_state()

        # pursuer acts via trained PPO (or any provided controller)
        pu_actions = self.act_pu(ev_state=ev, pu_state=pu)  # (N,4)

        # step dynamics
        self.pursuer.step_learn(pu_actions)
        self.evader.step(self._actions)
        self.timestep += 1

        # reward + done
        ev, pu = self.evader.get_state(), self.pursuer.get_state()
        rewards, dones = self._reward_and_done(ev, pu)
        infos = self._infos(dones)

        # reset finished and build next obs
        self.reset(dones)
        self.obs = self._build_evader_obs(ev, pu)
        return self.obs, rewards, dones, infos

    def close(self):
        pass

    def get_images(self):
        return []

    def _get_event_loop(self): ...
    def get_attr(self, *_, **__):
        return [None] * self.N

    def set_attr(self, *_, **__):
        pass

    def env_method(self, *_, **__):
        pass

    def env_is_wrapped(self, *_, **__):
        return [False] * self.N

    # -------- helpers --------
    def _parse_bounds(self, cfg):
        b = cfg["EVADER"]["BOUNDARIES"]["ENV_BOUNDS"]
        xs = tuple(b["x"])
        ys = tuple(b["y"])
        zs = tuple(b["z"])
        mins = [min(xs), min(ys), min(zs)]
        maxs = [max(xs), max(ys), max(zs)]
        return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)

    def _build_evader_obs(self, ev, pu):
        dp = pu["noisy_position"] - ev["filtered_position"]
        dv = pu["noisy_velocity"] - ev["filtered_velocity"]
        # ev["true_position"] is inertial/world (NED-friendly), included to learn boundaries
        return np.concatenate([dp, dv, ev["filtered_position"]], axis=1).astype(
            np.float32
        )

    def _reward_and_done(self, ev, pu):
        # --- distance shaping + capture ---
        d = np.linalg.norm(ev["true_position"] - pu["true_position"], axis=1).astype(
            np.float32
        )
        caught = d < self.R
        timed = self.timestep >= self.T
        delta_d = d - self.prev_dist
        r = delta_d
        # r = d / self.T
        r[caught] += self.cap_pen

        # --- boundary penalty (linear ramp) + OOB on pursuer (match PursuitVecEnv) ---
        force_field_shaping, oob_pos = self._boundary_penalty_and_oob(
            pu["true_position"].astype(np.float32)
        )
        r -= force_field_shaping
        r[oob_pos] -= self.oob_penalty

        # --- dones ---
        dones = caught | timed | oob_pos

        # store flags
        self._flags = dict(
            caught=caught,
            timed_out=timed,
            out_of_bounds=oob_pos,
        )

        self.prev_pu_pos = pu["true_position"].astype(np.float32)
        self.prev_dist = d
        return r.astype(np.float32), dones

    def _boundary_penalty_and_oob(self, p: np.ndarray):
        d_to_min = p - self.bounds_min[None, :]  # (N,3)
        d_to_max = self.bounds_max[None, :] - p  # (N,3)
        d_near = np.minimum(d_to_min, d_to_max)  # (N,3), >= 0 inside

        m = self.boundary_margin
        if m <= 0:
            ramp = (d_near <= 0).astype(np.float32)
        else:
            ramp = np.clip((m - d_near) / m, 0.0, 1.0)  # (N,3)

        if self.boundary_mode == "max":
            agg = np.max(ramp, axis=1)
        else:
            agg = np.sum(ramp, axis=1)

        penalty = self.boundary_weight * agg  # (N,)

        oob = np.any(
            (p < self.bounds_min[None, :]) | (p > self.bounds_max[None, :]), axis=1
        )

        return penalty.astype(np.float32), oob

    def _infos(self, dones):
        infos = []
        for i in range(self.N):
            info = {}
            # always expose OOB boolean (useful even if not a terminal cause)
            info["out_of_bounds"] = bool(self._flags["out_of_bounds"][i])

            if dones[i]:
                # collect all true reasons
                reasons = [k for k, mask in self._flags.items() if mask[i]]
                info["term_reasons"] = reasons
                # keep legacy single reason (first in a priority order)
                for key in ("caught", "out_of_bounds", "timed_out"):
                    if self._flags[key][i]:
                        info["term_reason"] = key
                        break
            infos.append(info)
        return infos
