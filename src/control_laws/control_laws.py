import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from stable_baselines3 import PPO

EPSILON = 1e-6


class Input(Enum):
    """What kind of data does the controller need each step?"""

    GUIDANCE_DICT = auto()  # legacy classic PN, FRPN …
    OBS_VECTOR = auto()  # RL policies (or anything that wants the raw obs)


class ControlLaw(ABC):
    """
    ONE interface for *all* controllers.
    """

    #: Each subclass *must* set this to Input.GUIDANCE_DICT or Input.OBS_VECTOR
    INPUT_TYPE: Input = None  # overwritten by subclasses

    @abstractmethod
    def act(self, data: "np.ndarray | dict") -> np.ndarray:
        """
        Returns body-frame *acceleration command* (shape = (3,) or (4,) etc.,
        depending on the pursuer model).
        `data` is:

        * the **observation vector**     if `INPUT_TYPE is OBS_VECTOR`
        * the **guidance_state dictionary**  otherwise
        """
        ...


def limit_acceleration(acceleration, max_acceleration):
    acceleration_norm = np.linalg.norm(acceleration)
    if acceleration_norm > max_acceleration:
        acceleration = acceleration * (max_acceleration / acceleration_norm)

    return acceleration


class FRPN(ControlLaw):
    INPUT_TYPE = Input.GUIDANCE_DICT

    def __init__(self, lambda_: float, max_acceleration: float, pp_weight: float):
        self.lambda_ = lambda_
        self.max_acceleration = max_acceleration
        self.pp_weight = pp_weight

    def act(self, guidance):
        r = guidance["r"]
        r_dot = guidance["r_dot"]
        t_go = np.linalg.norm(r) / (np.linalg.norm(r_dot) + EPSILON)

        # ! SCALING EXPLAINED. raw is often very large
        # ! raw --> limited. Clipped if exceeds max acceleration and normalized wrt. max acceleration
        # ! limited --> normalized to match API of acceleration pursuer models

        a_raw = self.lambda_ * (
            (1 - self.pp_weight) * ((r + r_dot * t_go) / (t_go * t_go))
            + self.pp_weight * r
        )

        a_limited = limit_acceleration(a_raw, self.max_acceleration)
        return a_limited / (self.max_acceleration + EPSILON)


class RLPolicy(ControlLaw):
    """
    Generic SB3 policy wrapper.
    The *simulation* builds the obs-vector, so we only have to pass it through.
    """

    INPUT_TYPE = Input.OBS_VECTOR  # key line

    def __init__(self, policy_path: str):
        self.pi = PPO.load(policy_path)

    def act(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        obs_vec : shape (obs_dim,) – already assembled by Simulation.
        """
        action, _ = self.pi.predict(obs_vec, deterministic=True)
        return action.squeeze().astype(np.float64)  # whatever dim the net emits
