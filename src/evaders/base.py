from abc import ABC, abstractmethod
import numpy as np


class IEvader(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Back to t = 0 (or initial state)"""

    @abstractmethod
    def step(self, **kwargs) -> None:
        """
        Advance one time‐step.
        - Trajectory‐based evaders ignore all kwargs.
        - RL evader pulls out pursuer_pos, pursuer_vel from kwargs.
        - Vectorized evader expects accel_cmd in kwargs.
        """

    @abstractmethod
    def get_state(self) -> dict:
        """Returns dict with keys:
        'true_position', 'noisy_position', 'velocity', 'acceleration'."""
