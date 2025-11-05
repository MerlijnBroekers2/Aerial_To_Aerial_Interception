from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class IPursuer(ABC):
    """
    All pursuer models – single-agent, vectorised, RL-controlled, etc. –
    must inherit from this interface so the simulation can treat them the same
    way the evader system already does.
    """

    # --- life-cycle ---------------------------------------------------------
    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """
        Bring the pursuer *or the indexed subset of a batched pursuer*
        back to its initial state.
        Single-agent models may ignore the arguments.
        """

    # --- dynamics -----------------------------------------------------------
    @abstractmethod
    def step(self, **kwargs) -> np.ndarray:
        """
        Advance one time step **and return the acceleration actually commanded**.
        Typical kwargs:
            guidance_state – output of Simulation.compute_guidance_state()
            command
        """

    # --- read-only state ----------------------------------------------------
    @abstractmethod
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Must return a dict containing at least
        'true_position', 'noisy_position', 'velocity', 'acceleration', 'attitude'.

        Vectorised pursuers return arrays of shape (N, 3);
        single-agent pursuers return shape (3,).
        """
