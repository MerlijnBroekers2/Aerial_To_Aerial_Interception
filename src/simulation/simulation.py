import numpy as np
from src.control_laws.control_laws import Input
from src.utils.config import CONFIG
from src.utils.observations import ObservationManager


class Simulation:
    def __init__(
        self,
        pursuer,
        evader,
        config,
    ):
        self.pursuer = pursuer
        self.evader = evader
        self.total_time = config["TOTAL_TIME"]
        self.dt = config["DT"]
        self.time = 0.0
        self.history = []  # Holds dicts with pursuer, evader, and guidance states
        self.interceptions = []
        self.Vc_initial = None
        self.stop_on_interception = config["STOP_ON_INTERCEPTION"]
        self.interception_radius = config["INTERCEPTION_RADIUS"]
        self.obs_mgr = ObservationManager(config)

    def run(self):
        self.evader.reset()
        self.pursuer.reset()
        self.obs_mgr.reset(batch=1)

        while self.time <= self.total_time:
            # Get states
            p_state = self.pursuer.get_state()
            e_state = self.evader.get_state()

            # Advance evader
            self.evader.step(
                pursuer_pos=p_state["noisy_position"],
                pursuer_vel=p_state["noisy_velocity"],
            )

            # Compute guidance state
            guidance_state = self.compute_guidance_state(
                evader_state=e_state, pursuer_state=p_state
            )
            obs_vector = self.obs_mgr.build(e_state, p_state)

            if self.Vc_initial is None:
                self.Vc_initial = guidance_state["Vc_current"]
            guidance_state["Vc_initial"] = self.Vc_initial

            # Step pursuer
            ctrl = self.pursuer.control_law
            if ctrl.INPUT_TYPE is Input.GUIDANCE_DICT:
                accel_cmd = ctrl.act(guidance_state)
            else:  # OBS_VECTOR
                accel_cmd = ctrl.act(obs_vector)

            self.pursuer.step(accel_cmd)

            base_dim = self.obs_mgr.base_dim
            self.obs_mgr.push(obs_vector[:, :base_dim], accel_cmd[None, :])

            # Save minimal state bundle
            self.history.append(
                {
                    "time": self.time,
                    "p_state": p_state,
                    "e_state": e_state,
                    "guidance_state": guidance_state,
                    "action": np.asarray(accel_cmd, dtype=float).ravel(),
                }
            )

            # Check interception
            if (
                self.check_interception(
                    evader_pos=e_state["true_position"],
                    pursuer_pos=p_state["true_position"],
                )
                and self.stop_on_interception
            ):
                break

            self.time += self.dt

    def compute_guidance_state(self, evader_state, pursuer_state):
        r = evader_state["filtered_position"] - pursuer_state["noisy_position"]
        R = np.linalg.norm(r)
        Ir = r / R if R > 0 else np.zeros_like(r)
        r_dot = evader_state["filtered_velocity"] - pursuer_state["noisy_velocity"]
        Vc_current = np.linalg.norm(r_dot)
        phi_dot = np.cross(Ir, r_dot) / (R**2) if R > 0 else np.zeros_like(r_dot)

        return {
            # Positions and velocities
            "e_pos": evader_state["filtered_position"],
            "e_vel": evader_state["filtered_velocity"],
            "p_pos": pursuer_state["noisy_position"],
            "p_vel": pursuer_state["noisy_velocity"],
            # Relative geometry
            "r": r,
            "R": R,
            "Ir": Ir,
            "r_dot": r_dot,
            "Vc_current": Vc_current,
            "phi_dot": phi_dot,
            # Time step
            "dt": self.dt,
            # Additional pursuer parameters
            "p_attitude": pursuer_state["attitude"],
            "p_rates": pursuer_state["rates"],
            "T_force": pursuer_state["T_force"],
            "omega": pursuer_state["omega"],
            "omega_norm": pursuer_state["omega_norm"],
        }

    def check_interception(self, evader_pos, pursuer_pos):
        distance = np.linalg.norm(evader_pos - pursuer_pos)
        if distance < self.interception_radius:
            self.interceptions.append(
                {
                    "time": self.time,
                    "idx": len(self.history) - 1,
                    "pursuer_pos": np.copy(pursuer_pos),
                    "evader_pos": np.copy(evader_pos),
                }
            )
            return True
        return False
