import numpy as np

from src.evaders.registry import build_evader
from src.pursuers.registry import build_pursuer
from src.simulation.simulation import Simulation
from src.utils.config import CONFIG
from src.utils.metric_computation import compute_simulation_metrics
from src.utils.plotting import (
    animate_pursuit_evasion,
)


def main():
    evader = build_evader(CONFIG["EVADER"], CONFIG)
    pursuer = build_pursuer(CONFIG["PURSUER"], CONFIG)

    sim = Simulation(pursuer, evader, CONFIG)
    sim.run()

    result = {
        "history": sim.history,
        "interceptions": sim.interceptions,
    }

    metrics = compute_simulation_metrics(
        [{"history": sim.history, "interceptions": sim.interceptions}]
    )[0]["metrics"]

    print(metrics)

    animate_pursuit_evasion(
        result, pos_offset=0, speed_factor=10, interval=5, save_file=None
    )


if __name__ == "__main__":
    main()
