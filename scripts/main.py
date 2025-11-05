import numpy as np

from src.models.evaders.registry import build_evader
from src.models.pursuers.registry import build_pursuer
from src.simulation.simulation import Simulation
from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger, coerce_level
from src.utils.plotting import (
    animate_pursuit_evasion,
    plot_all_accelerations,
    plot_angular_rates,
    plot_attitude,
    plot_interception_clusters,
    plot_motor_omega,
    plot_range_and_los,
    plot_thrust,
)


def main():
    setup_logging(
        coerce_level(CONFIG.get("LOG_LEVEL", "INFO")), name="drone_interception"
    )
    log = get_logger("main")

    log.info("Global LOG_LEVEL=%s", CONFIG.get("LOG_LEVEL", "INFO"))
    log.info("Building evader model=%s", CONFIG["EVADER"].get("MODEL"))
    evader = build_evader(CONFIG["EVADER"], CONFIG)
    log.info("Building pursuer model=%s", CONFIG["PURSUER"].get("MODEL"))
    pursuer = build_pursuer(CONFIG["PURSUER"], CONFIG)

    sim = Simulation(pursuer, evader, CONFIG)
    log.info(
        "Starting simulation DT=%.4f, TOTAL_TIME=%s, TIME_LIMIT=%s",
        CONFIG["DT"],
        CONFIG.get("TOTAL_TIME"),
        CONFIG.get("TIME_LIMIT"),
    )
    sim.run()

    result = {
        "history": sim.history,
        "interceptions": sim.interceptions,
    }

    # ax = plot_interception_clusters(result)

    animate_pursuit_evasion(
        result,
        pos_offset=0,
        speed_factor=10,
        interval=5,
        # save_file="report_figure.gif",
        save_file=None,
        env_planes=CONFIG["PURSUER"]["BOUNDARIES"]["PLANES"],
        env_bounds_margin=CONFIG["PURSUER"]["BOUNDARIES"]["BOUNDARY_MARGIN"],
    )

    # # Updated plotting functions for new history format:
    # # plot_all_accelerations(result["history"])
    # plot_motor_omega(result["history"])
    # plot_attitude(result["history"])
    # plot_angular_rates(result["history"])
    # plot_thrust(result["history"])
    # plot_range_and_los(result["history"])


if __name__ == "__main__":
    main()
