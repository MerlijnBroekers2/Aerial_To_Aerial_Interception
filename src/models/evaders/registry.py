from src.models.evaders.evader import ClassicEvader
from src.models.evaders.moth_evader import MothEvader
from src.models.evaders.reactive_evader import RLEvader
from src.models.evaders.pliska_evader import PliskaEvader
from src.utils.logger import get_logger, coerce_level

EVADER_REGISTRY = {
    "classic": {
        "class": ClassicEvader,
        "required": [
            "PATH_TYPE",
            "VELOCITY_MAGNITUDE",
            "NOISE_STD",
            "DT",
            "TOTAL_TIME",
        ],
    },
    "moth": {
        "class": MothEvader,
        "required": [],
    },
    "pliska": {
        "class": PliskaEvader,
        "required": [],
    },
    "rl": {
        "class": RLEvader,
        "required": ["RL_MODEL_PATH"],
    },
}


def build_evader(config: dict, global_config: dict) -> object:
    log = get_logger("evader.registry")
    evader_type = config["MODEL"].lower()
    if evader_type not in EVADER_REGISTRY:
        raise ValueError(f"Unknown evader type: {evader_type}")

    entry = EVADER_REGISTRY[evader_type]
    cls = entry["class"]

    # Special case: pass full global config for these
    if evader_type in {"moth", "pliska"}:
        obj = cls(global_config)
        log.info(
            "Using evader type=%s, LOG_LEVEL=%s",
            config.get("MODEL"),
            config.get("LOG_LEVEL", global_config.get("LOG_LEVEL")),
        )
        comp_level = config.get("LOG_LEVEL")
        if comp_level is not None:
            import logging as _logging

            get_logger("evader.registry").setLevel(
                coerce_level(comp_level, default=_logging.INFO)
            )
        return obj

    if evader_type == "rl":
        rl_model_path = config.get("RL_MODEL_PATH")
        if rl_model_path is None:
            raise ValueError("RL_MODEL_PATH is required for RL evader.")
        obj = cls(global_config, rl_model_path)
        log.info(
            "Using evader type=rl, RL_MODEL_PATH set, LOG_LEVEL=%s",
            config.get("LOG_LEVEL", global_config.get("LOG_LEVEL")),
        )
        comp_level = config.get("LOG_LEVEL")
        if comp_level is not None:
            import logging as _logging

            get_logger("evader.registry").setLevel(
                coerce_level(comp_level, default=_logging.INFO)
            )
        return obj

    # Standard case
    required_keys = entry["required"]
    kwargs = {}
    for key in required_keys:
        if key in config:
            kwargs[key.lower()] = config[key]
        elif key in global_config:
            kwargs[key.lower()] = global_config[key]
        else:
            raise ValueError(
                f"Missing required key '{key}' for evader type '{evader_type}'"
            )

    obj = cls(**kwargs)
    log.info(
        "Using evader type=%s with required keys=%s, LOG_LEVEL=%s",
        config.get("MODEL"),
        list(kwargs.keys()),
        config.get("LOG_LEVEL", global_config.get("LOG_LEVEL")),
    )
    comp_level = config.get("LOG_LEVEL")
    if comp_level is not None:
        import logging as _logging

        get_logger("evader.registry").setLevel(
            coerce_level(comp_level, default=_logging.INFO)
        )
    return obj
