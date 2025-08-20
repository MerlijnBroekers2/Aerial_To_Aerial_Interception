from src.evaders.moth_evader import MothEvader
from src.evaders.pliska_evader import PliskaEvader  # ✅ New import

EVADER_REGISTRY = {
    "moth": {
        "class": MothEvader,
        "required": [],
    },
    "pliska": {
        "class": PliskaEvader,
        "required": [],
    },
}


def build_evader(config: dict, global_config: dict) -> object:
    evader_type = config["MODEL"].lower()
    if evader_type not in EVADER_REGISTRY:
        raise ValueError(f"Unknown evader type: {evader_type}")

    entry = EVADER_REGISTRY[evader_type]
    cls = entry["class"]

    # Special case: pass full global config for these
    if evader_type in {"moth", "pliska"}:
        return cls(global_config)

    if evader_type == "rl":
        rl_model_path = config.get("RL_MODEL_PATH")
        if rl_model_path is None:
            raise ValueError("RL_MODEL_PATH is required for RL evader.")
        return cls(rl_model_path, global_config)

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

    return cls(**kwargs)
