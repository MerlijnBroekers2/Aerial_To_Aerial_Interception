# src/utils/pursuer_builder.py
# ---------------------------------------------------------------------
# 1)  pursuer models
# ---------------------------------------------------------------------
from src.pursuers.pursuer_Motor import Motor_Pursuer
from src.utils.control_laws import RLPolicy


PURSUER_REGISTRY = {
    "motor": Motor_Pursuer,
}

CONTROLLER_REGISTRY = {
    "rl": RLPolicy,
}


def _make_controller(ctrl_cfg: dict | None, p_cfg: dict):
    if ctrl_cfg is None:
        return None

    kind = ctrl_cfg["type"].lower()
    cls = CONTROLLER_REGISTRY[kind]

    if kind == "rl":  # RLPolicy signature
        return cls(ctrl_cfg["policy_path"])

    # classic PN-style controllers
    params = ctrl_cfg.get("params", {})
    params.setdefault("max_acceleration", p_cfg.get("MAX_ACCELERATION"))
    return cls(**params)


def build_pursuer(p_cfg: dict, global_cfg: dict):
    pursuer_cls = PURSUER_REGISTRY[p_cfg["MODEL"].lower()]

    kwargs = {
        "config": {
            "DT": global_cfg["DT"],
            "PURSUER": p_cfg,
        }
    }

    ctrl = _make_controller(p_cfg.get("CONTROLLER"), p_cfg)
    if ctrl is not None:
        kwargs["control_law"] = ctrl

    pursuer = pursuer_cls(**kwargs)
    pursuer.control_law = ctrl
    return pursuer
