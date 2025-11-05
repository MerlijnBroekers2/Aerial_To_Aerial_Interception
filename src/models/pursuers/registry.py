# src/utils/pursuer_builder.py
# ---------------------------------------------------------------------
# 1)  pursuer models
# ---------------------------------------------------------------------
from src.models.pursuers.pursuer_CTBR_INDI import CTBR_INDI_Pursuer
from src.models.pursuers.pursuer_Acc_1order import Acc_1order_Pursuer
from src.models.pursuers.pursuer_Acc_INDI_INDI import Acc_INDI_INDI_Pursuer
from src.models.pursuers.pursuer_Motor import Motor_Pursuer
from src.control_laws.control_laws import FRPN, RLPolicy
from src.utils.logger import get_logger, coerce_level


PURSUER_REGISTRY = {
    "firstorder": Acc_1order_Pursuer,
    "acc_indi": Acc_INDI_INDI_Pursuer,
    "ctbr": CTBR_INDI_Pursuer,
    "motor": Motor_Pursuer,
}

CONTROLLER_REGISTRY = {
    "frpn": FRPN,
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
    log = get_logger("pursuer.registry")
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
    # Emit concise config summary for transparency
    log.info(
        "Using pursuer model=%s, controller=%s, LOG_LEVEL=%s",
        p_cfg.get("MODEL"),
        (p_cfg.get("CONTROLLER", {}) or {}).get("type"),
        p_cfg.get("LOG_LEVEL", global_cfg.get("LOG_LEVEL")),
    )
    # If component overrides log level, apply to this logger name
    comp_level = p_cfg.get("LOG_LEVEL")
    if comp_level is not None:
        import logging as _logging

        get_logger("pursuer.registry").setLevel(
            coerce_level(comp_level, default=_logging.INFO)
        )
    return pursuer
