# src/utils/run_naming.py
# ------------------------------------------------------------
"""
Create a descriptive run-prefix from a CONFIG dict.

Fields included
---------------
PursuerClass  EvaderClass  DT  TIME_LIMIT  CAPTURE_RADIUS
ENV_BOUND  OBS_MODE  MAX_HISTORY_STEPS  HISTORY_STEPS
ACTION_HISTORY_STEPS  reward_type  DR_Percentage
"""
from __future__ import annotations
from typing import Dict


def _format_float(x: float) -> str:
    """Turns 0.01 → '0p01', 3    → '3' (no trailing zeros)."""
    if int(x) == x:
        return str(int(x))
    return str(x).replace(".", "p")


def _get_dr_percentage(cfg: Dict) -> str:
    """
    Aggregate the domain-randomisation percentages from the pursuer block.
    For now: return '0' when all values are 0.  Change logic later as needed.
    """
    dr = cfg["PURSUER"].get("domain_randomization_pct", {})
    max_val = max(dr.values()) if dr else 0.0
    return _format_float(max_val * 100)  # express as percent


def build_run_prefix(cfg: Dict) -> str:
    obs_cfg = cfg["OBSERVATIONS"]

    parts = [
        cfg["PURSUER"]["MODEL"].capitalize(),
        cfg["EVADER"]["MODEL"].capitalize(),
        f"dt{_format_float(cfg['DT'])}",
        f"T{cfg['TIME_LIMIT']}",
        f"CapRadius{_format_float(cfg['CAPTURE_RADIUS'])}",
        f"Bound{_format_float(cfg['ENV_BOUND'])}",
        obs_cfg["OBS_MODE"],
        f"His{obs_cfg['HISTORY_STEPS']}",
        f"ActHis{obs_cfg['ACTION_HISTORY_STEPS']}",
        cfg["reward_type"],
        f"DR{_get_dr_percentage(cfg)}",
    ]
    return "_".join(parts)
