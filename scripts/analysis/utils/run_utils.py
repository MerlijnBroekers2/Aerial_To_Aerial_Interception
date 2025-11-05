from __future__ import annotations

from pathlib import Path
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Any

import numpy as np
import pandas as pd

from src.simulation.simulation import Simulation


def simulate_and_measure(pursuer, evader, cfg, compute_metrics_fn: Callable):
    sim = Simulation(pursuer, evader, cfg)
    sim.run()
    metrics = compute_metrics_fn(
        [{"history": sim.history, "interceptions": sim.interceptions}]
    )[0]["metrics"]
    metrics["interception_flag"] = int(
        np.isfinite(metrics.get("first_interception_time", np.nan))
    )
    return metrics, sim


@dataclass
class Variant:
    key: str
    meta: dict


def list_evaders(evader_csv_dir: str, extra_fields: dict | None = None) -> list[dict]:
    """
    List moth CSV files in a directory as evader configs.
    extra_fields, if provided, is merged into each evader dict (e.g. {"NOISE_STD": 0.0}).
    """
    extra_fields = extra_fields or {}
    rows: list[dict] = []
    for csv in sorted(Path(evader_csv_dir).glob("*.csv")):
        row = {"MODEL": "moth", "CSV_FILE": str(csv), "label": csv.name}
        row.update(extra_fields)
        rows.append(row)
    return rows


def run_sweep_variants(
    *,
    evaders: list[dict],
    variants: Iterable[Variant],
    n_trials: int,
    build_cfg_fn: Callable[[dict, str, dict], dict],
    build_agents_fn: Callable[[dict, str, dict], tuple],
    compute_metrics_fn: Callable,
    extra_fields_fn: Callable[[dict, str, dict, dict, Any], dict] | None = None,
    per_evader_seed_bank: bool = False,
) -> pd.DataFrame:
    """
    Generic sweep over (evader × variant × trials).

    - build_cfg_fn(evader_cfg, variant_key, variant_meta) -> cfg dict to use for this trial
    - build_agents_fn(cfg, variant_key, variant_meta) -> (pursuer, evader)
    - compute_metrics_fn(history_bundle) -> metrics dict (uses project helper)
    - extra_fields_fn(evader_cfg, variant_key, variant_meta, metrics) -> dict of extra columns
    """
    rows: list[dict] = []

    # Optional per-evader seed bank for fairness across variants
    seed_bank: dict[str, list[int]] | None = None
    if per_evader_seed_bank:
        rng = np.random.default_rng(12345)
        seed_bank = {
            ev["label"]: rng.integers(0, 2**31 - 1, size=n_trials).tolist()
            for ev in evaders
        }

    for variant in variants:
        vkey, vmeta = variant.key, (variant.meta or {})
        print(f"\n===== variant: {vkey} =====")
        for ev in evaders:
            print(f"evader: {ev.get('label', 'no_label')}  ({n_trials} trials)")
            seeds = (seed_bank or {}).get(ev.get("label", ""), [None] * n_trials)
            for t in range(n_trials):
                seed = seeds[t]
                if seed is None:
                    seed = random.randint(0, 2**32 - 1)
                random.seed(seed)
                np.random.seed(seed)

                print(f"     trial {t+1}/{n_trials}")

                cfg = build_cfg_fn(ev, vkey, vmeta)
                pursuer, evader = build_agents_fn(cfg, vkey, vmeta)
                metrics, sim = simulate_and_measure(
                    pursuer, evader, cfg, compute_metrics_fn
                )

                row = {"evader": ev.get("label", ""), **metrics}
                if extra_fields_fn is not None:
                    row.update(extra_fields_fn(ev, vkey, vmeta, metrics, sim))
                rows.append(row)

    return pd.DataFrame(rows)
