from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle


@dataclass(frozen=True)
class AnalysisRunConfig:
    n_trials: int
    evader_csv_dir: str
    out_dir: str | None = None
    results_csv: str | None = None
    reuse_seed_per_evader: bool | None = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


