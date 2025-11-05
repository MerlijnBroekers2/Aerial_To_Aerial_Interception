import numpy as np
import pandas as pd


def bootstrap_ci_mean(x, iters: int = 2000, alpha: float = 0.05, seed: int | None = None):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(x)), (lo, hi)


def bootstrap_ci_rate(flags, iters: int = 2000, alpha: float = 0.05, seed: int | None = None):
    f = np.asarray(pd.to_numeric(flags, errors="coerce"), dtype=float)
    f = f[~np.isnan(f)]
    n = f.size
    if n == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = f[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(f.mean()), (lo, hi)


