"""Interval and paired-effect helpers with explicit units."""

from __future__ import annotations

import math

import numpy as np


def t_interval(values, confidence: float = 0.95) -> dict[str, float]:
    """Student-t interval for the mean of independent units (ddof=1).

    With n=5 seeds the multiplier is t_{0.975,4} ~= 2.776, not the normal 1.96.
    For fewer than two finite units the interval is undefined (NaN), never zero-width.
    """
    from scipy import stats

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    mean = float(array.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "sd": float("nan"), "multiplier": float("nan"), "low": float("nan"), "high": float("nan")}
    sd = float(array.std(ddof=1))
    multiplier = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    half = multiplier * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "multiplier": multiplier, "low": mean - half, "high": mean + half}


def paired_differences(a: dict, b: dict) -> np.ndarray:
    """a[k] - b[k] for identical key sets; mismatched pairings are rejected, not dropped."""
    if set(a) != set(b):
        missing = sorted(set(a) ^ set(b))[:5]
        raise ValueError(f"Paired comparison requires identical units; unmatched keys e.g. {missing}")
    keys = sorted(a)
    return np.asarray([float(a[k]) - float(b[k]) for k in keys], dtype=float)


def cohens_dz(differences) -> float:
    """d_z = mean(paired differences) / sd(paired differences), ddof=1."""
    diff = np.asarray(differences, dtype=float)
    if diff.size < 2:
        return float("nan")
    sd = float(diff.std(ddof=1))
    return float(diff.mean() / sd) if sd > 0 else float("nan")
