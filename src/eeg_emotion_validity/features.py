"""Feature extraction utilities for baseline models."""

from __future__ import annotations

import numpy as np


def bandpower(signal, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Compute mean Welch power in a frequency band.

    Input shape is expected to be `(n_samples, n_channels, n_times)` or
    `(n_channels, n_times)`. Output is one value per channel.
    """

    from scipy.signal import welch

    x = np.asarray(signal, dtype=float)
    single = x.ndim == 2
    if single:
        x = x[None, ...]
    freqs, psd = welch(x, fs=sfreq, axis=-1, nperseg=min(x.shape[-1], int(sfreq * 2)))
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        raise ValueError(f"Band {band} is empty for sampling frequency {sfreq}")
    values = psd[..., mask].mean(axis=-1)
    return values[0] if single else values


def alpha_power_features(signal, sfreq: float, alpha_band=(8.0, 13.0)) -> np.ndarray:
    return bandpower(signal, sfreq, alpha_band)


def spectral_features(signal, sfreq: float, bands: dict[str, tuple[float, float]]) -> np.ndarray:
    parts = [bandpower(signal, sfreq, band) for band in bands.values()]
    return np.concatenate(parts, axis=-1)


def temporal_shuffle_within_trials(frame, seed: int = 20260530):
    """Return a metadata frame with segment order shuffled within each trial."""

    rng = np.random.default_rng(seed)
    shuffled = frame.copy()
    group_cols = ["dataset", "subject_id", "session_id", "trial_id"]
    for _, idx in shuffled.groupby(group_cols).groups.items():
        segment_values = shuffled.loc[idx, "segment_id"].to_numpy()
        rng.shuffle(segment_values)
        shuffled.loc[idx, "segment_id"] = segment_values
    return shuffled

