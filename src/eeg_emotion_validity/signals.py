"""TorchEEG cache helpers shared by signal baseline scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eeg_emotion_validity.features import alpha_power_features, spectral_features


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_TORCHEEG_ROOT = DEFAULT_DATA_ROOT / "processed" / "torcheeg"
DEFAULT_SPLIT_ROOT = DEFAULT_DATA_ROOT / "processed" / "splits" / "torcheeg"
DEFAULT_FEATURE_ROOT = DEFAULT_DATA_ROOT / "processed" / "features" / "bandpower"
DEFAULT_RUN_ROOT = Path("runs")

DEFAULT_DEAP_ROOT = DEFAULT_DATA_ROOT / "raw" / "deap" / "data_preprocessed_python"
DEFAULT_DREAMER_MAT = DEFAULT_DATA_ROOT / "raw" / "dreamer" / "DREAMER.mat"
DEFAULT_SEED_IV_ROOT = DEFAULT_DATA_ROOT / "raw" / "seed_iv" / "eeg_raw_data"

DATASET_SFREQ = {"deap": 128.0, "dreamer": 128.0, "seed_iv": 200.0}
DATASET_CHANNELS = {"deap": 32, "dreamer": 14, "seed_iv": 62}
DATASET_CHUNK_SIZE = {"deap": 128, "dreamer": 128, "seed_iv": 800}
DATASET_IO_NAME = {
    "deap": "deap_chunk128_overlap0",
    "dreamer": "dreamer_chunk128_overlap0",
    "seed_iv": "seed_iv_chunk800_overlap0",
}

SPECTRAL_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def make_torcheeg_dataset(
    dataset: str,
    torcheeg_root: Path = DEFAULT_TORCHEEG_ROOT,
    io_mode: str = "lmdb",
    deap_root: Path = DEFAULT_DEAP_ROOT,
    dreamer_mat: Path = DEFAULT_DREAMER_MAT,
    seed_iv_root: Path = DEFAULT_SEED_IV_ROOT,
) -> Any:
    io_path = torcheeg_root / DATASET_IO_NAME[dataset]
    if dataset == "deap":
        from torcheeg.datasets import DEAPDataset

        return DEAPDataset(
            root_path=str(deap_root),
            chunk_size=DATASET_CHUNK_SIZE[dataset],
            overlap=0,
            num_channel=DATASET_CHANNELS[dataset],
            io_path=str(io_path),
            io_mode=io_mode,
            num_worker=0,
            verbose=False,
        )
    if dataset == "dreamer":
        from torcheeg.datasets import DREAMERDataset

        return DREAMERDataset(
            mat_path=str(dreamer_mat),
            chunk_size=DATASET_CHUNK_SIZE[dataset],
            overlap=0,
            num_channel=DATASET_CHANNELS[dataset],
            io_path=str(io_path),
            io_mode=io_mode,
            num_worker=0,
            verbose=False,
        )
    if dataset == "seed_iv":
        from torcheeg.datasets import SEEDIVDataset

        return SEEDIVDataset(
            root_path=str(seed_iv_root),
            chunk_size=DATASET_CHUNK_SIZE[dataset],
            overlap=0,
            num_channel=DATASET_CHANNELS[dataset],
            io_path=str(io_path),
            io_mode=io_mode,
            num_worker=0,
            verbose=False,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def signal_from_sample(sample: Any) -> np.ndarray:
    x = sample[0] if isinstance(sample, (tuple, list)) else sample
    if isinstance(x, dict):
        for key in ("eeg", "signal", "x", "data"):
            if key in x:
                x = x[key]
                break
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    return x


def feature_vector(signal: np.ndarray, sfreq: float, feature_set: str) -> np.ndarray:
    if feature_set == "alpha":
        values = alpha_power_features(signal, sfreq)
    elif feature_set == "spectral":
        values = spectral_features(signal, sfreq, SPECTRAL_BANDS)
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


def feature_names(dataset: str, feature_set: str) -> list[str]:
    channels = DATASET_CHANNELS[dataset]
    if feature_set == "alpha":
        return [f"alpha_ch{idx:03d}" for idx in range(channels)]
    if feature_set == "spectral":
        return [
            f"{band}_ch{idx:03d}"
            for band in SPECTRAL_BANDS
            for idx in range(channels)
        ]
    raise ValueError(f"Unsupported feature set: {feature_set}")
