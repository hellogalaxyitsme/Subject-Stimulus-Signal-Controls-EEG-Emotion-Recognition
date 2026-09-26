"""Run DEAP <-> DREAMER spectral transfer with z-score and CORAL alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.labels import require_verified_ratings

from eeg_emotion_validity.baselines import classification_metrics
from eeg_emotion_validity.signals import DATASET_CHANNELS, DEFAULT_FEATURE_ROOT
from eeg_emotion_validity.transfer import assert_transfer_allowed


DEFAULT_MANIFEST_ROOT = Path("data/processed/torcheeg/manifests")
DEFAULT_OUTPUT_DIR = Path("results/aligned_cross_dataset_transfer")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

DEAP_CHANNELS = [
    "FP1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "PZ",
    "PO3",
    "O1",
    "OZ",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "FP2",
    "FZ",
    "CZ",
]
DREAMER_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
SHARED_CHANNELS = DREAMER_CHANNELS
CHANNELS_BY_DATASET = {"deap": DEAP_CHANNELS, "dreamer": DREAMER_CHANNELS}


def shared_channel_indices(dataset: str) -> list[int]:
    lookup = {name.upper(): index for index, name in enumerate(CHANNELS_BY_DATASET[dataset])}
    return [lookup[name.upper()] for name in SHARED_CHANNELS]


def label_threshold(dataset: str) -> float:
    if dataset == "deap":
        return 5.0
    if dataset == "dreamer":
        return 3.0
    raise ValueError(f"Unsupported transfer dataset: {dataset}")


def load_trial_features(
    dataset: str,
    target: str,
    manifest_root: Path,
    features_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = manifest_root / f"{dataset}_torcheeg_manifest.csv"
    feature_path = features_dir / f"{dataset}_spectral.npy"
    manifest = require_verified_ratings(pd.read_csv(manifest_path), dataset).reset_index(drop=True)
    manifest.insert(0, "row_index", manifest.index.astype(int))
    if target not in manifest.columns:
        raise ValueError(f"{target!r} is absent from {manifest_path}")
    features = np.load(feature_path, mmap_mode="r")
    manifest = manifest[manifest["row_index"] < features.shape[0]].copy()
    manifest = manifest[manifest[target].notna()].copy()

    n_channels = DATASET_CHANNELS[dataset]
    channel_indices = shared_channel_indices(dataset)
    row_indices = manifest["row_index"].astype(int).to_numpy()
    spectral = features[row_indices].reshape(len(row_indices), len(BANDS), n_channels)
    shared = spectral[:, :, channel_indices].reshape(len(row_indices), -1)
    chunk_frame = manifest[["subject_id", "trial_id", target]].copy()
    chunk_frame["feature_row"] = np.arange(len(chunk_frame))

    x_rows = []
    y_rows = []
    threshold = label_threshold(dataset)
    for _, group in chunk_frame.groupby(["subject_id", "trial_id"], sort=True):
        idx = group["feature_row"].to_numpy()
        x_rows.append(shared[idx].mean(axis=0))
        rating = float(group[target].iloc[0])
        y_rows.append(f"high_{target}" if rating > threshold else f"low_{target}")
    return np.vstack(x_rows).astype("float32"), np.asarray(y_rows, dtype=str)


def zscore_independent(x_source: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean = x_source.mean(axis=0, keepdims=True)
    target_mean = x_target.mean(axis=0, keepdims=True)
    source_std = x_source.std(axis=0, keepdims=True) + 1e-6
    target_std = x_target.std(axis=0, keepdims=True) + 1e-6
    return (x_source - source_mean) / source_std, (x_target - target_mean) / target_std


def symmetric_matrix_power(matrix: np.ndarray, power: float, eps: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eps)
    return (vectors * (values**power)) @ vectors.T


def coral_align_source(x_source: np.ndarray, x_target: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    source_mean = x_source.mean(axis=0, keepdims=True)
    target_mean = x_target.mean(axis=0, keepdims=True)
    source_centered = x_source - source_mean
    target_centered = x_target - target_mean
    source_cov = np.cov(source_centered, rowvar=False) + eps * np.eye(x_source.shape[1])
    target_cov = np.cov(target_centered, rowvar=False) + eps * np.eye(x_target.shape[1])
    whitening = symmetric_matrix_power(source_cov, -0.5, eps)
    recoloring = symmetric_matrix_power(target_cov, 0.5, eps)
    return (source_centered @ whitening @ recoloring + target_mean).astype("float32")


def fit_predict_logistic(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
    )
    clf.fit(x_train, y_train)
    return clf.predict(x_test)


def metric_row(
    source_dataset: str,
    target_dataset: str,
    target: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_trials: int,
    test_trials: int,
) -> dict[str, object]:
    metrics = classification_metrics(y_true, y_pred)
    return {
        "direction": f"{source_dataset}_to_{target_dataset}",
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "target": target,
        "method": method,
        "train_trials": int(train_trials),
        "test_trials": int(test_trials),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
    }


def run_direction(args: argparse.Namespace, target: str) -> list[dict[str, object]]:
    assert_transfer_allowed(args.source_dataset, args.target_dataset, target)
    x_source, y_source = load_trial_features(args.source_dataset, target, args.manifest_root, args.features_dir)
    x_target, y_target = load_trial_features(args.target_dataset, target, args.manifest_root, args.features_dir)

    rows = []
    transforms = {
        "raw_spectral": (x_source, x_target),
        "znorm_spectral": zscore_independent(x_source, x_target),
        "coral_spectral": (coral_align_source(x_source, x_target, eps=args.coral_eps), x_target),
    }
    for method, (x_train, x_test) in transforms.items():
        y_pred = fit_predict_logistic(x_train, y_source, x_test, args.seed)
        row = metric_row(
            args.source_dataset,
            args.target_dataset,
            target,
            method,
            y_target,
            y_pred,
            len(y_source),
            len(y_target),
        )
        print(row)
        rows.append(row)
    return rows


def parse_targets(value: str) -> list[str]:
    targets = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    invalid = sorted(set(targets) - {"valence", "arousal"})
    if invalid:
        raise ValueError(f"Unsupported transfer targets: {invalid}")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", choices=["deap", "dreamer"], required=True)
    parser.add_argument("--target-dataset", choices=["deap", "dreamer"], required=True)
    parser.add_argument("--targets", default="valence arousal")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--coral-eps", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_dataset == args.target_dataset:
        raise ValueError("source-dataset and target-dataset must differ")
    rows: list[dict[str, object]] = []
    for target in parse_targets(args.targets):
        rows.extend(run_direction(args, target))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.source_dataset}_to_{args.target_dataset}_aligned_transfer"
    csv_path = args.output_csv or args.output_dir / f"{stem}.csv"
    json_path = args.output_json or args.output_dir / f"{stem}.json"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps({"aligned_transfer_metrics": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path} with {len(frame)} rows")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
