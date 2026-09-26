"""Run DEAP <-> DREAMER cross-dataset transfer on channel-agnostic band features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.labels import require_verified_ratings

from eeg_emotion_validity.baselines import logistic_baseline, majority_baseline, random_prior_baseline, svm_baseline
from eeg_emotion_validity.signals import DATASET_CHANNELS, DEFAULT_FEATURE_ROOT


DEFAULT_MANIFEST_ROOT = Path("data/processed/torcheeg/manifests")
DEFAULT_OUTPUT_DIR = Path("results/cross_dataset_transfer")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def unique_trials(frame: pd.DataFrame, dataset: str, target: str) -> pd.DataFrame:
    cols = ["subject_id", "trial_id", target]
    out = frame[cols].drop_duplicates(["subject_id", "trial_id"]).reset_index(drop=True)
    threshold = 5.0 if dataset == "deap" else 3.0
    out["target_label"] = np.where(out[target].astype(float) > threshold, f"high_{target}", f"low_{target}")
    return out


def summarize_spectral(features: np.ndarray, dataset: str) -> np.ndarray:
    n_channels = DATASET_CHANNELS[dataset]
    values = features.reshape(features.shape[0], len(BANDS), n_channels)
    return np.concatenate(
        [
            values.mean(axis=2),
            values.std(axis=2),
            values.min(axis=2),
            values.max(axis=2),
        ],
        axis=1,
    )


def trial_features(dataset: str, target: str, manifest_root: Path, feature_root: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = require_verified_ratings(pd.read_csv(manifest_root / f"{dataset}_torcheeg_manifest.csv"), dataset).reset_index(drop=True)
    manifest.insert(0, "row_index", manifest.index)
    features = np.load(feature_root / f"{dataset}_spectral.npy", mmap_mode="r")
    if len(manifest) > features.shape[0]:
        manifest = manifest[manifest["row_index"] < features.shape[0]].copy()

    chunks = summarize_spectral(features[manifest["row_index"].astype(int).to_numpy()], dataset)
    chunk_frame = manifest[["subject_id", "trial_id", target]].copy()
    chunk_frame["feature_row"] = np.arange(len(chunk_frame))

    x_rows = []
    y_rows = []
    threshold = 5.0 if dataset == "deap" else 3.0
    for _, group in chunk_frame.groupby(["subject_id", "trial_id"]):
        idx = group["feature_row"].to_numpy()
        x_rows.append(chunks[idx].mean(axis=0))
        rating = float(group[target].iloc[0])
        y_rows.append(f"high_{target}" if rating > threshold else f"low_{target}")
    return np.vstack(x_rows).astype("float32"), np.asarray(y_rows)


def run_direction(source: str, target_dataset: str, target: str, args: argparse.Namespace) -> list[dict[str, object]]:
    x_train, y_train = trial_features(source, target, args.manifest_root, args.feature_root)
    x_test, y_test = trial_features(target_dataset, target, args.manifest_root, args.feature_root)
    rows = []
    for name, result in [
        ("majority", majority_baseline(y_train, y_test)),
        ("random_prior", random_prior_baseline(y_train, y_test, seed=args.seed)),
        ("transfer_spectral_logistic", logistic_baseline(x_train, y_train, x_test, y_test, name="transfer_spectral_logistic")),
        ("transfer_spectral_svm", svm_baseline(x_train, y_train, x_test, y_test, name="transfer_spectral_svm")),
    ]:
        row = {
            "source_dataset": source,
            "target_dataset": target_dataset,
            "target": target,
            "model": name,
            "train_trials": int(len(y_train)),
            "test_trials": int(len(y_test)),
            **result.metrics,
        }
        rows.append(row)
        print(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["valence", "arousal"], default="valence")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    rows.extend(run_direction("deap", "dreamer", args.target, args))
    rows.extend(run_direction("dreamer", "deap", args.target, args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"deap_dreamer_{args.target}_transfer.csv"
    json_path = args.output_dir / f"deap_dreamer_{args.target}_transfer.json"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"transfer_metrics": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {csv_path} with {len(frame)} rows")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
