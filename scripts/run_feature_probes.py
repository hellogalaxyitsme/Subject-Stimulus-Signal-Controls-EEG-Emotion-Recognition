"""Run linear confound probes on cached bandpower features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.baselines import classification_metrics
from eeg_emotion_validity.signals import DEFAULT_FEATURE_ROOT


DEFAULT_MANIFEST_ROOT = Path("data/processed/torcheeg/manifests")
DEFAULT_OUTPUT_DIR = Path("results/confound_probes")


def load_manifest(path: Path, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(path).reset_index(drop=True)
    frame.insert(0, "row_index", frame.index)
    frame["dataset"] = dataset
    if dataset == "deap":
        frame["subject_key"] = frame["subject_id"].astype(str)
        frame["stimulus_key"] = "trial_" + frame["trial_id"].astype(str)
        frame["session_key"] = "session_1"
    elif dataset == "dreamer":
        frame["subject_key"] = "s" + (frame["subject_id"].astype(int) + 1).astype(str).str.zfill(2)
        frame["stimulus_key"] = "trial_" + frame["trial_id"].astype(str)
        frame["session_key"] = "session_1"
    elif dataset == "seed_iv":
        frame["subject_key"] = "s" + frame["subject_id"].astype(int).astype(str).str.zfill(2)
        frame["stimulus_key"] = (
            "session_" + frame["session_id"].astype(str) + "::trial_" + frame["trial_id"].astype(str)
        )
        frame["session_key"] = "session_" + frame["session_id"].astype(str)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    frame["trial_key"] = (
        frame["subject_key"].astype(str) + "::" + frame["session_key"].astype(str) + "::trial_" + frame["trial_id"].astype(str)
    )
    return frame


def feature_path(feature_root: Path, dataset: str, feature_set: str) -> Path:
    return feature_root / f"{dataset}_{feature_set}.npy"


def sample_probe_frame(frame: pd.DataFrame, target_col: str, max_rows: int, seed: int, min_class_count: int) -> pd.DataFrame:
    valid_labels = frame[target_col].value_counts()
    valid_labels = valid_labels[valid_labels >= min_class_count].index
    frame = frame[frame[target_col].isin(valid_labels)].copy()
    if len(frame) <= max_rows:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    per_class = max(min_class_count, max_rows // max(1, frame[target_col].nunique()))
    parts = [
        group.sample(n=min(len(group), per_class), random_state=seed)
        for _, group in frame.groupby(target_col)
    ]
    sampled = pd.concat(parts)
    if len(sampled) < max_rows:
        remaining = frame.drop(sampled.index)
        fill = remaining.sample(n=min(len(remaining), max_rows - len(sampled)), random_state=seed)
        sampled = pd.concat([sampled, fill])
    elif len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_probe_grouped(features: np.ndarray, frame: pd.DataFrame, target_col: str, seed: int) -> dict[str, object]:
    """Closed-set probe with a group-safe split (see eeg_emotion_validity.probes.default_query)."""
    from eeg_emotion_validity.probes import default_query, probe_one_encoder

    frame = frame.reset_index(drop=True)
    x = np.asarray(features[frame["row_index"].astype(int).to_numpy()])
    query = default_query(target_col, seed)
    row = probe_one_encoder(x, frame, query)
    row.update({"target": target_col, "split_mode": query.split, "n_rows": int(len(frame))})
    return row


def run_probe(features: np.ndarray, frame: pd.DataFrame, target_col: str, seed: int) -> dict[str, object]:
    """Window-random stratified split; windows of one trial can fall on both sides (for comparison only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = frame[target_col].astype(str).to_numpy()
    row_indices = frame["row_index"].astype(int).to_numpy()
    x = features[row_indices]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = classification_metrics(y_test, pred)
    return {
        "target": target_col,
        "n_classes": int(pd.Series(labels).nunique()),
        "chance_accuracy": float(1.0 / pd.Series(labels).nunique()),
        "n_rows": int(len(frame)),
        **metrics,
    }


def run_dataset(args: argparse.Namespace, dataset: str) -> list[dict[str, object]]:
    manifest = load_manifest(args.manifest_root / f"{dataset}_torcheeg_manifest.csv", dataset)
    features = np.load(feature_path(args.feature_root, dataset, args.feature_set), mmap_mode="r")
    if len(manifest) > features.shape[0]:
        manifest = manifest[manifest["row_index"] < features.shape[0]].copy()

    targets = ["subject_key", "stimulus_key"]
    if manifest["session_key"].nunique() > 1:
        targets.append("session_key")

    rows = []
    for target in targets:
        probe_frame = sample_probe_frame(
            manifest,
            target,
            max_rows=args.max_rows,
            seed=args.seed,
            min_class_count=args.min_class_count,
        )
        if probe_frame[target].nunique() < 2:
            continue
        if args.split_mode == "trial_grouped":
            result = run_probe_grouped(features, probe_frame, target, args.seed)
        else:
            result = run_probe(features, probe_frame, target, args.seed)
            result["split_mode"] = "window_random"
        result.update({"dataset": dataset, "feature_set": args.feature_set})
        rows.append(result)
        print(result)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "deap", "dreamer", "seed_iv"], default="all")
    parser.add_argument("--feature-set", choices=["alpha", "spectral"], default="spectral")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument(
        "--split-mode",
        choices=["trial_grouped", "window_random"],
        default="trial_grouped",
        help="trial_grouped keeps every trial on one side; window_random splits windows at random (for comparison only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["deap", "dreamer", "seed_iv"] if args.dataset == "all" else [args.dataset]
    rows = []
    for dataset in datasets:
        rows.extend(run_dataset(args, dataset))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    suffix = "" if args.split_mode == "window_random" else f"_{args.split_mode}"
    csv_path = args.output_dir / f"bandpower_{args.feature_set}_confound_probes{suffix}.csv"
    json_path = args.output_dir / f"bandpower_{args.feature_set}_confound_probes{suffix}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"probes": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {csv_path} with {len(frame)} rows")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
