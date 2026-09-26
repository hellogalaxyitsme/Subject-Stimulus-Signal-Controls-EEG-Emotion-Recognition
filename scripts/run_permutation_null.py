"""Run strict-fold label-permutation null checks from bandpower feature caches.

This sanity check keeps the split, features, and held-out test labels unchanged,
but permutes training labels within each fold. Performance should remain near
chance-balanced behavior; materially above-chance results would suggest a
pipeline bug, leakage, or another non-label shortcut worth investigating.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.labels import resolve_split_path, validate_assignments_target
from eeg_emotion_validity.baselines import logistic_baseline, svm_baseline
from eeg_emotion_validity.signals import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SPLIT_ROOT,
)


def validate_fold(fold_df: pd.DataFrame, protocol: str) -> None:
    counts = fold_df["row_index"].value_counts()
    if (counts != 1).any():
        raise AssertionError("row_index assignment is not one-to-one within fold")
    train_val = fold_df[fold_df["split"].isin(["train", "val"])]
    test = fold_df[fold_df["split"] == "test"]
    if protocol in {"trial_within_subject", "strict_loso"}:
        overlap = set(train_val["trial_key"]).intersection(set(test["trial_key"]))
        if overlap:
            raise AssertionError(f"trial leakage detected: {sorted(overlap)[:3]}")
    if protocol in {"loso", "strict_loso"}:
        overlap = set(train_val["subject_key"]).intersection(set(test["subject_key"]))
        if overlap:
            raise AssertionError(f"subject leakage detected: {sorted(overlap)[:3]}")


def load_feature_cache(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def feature_path(args: argparse.Namespace, feature_set: str) -> Path:
    override = args.alpha_features if feature_set == "alpha" else args.spectral_features
    if override is not None:
        return override
    return args.feature_root / f"{args.dataset}_{feature_set}.npy"


def sample_split(frame: pd.DataFrame, split: str, max_rows: int | None, seed: int) -> pd.DataFrame:
    part = frame[frame["split"] == split].copy()
    if max_rows is None or len(part) <= max_rows:
        return part
    per_class = max(1, max_rows // max(1, part["target_label"].nunique()))
    class_samples = [
        group.sample(n=min(len(group), per_class), random_state=seed)
        for _, group in part.groupby("target_label")
    ]
    sampled = pd.concat(class_samples).sample(frac=1.0, random_state=seed)
    if len(sampled) < max_rows:
        remaining = part.drop(sampled.index)
        fill = remaining.sample(n=min(len(remaining), max_rows - len(sampled)), random_state=seed)
        sampled = pd.concat([sampled, fill]).sample(frac=1.0, random_state=seed)
    return sampled


def model_feature_set(model: str) -> str:
    if model.startswith("alpha_"):
        return "alpha"
    if model.startswith("spectral_"):
        return "spectral"
    raise ValueError(
        f"Unsupported null-check model {model!r}. Use alpha_* or spectral_* feature models."
    )


def train_model(model: str, x_train, y_train, x_test, y_test) -> dict[str, object]:
    if model.endswith("_logistic"):
        result = logistic_baseline(x_train, y_train, x_test, y_test, name=model)
    elif model.endswith("_svm"):
        result = svm_baseline(x_train, y_train, x_test, y_test, name=model)
    else:
        raise ValueError(f"Unsupported null-check model: {model}")
    return {"model": result.name, **result.metrics}


def permute_labels(y_train: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permuted = np.asarray(y_train).copy()
    rng.shuffle(permuted)
    return permuted


def class_counts(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


def ci95(values: pd.Series) -> tuple[float, float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, math.nan, math.nan, math.nan
    sd = float(arr.std(ddof=1))
    half = 1.96 * sd / math.sqrt(len(arr))
    return mean, sd, mean - half, mean + half


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summary_rows = []
    group_cols = ["dataset", "target", "protocol", "model"]
    for keys, group in frame.groupby(group_cols, sort=True):
        n_classes = int(group["n_classes"].max())
        chance_ba = 1.0 / n_classes if n_classes > 0 else math.nan
        ba_mean, ba_sd, ba_low, ba_high = ci95(group["balanced_accuracy"])
        acc_mean, acc_sd, acc_low, acc_high = ci95(group["accuracy"])
        summary_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_rows": int(len(group)),
                "n_folds": int(group["fold"].nunique()),
                "n_permutations": int(group["permutation_index"].nunique()),
                "n_classes": n_classes,
                "chance_balanced_accuracy": chance_ba,
                "balanced_accuracy_mean": ba_mean,
                "balanced_accuracy_sd": ba_sd,
                "balanced_accuracy_ci95_low": ba_low,
                "balanced_accuracy_ci95_high": ba_high,
                "balanced_accuracy_minus_chance": ba_mean - chance_ba,
                "accuracy_mean": acc_mean,
                "accuracy_sd": acc_sd,
                "accuracy_ci95_low": acc_low,
                "accuracy_ci95_high": acc_high,
                "interpretation": null_interpretation(ba_mean - chance_ba),
            }
        )
    return summary_rows


def null_interpretation(ba_minus_chance: float) -> str:
    if not np.isfinite(ba_minus_chance):
        return "insufficient data"
    if abs(ba_minus_chance) <= 0.03:
        return "near chance; null sanity check passed"
    if abs(ba_minus_chance) <= 0.07:
        return "small deviation from chance; inspect fold variability"
    return "material deviation from chance; investigate leakage or pipeline bug"


def run(args: argparse.Namespace) -> dict[str, object]:
    split_path = resolve_split_path(args.split_root, args.dataset, args.target, args.protocol, args.split_csv)
    assignments = pd.read_csv(split_path)
    validate_assignments_target(assignments, args.dataset, args.target)
    protocol = assignments["protocol"].iloc[0]
    dataset = assignments["dataset"].iloc[0]
    if dataset != args.dataset:
        raise ValueError(f"Split CSV dataset {dataset} does not match requested {args.dataset}")
    if protocol != args.protocol:
        raise ValueError(f"Split CSV protocol {protocol} does not match requested {args.protocol}")

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    needed_feature_sets = sorted({model_feature_set(model) for model in models})
    features = {
        feature_set: load_feature_cache(feature_path(args, feature_set))
        for feature_set in needed_feature_sets
    }

    min_feature_rows = min(matrix.shape[0] for matrix in features.values())
    if args.allow_partial_feature_cache:
        assignments = assignments[assignments["row_index"].astype(int) < min_feature_rows].copy()
    elif assignments["row_index"].max() >= min_feature_rows:
        raise ValueError(
            "Feature cache does not cover all split row_index values. "
            "Use full feature extraction or pass --allow-partial-feature-cache only for smoke tests."
        )

    folds = sorted(assignments["fold"].unique())
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    rows = []
    for fold in folds:
        fold_int = int(fold)
        fold_df = assignments[assignments["fold"] == fold].copy()
        validate_fold(fold_df, protocol)
        train = sample_split(fold_df, "train", args.max_train_rows, args.seed + fold_int)
        test = sample_split(fold_df, "test", args.max_test_rows, args.seed + fold_int + 1000)
        y_train = train["target_label"].to_numpy()
        y_test = test["target_label"].to_numpy()
        n_classes = int(pd.Series(np.concatenate([y_train, y_test])).nunique())
        print(
            f"{dataset}/{protocol}/fold {fold_int}: "
            f"train={len(train)} test={len(test)} classes={n_classes}"
        )

        feature_matrices = {}
        for feature_set in needed_feature_sets:
            matrix = features[feature_set]
            feature_matrices[feature_set] = (
                matrix[train["row_index"].astype(int).to_numpy()],
                matrix[test["row_index"].astype(int).to_numpy()],
            )

        for permutation_index in range(args.n_permutations):
            permutation_seed = args.seed + 100000 * (permutation_index + 1) + fold_int
            y_train_permuted = permute_labels(y_train, permutation_seed)
            before_counts = class_counts(y_train)
            after_counts = class_counts(y_train_permuted)
            if before_counts != after_counts:
                raise AssertionError("Label permutation changed the training class histogram")

            for model in models:
                feature_set = model_feature_set(model)
                x_train, x_test = feature_matrices[feature_set]
                metrics = train_model(model, x_train, y_train_permuted, x_test, y_test)
                row = {
                    "dataset": dataset,
                    "protocol": protocol,
                    "fold": fold_int,
                    "target": args.target,
                    "model": model,
                    "feature_set": feature_set,
                    "permutation_index": int(permutation_index),
                    "permutation_seed": int(permutation_seed),
                    "n_classes": n_classes,
                    "chance_balanced_accuracy": 1.0 / n_classes if n_classes else math.nan,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_class_counts": json.dumps(before_counts, sort_keys=True),
                    "test_class_counts": json.dumps(class_counts(y_test), sort_keys=True),
                    "permutation_scope": "train_labels_within_fold_only",
                    **metrics,
                }
                rows.append(row)
                print(row)

    summary_rows = summarize(rows)
    result = {
        "metadata": {
            "dataset": dataset,
            "protocol": protocol,
            "target": args.target,
            "null_test": "train_label_permutation",
            "permutation_scope": "train_labels_within_fold_only",
            "split_path": str(split_path),
            "feature_root": str(args.feature_root),
            "feature_paths": {
                feature_set: str(feature_path(args, feature_set))
                for feature_set in needed_feature_sets
            },
            "models": models,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "max_folds": args.max_folds,
            "max_train_rows": args.max_train_rows,
            "max_test_rows": args.max_test_rows,
            "allow_partial_feature_cache": args.allow_partial_feature_cache,
        },
        "fold_metrics": rows,
        "summary": summary_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deap", "dreamer", "seed_iv"], required=True)
    parser.add_argument(
        "--protocol",
        choices=["segment_random", "trial_within_subject", "loso", "strict_loso"],
        default="strict_loso",
    )
    parser.add_argument("--target", default="valence")
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--alpha-features", type=Path)
    parser.add_argument("--spectral-features", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN_ROOT / "label_permutation_null" / "feature_null.json",
    )
    parser.add_argument("--models", default="alpha_logistic,spectral_logistic")
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--allow-partial-feature-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(f"Wrote {args.output}")
    print(
        json.dumps(
            {
                "n_fold_metrics": len(result["fold_metrics"]),
                "n_summary_rows": len(result["summary"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
