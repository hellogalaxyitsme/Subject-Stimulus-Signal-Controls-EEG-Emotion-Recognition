"""Train leakage-safe baselines from precomputed bandpower feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.labels import resolve_split_path, validate_assignments_target
from eeg_emotion_validity.baselines import (
    logistic_baseline,
    majority_baseline,
    random_prior_baseline,
    svm_baseline,
)
from eeg_emotion_validity.signals import DEFAULT_FEATURE_ROOT, DEFAULT_RUN_ROOT, DEFAULT_SPLIT_ROOT


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
    if protocol in {"loso", "strict_loso", "loso_stimulus_heldout", "loso_stimulus_seen"}:
        overlap = set(train_val["subject_key"]).intersection(set(test["subject_key"]))
        if overlap:
            raise AssertionError(f"subject leakage detected: {sorted(overlap)[:3]}")
    if protocol == "loso_stimulus_heldout":
        overlap = set(train_val["stimulus_key"]).intersection(set(test["stimulus_key"]))
        if overlap:
            raise AssertionError(f"stimulus leakage detected: {sorted(overlap)[:3]}")


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


def model_feature_set(model: str) -> str | None:
    if model in {"majority", "random_prior"}:
        return None
    if model.startswith("alpha_"):
        return "alpha"
    if model.startswith("spectral_"):
        return "spectral"
    raise ValueError(f"Unsupported model: {model}")


def train_model(model: str, x_train, y_train, x_test, y_test, seed: int) -> dict[str, object]:
    if model == "majority":
        result = majority_baseline(y_train, y_test)
    elif model == "random_prior":
        result = random_prior_baseline(y_train, y_test, seed=seed)
    elif model.endswith("_logistic"):
        result = logistic_baseline(x_train, y_train, x_test, y_test, name=model)
    elif model.endswith("_svm"):
        result = svm_baseline(x_train, y_train, x_test, y_test, name=model)
    else:
        raise ValueError(f"Unsupported model: {model}")
    return {"model": result.name, **result.metrics}, result.predictions


def run(args: argparse.Namespace) -> dict[str, object]:
    split_path = resolve_split_path(
        args.split_root, args.dataset, args.target, args.protocol, args.split_csv, args.label_variant
    )
    assignments = pd.read_csv(split_path)
    policy = validate_assignments_target(assignments, args.dataset, args.target, args.label_variant)
    protocol = assignments["protocol"].iloc[0]
    dataset = assignments["dataset"].iloc[0]
    if dataset != args.dataset:
        raise ValueError(f"Split CSV dataset {dataset} does not match requested {args.dataset}")
    if protocol != args.protocol:
        raise ValueError(f"Split CSV protocol {protocol} does not match requested {args.protocol}")

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    needed_feature_sets = sorted({model_feature_set(model) for model in models} - {None})
    features = {feature_set: load_feature_cache(feature_path(args, feature_set)) for feature_set in needed_feature_sets}

    min_feature_rows = min((matrix.shape[0] for matrix in features.values()), default=len(assignments))
    if args.allow_partial_feature_cache:
        assignments = assignments[assignments["row_index"].astype(int) < min_feature_rows].copy()
    elif features and assignments["row_index"].max() >= min_feature_rows:
        raise ValueError(
            "Feature cache does not cover all split row_index values. "
            "Use full feature extraction or pass --allow-partial-feature-cache only for smoke tests."
        )

    rows = []
    folds = sorted(assignments["fold"].unique())
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    for fold in folds:
        fold_df = assignments[assignments["fold"] == fold].copy()
        validate_fold(fold_df, protocol)
        train = sample_split(fold_df, "train", args.max_train_rows, args.seed + int(fold))
        test = sample_split(fold_df, "test", args.max_test_rows, args.seed + int(fold) + 1000)
        y_train = train["target_label"].to_numpy()
        y_test = test["target_label"].to_numpy()
        print(f"{dataset}/{protocol}/fold {fold}: train={len(train)} test={len(test)}")

        feature_matrices = {}
        for feature_set in needed_feature_sets:
            matrix = features[feature_set]
            feature_matrices[feature_set] = (
                matrix[train["row_index"].astype(int).to_numpy()],
                matrix[test["row_index"].astype(int).to_numpy()],
            )

        for model in models:
            feature_set = model_feature_set(model)
            if feature_set is None:
                metrics, predictions = train_model(model, None, y_train, None, y_test, args.seed + int(fold))
            else:
                x_train, x_test = feature_matrices[feature_set]
                metrics, predictions = train_model(model, x_train, y_train, x_test, y_test, args.seed + int(fold))
            if args.predictions_dir is not None:
                pred_path = args.predictions_dir / f"{args.output.stem}__{model}__fold{int(fold):03d}.csv"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "row_index": test["row_index"].astype(int).to_numpy(),
                        "subject_key": test["subject_key"].astype(str).to_numpy(),
                        "trial_key": test["trial_key"].astype(str).to_numpy(),
                        "stimulus_key": test["stimulus_key"].astype(str).to_numpy() if "stimulus_key" in test else None,
                        "y_true": y_test,
                        "y_pred": np.asarray(predictions),
                    }
                ).to_csv(pred_path, index=False)
            row = {
                "dataset": dataset,
                "protocol": protocol,
                "fold": int(fold),
                "target": args.target,
                "seed": int(args.seed),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                **metrics,
            }
            rows.append(row)
            print(row)

    result = {
        "metadata": {
            "dataset": dataset,
            "protocol": protocol,
            "target": args.target,
            "label_policy_id": policy.policy_id,
            "label_variant": args.label_variant,
            "split_path": str(split_path),
            "feature_root": str(args.feature_root),
            "feature_paths": {feature_set: str(feature_path(args, feature_set)) for feature_set in needed_feature_sets},
            "models": models,
            "seed": int(args.seed),
            "max_folds": args.max_folds,
            "max_train_rows": args.max_train_rows,
            "max_test_rows": args.max_test_rows,
            "allow_partial_feature_cache": args.allow_partial_feature_cache,
        },
        "fold_metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deap", "dreamer", "seed_iv"], required=True)
    parser.add_argument(
        "--protocol",
        choices=["segment_random", "trial_within_subject", "loso", "strict_loso", "loso_stimulus_heldout", "loso_stimulus_seen"],
        required=True,
    )
    parser.add_argument("--target", default="valence")
    parser.add_argument("--label-variant", default="primary", choices=["primary", "tie_high", "exclude_ties"])
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--alpha-features", type=Path)
    parser.add_argument("--spectral-features", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN_ROOT / "bandpower_baselines" / "feature_baseline.json",
    )
    parser.add_argument("--models", default="majority,random_prior,alpha_logistic,spectral_logistic,spectral_svm")
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
    print(json.dumps({"n_fold_metrics": len(result["fold_metrics"])}, indent=2))


if __name__ == "__main__":
    main()
