"""Generate split-safe train/validation/test assignments for TorchEEG chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.labels import (
    build_target_labels,
    file_sha256,
    get_policy,
    label_vector_sha256,
    split_filename,
)


DEFAULT_MANIFEST_ROOT = Path("data/processed/torcheeg/manifests")
DEFAULT_OUTPUT_ROOT = Path("data/processed/splits/torcheeg")

PROTOCOLS = ("segment_random", "trial_within_subject", "loso", "strict_loso")
# "loso" and "strict_loso" are the same procedure (LeaveOneGroupOut over subjects
# with subject-grouped validation); "loso" is kept as an alias.
PROTOCOL_ALIASES = {"loso": "strict_loso"}
DATASET_TARGETS = {"deap": ("valence", "arousal"), "dreamer": ("valence", "arousal"), "seed_iv": ("emotion",)}


def label_from_target(frame: pd.DataFrame, dataset: str, target: str, variant: str = "primary") -> pd.Series:
    return build_target_labels(frame, dataset, target, variant)


def standardize_manifest(frame: pd.DataFrame, dataset: str, target: str, variant: str = "primary") -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out.insert(0, "row_index", out.index)
    out["dataset"] = dataset
    out["target"] = target
    out["label_policy_id"] = get_policy(dataset, target, variant).policy_id
    out["target_label"] = label_from_target(out, dataset, target, variant).to_numpy()
    # exclude_ties removes tied ratings; row_index keeps the original cache position.
    out = out[out["target_label"].notna()].reset_index(drop=True)

    if dataset == "deap":
        out["subject_key"] = out["subject_id"].astype(str)
        out["session_key"] = "session_1"
        out["trial_key"] = out["subject_key"] + "::session_1::trial_" + out["trial_id"].astype(str)
    elif dataset == "dreamer":
        out["subject_key"] = "s" + (out["subject_id"].astype(int) + 1).astype(str).str.zfill(2)
        out["session_key"] = "session_1"
        out["trial_key"] = out["subject_key"] + "::session_1::trial_" + out["trial_id"].astype(str)
    elif dataset == "seed_iv":
        out["subject_key"] = "s" + out["subject_id"].astype(int).astype(str).str.zfill(2)
        out["session_key"] = "session_" + out["session_id"].astype(int).astype(str)
        out["trial_key"] = (
            out["subject_key"]
            + "::"
            + out["session_key"]
            + "::trial_"
            + out["trial_id"].astype(str)
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    out["chunk_id"] = (
        dataset
        + "::"
        + out["subject_key"].astype(str)
        + "::"
        + out["trial_key"].astype(str)
        + "::"
        + out["clip_id"].astype(str)
        + "::"
        + out["row_index"].astype(str)
    )
    return out


def _safe_stratified_kfold(y: pd.Series, n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedKFold, KFold

    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < n_splits:
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter.split(np.zeros(len(y)), y)


def _sample_validation_indices(frame: pd.DataFrame, pool_idx: np.ndarray, seed: int, group_col: str | None = None):
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit

    if len(pool_idx) <= 1:
        return pool_idx, np.array([], dtype=int)

    pool = frame.iloc[pool_idx]
    if group_col is not None and pool[group_col].nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_rel, val_rel = next(splitter.split(pool, pool["target_label"], pool[group_col]))
    elif pool["target_label"].nunique() > 1 and pool["target_label"].value_counts().min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_rel, val_rel = next(splitter.split(pool, pool["target_label"]))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_rel, val_rel = next(splitter.split(pool))
    return pool_idx[train_rel], pool_idx[val_rel]


def build_assignments(frame: pd.DataFrame, protocol: str, n_splits: int, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut

    rows = []
    indices = np.arange(len(frame))

    if protocol == "segment_random":
        iterator = _safe_stratified_kfold(frame["target_label"], n_splits, seed)
        for fold, (train_pool, test_idx) in enumerate(iterator, start=1):
            train_idx, val_idx = _sample_validation_indices(frame, np.asarray(train_pool), seed + fold)
            rows.extend(_rows_for_fold(frame, protocol, fold, train_idx, val_idx, test_idx))

    elif protocol == "trial_within_subject":
        groups = frame["trial_key"].astype(str)
        if groups.nunique() < n_splits:
            iterator = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                frame, frame["target_label"], groups
            )
        else:
            iterator = GroupKFold(n_splits=n_splits).split(frame, frame["target_label"], groups)
        for fold, (train_pool, test_idx) in enumerate(iterator, start=1):
            train_idx, val_idx = _sample_validation_indices(
                frame, np.asarray(train_pool), seed + fold, group_col="trial_key"
            )
            rows.extend(_rows_for_fold(frame, protocol, fold, train_idx, val_idx, test_idx))

    elif protocol in {"loso", "strict_loso"}:
        groups = frame["subject_key"].astype(str)
        iterator = LeaveOneGroupOut().split(frame, frame["target_label"], groups)
        for fold, (train_pool, test_idx) in enumerate(iterator, start=1):
            train_idx, val_idx = _sample_validation_indices(
                frame, np.asarray(train_pool), seed + fold, group_col="subject_key"
            )
            rows.extend(_rows_for_fold(frame, protocol, fold, train_idx, val_idx, test_idx))

    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    assignments = pd.concat(rows, ignore_index=True)
    validate_assignments(frame, assignments, protocol)
    return assignments


def _rows_for_fold(frame, protocol, fold, train_idx, val_idx, test_idx):
    keep = [
        "row_index",
        "chunk_id",
        "dataset",
        "clip_id",
        "_record_id",
        "subject_key",
        "session_key",
        "trial_key",
        "target",
        "label_policy_id",
        "target_label",
    ]
    parts = []
    for split, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        part = frame.iloc[np.asarray(idx)][keep].copy()
        part["protocol"] = protocol
        part["fold"] = fold
        part["split"] = split
        parts.append(part)
    return parts


def validate_assignments(frame: pd.DataFrame, assignments: pd.DataFrame, protocol: str) -> None:
    for fold, fold_df in assignments.groupby("fold"):
        ids_by_split = {
            split: set(fold_df.loc[fold_df["split"] == split, "row_index"].astype(int))
            for split in ("train", "val", "test")
        }
        if ids_by_split["train"] & ids_by_split["val"]:
            raise AssertionError(f"{protocol} fold {fold}: train/val row overlap")
        if ids_by_split["train"] & ids_by_split["test"]:
            raise AssertionError(f"{protocol} fold {fold}: train/test row overlap")
        if ids_by_split["val"] & ids_by_split["test"]:
            raise AssertionError(f"{protocol} fold {fold}: val/test row overlap")
        if sum(len(v) for v in ids_by_split.values()) != len(frame):
            raise AssertionError(f"{protocol} fold {fold}: not all rows assigned exactly once")

        train_val = fold_df[fold_df["split"].isin(["train", "val"])]
        test = fold_df[fold_df["split"] == "test"]
        if protocol in {"trial_within_subject", "strict_loso"}:
            overlap = set(train_val["trial_key"]).intersection(set(test["trial_key"]))
            if overlap:
                raise AssertionError(f"{protocol} fold {fold}: trial leakage {sorted(overlap)[:3]}")
        if protocol in {"loso", "strict_loso"}:
            overlap = set(train_val["subject_key"]).intersection(set(test["subject_key"]))
            if overlap:
                raise AssertionError(f"{protocol} fold {fold}: subject leakage {sorted(overlap)[:3]}")


def summarize(assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        assignments.groupby(["dataset", "target", "protocol", "fold", "split"])
        .agg(
            chunks=("row_index", "count"),
            subjects=("subject_key", "nunique"),
            trials=("trial_key", "nunique"),
            classes=("target_label", "nunique"),
        )
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", choices=["all", "deap", "dreamer", "seed_iv"], default="all")
    parser.add_argument("--protocol", choices=["all", *PROTOCOLS], default="all")
    parser.add_argument(
        "--targets",
        default="auto",
        help="Comma-separated targets, or 'auto' for every supported target of each dataset",
    )
    parser.add_argument("--label-variant", default="primary", choices=["primary", "tie_high", "exclude_ties"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def targets_for(dataset: str, requested: str) -> list[str]:
    if requested == "auto":
        return list(DATASET_TARGETS[dataset])
    targets = [item.strip() for item in requested.split(",") if item.strip()]
    for target in targets:
        get_policy(dataset, target)
    return targets


def main() -> None:
    args = parse_args()
    datasets = ["deap", "dreamer", "seed_iv"] if args.dataset == "all" else [args.dataset]
    protocols = list(PROTOCOLS) if args.protocol == "all" else [args.protocol]
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    outputs = []
    for dataset in datasets:
        manifest = args.manifest_root / f"{dataset}_torcheeg_manifest.csv"
        frame = pd.read_csv(manifest)
        for target in targets_for(dataset, args.targets):
            standard = standardize_manifest(frame, dataset, target, args.label_variant)
            label_sha = label_vector_sha256(standard["chunk_id"], standard["target_label"])
            for protocol in protocols:
                assignments = build_assignments(standard, protocol, args.n_splits, args.seed)
                output_path = args.output_root / split_filename(dataset, target, protocol, args.label_variant)
                assignments.to_csv(output_path, index=False)
                summary = summarize(assignments)
                summary_path = output_path.with_name(output_path.stem + "_summary.csv")
                summary.to_csv(summary_path, index=False)
                all_summaries.append(summary)
                outputs.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "label_policy_id": get_policy(dataset, target, args.label_variant).policy_id,
                        "label_variant": args.label_variant,
                        "n_label_rows": int(len(standard)),
                        "label_vector_sha256": label_sha,
                        "manifest_path": str(manifest),
                        "manifest_sha256": file_sha256(manifest),
                        "protocol": protocol,
                        "protocol_canonical": PROTOCOL_ALIASES.get(protocol, protocol),
                        "path": str(output_path),
                        "split_sha256": file_sha256(output_path),
                        "summary_path": str(summary_path),
                        "rows": int(len(assignments)),
                        "folds": int(assignments["fold"].nunique()),
                        "seed": int(args.seed),
                    }
                )
                print(outputs[-1])

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(args.output_root / "all_split_summaries.csv", index=False)
    (args.output_root / "split_generation_summary.json").write_text(
        json.dumps({"outputs": outputs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

