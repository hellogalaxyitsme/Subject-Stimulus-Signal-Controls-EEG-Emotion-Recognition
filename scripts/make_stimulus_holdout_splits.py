"""Generate matched subject x stimulus holdout splits.

For LOSO fold f with test subject s_f, stimulus group g_f = (f-1) mod G and a
same-size control group g'_f = (g_f + G//2) mod G:

  loso_stimulus_heldout: train/val = other subjects, stimuli not in g_f
  loso_stimulus_seen:    train/val = other subjects, stimuli not in g'_f
  both:                  test      = subject s_f, stimuli in g_f

Both conditions therefore share test chunks, training-set size, and validation
subjects; they differ only in whether the test stimuli were seen (from other
subjects) during training.

Stimulus identity: DEAP provider-preprocessed trials are in video (Experiment_id)
order (verified through the unaltered dominance and liking ratings); DREAMER trials
follow the fixed 18-clip order; SEED-IV stimuli are session x trial (72 clips, one
label each across subjects, verified against ReadMe.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_splits import standardize_manifest  # noqa: E402

from eeg_emotion_validity.labels import file_sha256, get_policy, label_vector_sha256, split_filename  # noqa: E402

PROTOCOLS = ("loso_stimulus_heldout", "loso_stimulus_seen")
N_GROUPS = {"deap": 5, "dreamer": 3, "seed_iv": 6}


def add_stimulus_columns(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = frame.copy()
    trial = out["trial_id"].astype(int)
    if dataset == "deap":
        out["stimulus_key"] = "video_" + trial.astype(str)
        out["stimulus_group"] = trial % N_GROUPS[dataset]
    elif dataset == "dreamer":
        out["stimulus_key"] = "clip_" + trial.astype(str)
        out["stimulus_group"] = trial % N_GROUPS[dataset]
    elif dataset == "seed_iv":
        session = out["session_id"].astype(int)
        out["stimulus_key"] = "s" + session.astype(str) + "_t" + trial.astype(str)
        clips = out[["session_id", "trial_id", "emotion"]].drop_duplicates()
        clips["stimulus_group"] = clips.groupby(["session_id", "emotion"])["trial_id"].rank(method="first").astype(int) - 1
        out = out.merge(clips, on=["session_id", "trial_id", "emotion"], how="left", validate="many_to_one")
    else:
        raise ValueError(dataset)
    return out


def build(frame: pd.DataFrame, protocol: str, seed: int) -> pd.DataFrame:
    groups = N_GROUPS[frame["dataset"].iloc[0]]
    subjects = sorted(frame["subject_key"].astype(str).unique())
    keep = ["row_index", "chunk_id", "dataset", "clip_id", "_record_id", "subject_key", "session_key",
            "trial_key", "stimulus_key", "stimulus_group", "target", "label_policy_id", "target_label"]
    parts = []
    for fold, subject in enumerate(subjects, start=1):
        test_group = (fold - 1) % groups
        excluded = test_group if protocol == "loso_stimulus_heldout" else (test_group + groups // 2) % groups
        others = [s for s in subjects if s != subject]
        rng = np.random.default_rng(seed + fold)
        val_subjects = set(np.asarray(others)[rng.permutation(len(others))][: max(1, round(0.2 * len(others)))])
        is_subject = frame["subject_key"].astype(str) == subject
        test = frame[is_subject & (frame["stimulus_group"] == test_group)]
        pool = frame[~is_subject & (frame["stimulus_group"] != excluded)]
        val = pool[pool["subject_key"].astype(str).isin(val_subjects)]
        train = pool[~pool["subject_key"].astype(str).isin(val_subjects)]
        for split, part in (("train", train), ("val", val), ("test", test)):
            part = part[keep].copy()
            part["protocol"] = protocol
            part["fold"] = fold
            part["split"] = split
            part["test_stimulus_group"] = test_group
            part["excluded_stimulus_group"] = excluded
            parts.append(part)
    assignments = pd.concat(parts, ignore_index=True)
    validate(assignments, protocol)
    return assignments


def validate(assignments: pd.DataFrame, protocol: str) -> None:
    for fold, fold_df in assignments.groupby("fold"):
        if fold_df["row_index"].duplicated().any():
            raise AssertionError(f"fold {fold}: duplicated rows")
        train_val = fold_df[fold_df["split"] != "test"]
        test = fold_df[fold_df["split"] == "test"]
        if set(train_val["subject_key"]) & set(test["subject_key"]):
            raise AssertionError(f"fold {fold}: subject leakage")
        shared = set(train_val["stimulus_key"]) & set(test["stimulus_key"])
        if protocol == "loso_stimulus_heldout" and shared:
            raise AssertionError(f"fold {fold}: stimulus leakage {sorted(shared)[:3]}")
        if protocol == "loso_stimulus_seen" and set(test["stimulus_key"]) - set(train_val["stimulus_key"]):
            raise AssertionError(f"fold {fold}: seen-control test stimuli missing from training")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/processed/torcheeg/manifests"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/splits/torcheeg"))
    parser.add_argument("--datasets", default="deap,dreamer,seed_iv")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for dataset in args.datasets.split(","):
        manifest = args.manifest_root / f"{dataset}_torcheeg_manifest.csv"
        raw = pd.read_csv(manifest)
        for target in (["emotion"] if dataset == "seed_iv" else ["valence", "arousal"]):
            frame = add_stimulus_columns(standardize_manifest(raw, dataset, target), dataset)
            for protocol in PROTOCOLS:
                assignments = build(frame, protocol, args.seed)
                path = args.output_root / split_filename(dataset, target, protocol)
                assignments.to_csv(path, index=False)
                test = assignments[assignments["split"] == "test"]
                support = test.groupby(["fold", "target_label"]).size().unstack(fill_value=0)
                outputs.append({
                    "dataset": dataset, "target": target, "protocol": protocol, "path": str(path),
                    "split_sha256": file_sha256(path), "label_policy_id": get_policy(dataset, target).policy_id,
                    "label_vector_sha256": label_vector_sha256(frame["chunk_id"], frame["target_label"]),
                    "folds": int(assignments["fold"].nunique()),
                    "test_chunks_per_fold_min": int(test.groupby("fold").size().min()),
                    "test_trials_per_fold_min": int(test.groupby("fold")["trial_key"].nunique().min()),
                    "test_folds_missing_a_class": int((support == 0).any(axis=1).sum()),
                    "train_chunks_mean": float(assignments[assignments["split"] == "train"].groupby("fold").size().mean()),
                })
                print(outputs[-1])
    (args.output_root / "stimulus_holdout_summary.json").write_text(json.dumps(outputs, indent=1))


if __name__ == "__main__":
    main()
