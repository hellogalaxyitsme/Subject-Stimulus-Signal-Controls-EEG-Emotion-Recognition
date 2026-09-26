"""Run DEAP <-> DREAMER raw-signal deep cross-dataset transfer.

The transfer setting intentionally uses the 14 electrodes shared by DEAP and
DREAMER, ordered as the DREAMER/Emotiv montage. Models train on chunks from one
dataset, validate on held-out source subjects, and test on all chunks from the
other dataset. Metrics are reported at both chunk level and trial level.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.labels import require_verified_ratings

from eeg_emotion_validity.baselines import classification_metrics, majority_baseline, random_prior_baseline
from eeg_emotion_validity.models import build_braindecode_model
from eeg_emotion_validity.signals import (
    DATASET_CHUNK_SIZE,
    DATASET_SFREQ,
    DEFAULT_DEAP_ROOT,
    DEFAULT_DREAMER_MAT,
    DEFAULT_TORCHEEG_ROOT,
    make_torcheeg_dataset,
    signal_from_sample,
)
from eeg_emotion_validity.transfer import assert_transfer_allowed


DEFAULT_MANIFEST_ROOT = Path("data/processed/torcheeg/manifests")
DEFAULT_OUTPUT = Path("runs/deep_cross_dataset_transfer/metrics.json")

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


def label_threshold(dataset: str) -> float:
    if dataset == "deap":
        return 5.0
    if dataset == "dreamer":
        return 3.0
    raise ValueError(f"Unsupported transfer dataset: {dataset}")


def shared_channel_indices(dataset: str) -> list[int]:
    names = CHANNELS_BY_DATASET[dataset]
    lookup = {name.upper(): index for index, name in enumerate(names)}
    return [lookup[name.upper()] for name in SHARED_CHANNELS]


def manifest_frame(dataset: str, target: str, manifest_root: Path) -> pd.DataFrame:
    manifest_path = manifest_root / f"{dataset}_torcheeg_manifest.csv"
    frame = require_verified_ratings(pd.read_csv(manifest_path), dataset).reset_index(drop=True)
    if target not in frame.columns:
        raise ValueError(f"{target!r} is absent from {manifest_path}")
    frame = frame[frame[target].notna()].copy()
    frame.insert(0, "row_index", frame.index.astype(int))

    threshold = label_threshold(dataset)
    frame["target_label"] = np.where(
        frame[target].astype(float) > threshold,
        f"high_{target}",
        f"low_{target}",
    )
    frame["subject_key"] = dataset + "_subject_" + frame["subject_id"].astype(str)
    frame["trial_key"] = (
        dataset
        + "_subject_"
        + frame["subject_id"].astype(str)
        + "_trial_"
        + frame["trial_id"].astype(str)
    )
    return frame[["row_index", "target_label", "subject_key", "trial_key"]].reset_index(drop=True)


def subject_holdout_split(frame: pd.DataFrame, seed: int, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = np.asarray(sorted(frame["subject_key"].astype(str).unique()))
    if len(subjects) < 2:
        raise ValueError("Need at least two source subjects for held-out validation")
    rng = np.random.default_rng(seed)
    subjects = subjects[rng.permutation(len(subjects))]
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    n_val = min(n_val, len(subjects) - 1)
    val_subjects = set(subjects[:n_val])
    val = frame[frame["subject_key"].isin(val_subjects)].copy()
    train = frame[~frame["subject_key"].isin(val_subjects)].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def sample_rows(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    per_class = max(1, max_rows // max(1, frame["target_label"].nunique()))
    chunks = [
        group.sample(n=min(len(group), per_class), random_state=seed)
        for _, group in frame.groupby("target_label")
    ]
    sampled = pd.concat(chunks).sample(frac=1.0, random_state=seed)
    if len(sampled) < max_rows:
        remaining = frame.drop(sampled.index)
        fill = remaining.sample(n=min(len(remaining), max_rows - len(sampled)), random_state=seed)
        sampled = pd.concat([sampled, fill]).sample(frac=1.0, random_state=seed)
    return sampled.reset_index(drop=True)


class TransferTorchEEGDataset:
    def __init__(self, dataset_obj, frame: pd.DataFrame, label_to_id: dict[str, int], channel_indices: list[int]):
        self.dataset_obj = dataset_obj
        self.frame = frame.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.channel_indices = channel_indices

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        import torch

        row = self.frame.iloc[index]
        signal = signal_from_sample(self.dataset_obj[int(row["row_index"])])
        signal = signal[self.channel_indices, :].astype("float32")
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True) + 1e-6
        signal = (signal - mean) / std
        label = self.label_to_id[str(row["target_label"])]
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long), str(row["trial_key"])


def logits_from_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def class_weights(labels: np.ndarray, n_classes: int):
    import torch

    counts = np.bincount(labels.astype(int), minlength=n_classes).astype("float32")
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def trial_labels(frame: pd.DataFrame, label_to_id: dict[str, int]) -> tuple[np.ndarray, list[str]]:
    labels = []
    keys = []
    for trial_key, group in frame.groupby("trial_key", sort=True):
        unique = sorted(group["target_label"].astype(str).unique())
        if len(unique) != 1:
            raise ValueError(f"Mixed labels within trial {trial_key}: {unique}")
        keys.append(str(trial_key))
        labels.append(label_to_id[unique[0]])
    return np.asarray(labels, dtype=int), keys


def aggregate_trial_predictions(logits: np.ndarray, labels: np.ndarray, trial_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grouped_logits: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_labels: dict[str, set[int]] = defaultdict(set)
    for logit, label, trial_key in zip(logits, labels, trial_keys, strict=True):
        grouped_logits[trial_key].append(logit)
        grouped_labels[trial_key].add(int(label))
    y_true = []
    y_pred = []
    for trial_key in sorted(grouped_logits):
        if len(grouped_labels[trial_key]) != 1:
            raise ValueError(f"Mixed labels within trial {trial_key}: {grouped_labels[trial_key]}")
        mean_logits = np.vstack(grouped_logits[trial_key]).mean(axis=0)
        y_true.append(next(iter(grouped_labels[trial_key])))
        y_pred.append(int(mean_logits.argmax()))
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)


def evaluate(model, loader, device: str, loss_fn=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float | None]:
    import torch

    model.eval()
    logits_rows = []
    y_true = []
    trial_keys = []
    losses = []
    n_seen = 0
    with torch.no_grad():
        for xb, yb, keys in loader:
            xb = xb.to(device)
            yb_device = yb.to(device)
            logits = logits_from_output(model(xb))
            if loss_fn is not None:
                loss = loss_fn(logits, yb_device)
                losses.append(float(loss.detach().cpu()) * int(len(yb)))
                n_seen += int(len(yb))
            logits_rows.append(logits.detach().cpu().numpy())
            y_true.extend(yb.numpy())
            trial_keys.extend([str(key) for key in keys])
    logits_array = np.vstack(logits_rows)
    y_array = np.asarray(y_true, dtype=int)
    pred_array = logits_array.argmax(axis=1).astype(int)
    mean_loss = float(sum(losses) / n_seen) if losses and n_seen else None
    return y_array, pred_array, logits_array, trial_keys, mean_loss


def metric_row(
    *,
    metadata: dict[str, object],
    model_name: str,
    prediction_level: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: float | None = None,
) -> dict[str, object]:
    row = {
        **metadata,
        "model": model_name,
        "prediction_level": prediction_level,
        "n_eval": int(len(y_true)),
        **classification_metrics(y_true, y_pred),
    }
    row["loss"] = loss
    row["balanced_accuracy_minus_chance"] = float(row["balanced_accuracy"] - 0.5)
    return row


def baseline_rows(
    *,
    metadata: dict[str, object],
    train_chunk_labels: np.ndarray,
    test_chunk_labels: np.ndarray,
    train_trial_labels: np.ndarray,
    test_trial_labels: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    for level, y_train, y_test in [
        ("chunk", train_chunk_labels, test_chunk_labels),
        ("trial", train_trial_labels, test_trial_labels),
    ]:
        majority = majority_baseline(y_train, y_test)
        random_prior = random_prior_baseline(y_train, y_test, seed=seed)
        rows.append(
            metric_row(
                metadata=metadata,
                model_name="majority",
                prediction_level=level,
                y_true=y_test,
                y_pred=np.asarray(majority.predictions),
            )
        )
        rows.append(
            metric_row(
                metadata=metadata,
                model_name="random_prior",
                prediction_level=level,
                y_true=y_test,
                y_pred=np.asarray(random_prior.predictions),
            )
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    assert_transfer_allowed(args.source_dataset, args.target_dataset, args.target)
    if DATASET_CHUNK_SIZE[args.source_dataset] != DATASET_CHUNK_SIZE[args.target_dataset]:
        raise ValueError("Source and target chunk sizes must match for raw-signal transfer")
    labels = [f"high_{args.target}", f"low_{args.target}"]
    label_to_id = {label: index for index, label in enumerate(labels)}

    source = manifest_frame(args.source_dataset, args.target, args.manifest_root)
    target = manifest_frame(args.target_dataset, args.target, args.manifest_root)
    train, val = subject_holdout_split(source, args.seed, args.val_fraction)
    train = sample_rows(train, args.max_train_rows, args.seed)
    val = sample_rows(val, args.max_val_rows, args.seed + 500)
    target = sample_rows(target, args.max_test_rows, args.seed + 1000)

    source_obj = make_torcheeg_dataset(
        dataset=args.source_dataset,
        torcheeg_root=args.torcheeg_root,
        io_mode=args.io_mode,
        deap_root=args.deap_root,
        dreamer_mat=args.dreamer_mat,
    )
    target_obj = make_torcheeg_dataset(
        dataset=args.target_dataset,
        torcheeg_root=args.torcheeg_root,
        io_mode=args.io_mode,
        deap_root=args.deap_root,
        dreamer_mat=args.dreamer_mat,
    )

    train_ds = TransferTorchEEGDataset(source_obj, train, label_to_id, shared_channel_indices(args.source_dataset))
    val_ds = TransferTorchEEGDataset(source_obj, val, label_to_id, shared_channel_indices(args.source_dataset))
    test_ds = TransferTorchEEGDataset(target_obj, target, label_to_id, shared_channel_indices(args.target_dataset))

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = build_braindecode_model(
        args.model,
        n_chans=len(SHARED_CHANNELS),
        n_times=DATASET_CHUNK_SIZE[args.source_dataset],
        n_outputs=len(labels),
        sfreq=args.sfreq or DATASET_SFREQ[args.source_dataset],
    ).to(device)
    train_label_ids = np.asarray([label_to_id[label] for label in train["target_label"].astype(str)], dtype=int)
    weight = class_weights(train_label_ids, len(labels)).to(device) if args.class_weights == "balanced" else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = logits_from_output(model(xb))
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        y_val, pred_val, _, _, val_loss = evaluate(model, val_loader, device, loss_fn=loss_fn)
        val_metrics = classification_metrics(y_val, pred_val)
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_row)
        print(epoch_row)

    y_test, pred_test, test_logits, test_trial_keys, test_loss = evaluate(model, test_loader, device, loss_fn=loss_fn)
    y_trial, pred_trial = aggregate_trial_predictions(test_logits, y_test, test_trial_keys)
    train_trial_y, _ = trial_labels(train, label_to_id)
    test_trial_y, _ = trial_labels(target, label_to_id)

    metadata = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "target": args.target,
        "seed": int(args.seed),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(target)),
        "train_trials": int(train["trial_key"].nunique()),
        "val_trials": int(val["trial_key"].nunique()),
        "test_trials": int(target["trial_key"].nunique()),
        "train_subjects": int(train["subject_key"].nunique()),
        "val_subjects": int(val["subject_key"].nunique()),
        "shared_channels": ",".join(SHARED_CHANNELS),
        "class_weights": args.class_weights,
    }
    rows = baseline_rows(
        metadata=metadata,
        train_chunk_labels=train_label_ids,
        test_chunk_labels=y_test,
        train_trial_labels=train_trial_y,
        test_trial_labels=test_trial_y,
        seed=args.seed,
    )
    rows.append(
        metric_row(
            metadata=metadata,
            model_name=args.model,
            prediction_level="chunk",
            y_true=y_test,
            y_pred=pred_test,
            loss=test_loss,
        )
    )
    rows.append(
        metric_row(
            metadata=metadata,
            model_name=args.model,
            prediction_level="trial",
            y_true=y_trial,
            y_pred=pred_trial,
        )
    )

    result = {
        "metadata": {
            **metadata,
            "model": args.model,
            "labels": labels,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "val_fraction": float(args.val_fraction),
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "max_test_rows": args.max_test_rows,
        },
        "history": history,
        "metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", choices=["deap", "dreamer"], required=True)
    parser.add_argument("--target-dataset", choices=["deap", "dreamer"], required=True)
    parser.add_argument("--target", choices=["valence", "arousal"], default="valence")
    parser.add_argument("--model", choices=["eegnet", "eegnetv4"], default="eegnet")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--torcheeg-root", type=Path, default=DEFAULT_TORCHEEG_ROOT)
    parser.add_argument("--deap-root", type=Path, default=DEFAULT_DEAP_ROOT)
    parser.add_argument("--dreamer-mat", type=Path, default=DEFAULT_DREAMER_MAT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sfreq", type=float)
    parser.add_argument("--io-mode", default="lmdb")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_dataset == args.target_dataset:
        raise ValueError("source-dataset and target-dataset must differ")
    result = run(args)
    print(f"Wrote {args.output}")
    print(json.dumps({"n_metric_rows": len(result["metrics"])}, indent=2))


if __name__ == "__main__":
    main()
