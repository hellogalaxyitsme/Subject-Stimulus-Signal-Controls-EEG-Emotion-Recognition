"""Train split-safe Braindecode deep models with temporal ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.baselines import classification_metrics
from eeg_emotion_validity.models import build_braindecode_model
from eeg_emotion_validity.labels import (
    class_mapping,
    file_sha256,
    label_vector_sha256,
    resolve_split_path,
    validate_assignments_target,
)
from eeg_emotion_validity.signals import (
    DATASET_CHANNELS,
    DATASET_CHUNK_SIZE,
    DATASET_SFREQ,
    DEFAULT_DEAP_ROOT,
    DEFAULT_DREAMER_MAT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SEED_IV_ROOT,
    DEFAULT_SPLIT_ROOT,
    DEFAULT_TORCHEEG_ROOT,
    make_torcheeg_dataset,
    signal_from_sample,
)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def fold_checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.fold_checkpoint_dir is not None:
        return args.fold_checkpoint_dir
    return args.output.parent / f"{args.output.stem}_fold_checkpoints"


def fold_checkpoint_path(args: argparse.Namespace, fold: int) -> Path:
    return fold_checkpoint_dir(args) / f"fold{int(fold):03d}.json"


RUN_CONTRACT_VERSION = "1.0"
INPUT_TRANSFORM_ID = "temporal_ablation_then_per_window_per_channel_zscore_eps1e-6"


def run_fingerprint(args: argparse.Namespace, split_info: dict[str, str]) -> dict[str, object]:
    """Every setting that changes the scientific meaning of a fold result."""
    fingerprint = {
        "run_contract_version": RUN_CONTRACT_VERSION,
        "dataset": args.dataset,
        "target": args.target,
        "label_policy_id": split_info["label_policy_id"],
        "label_vector_sha256": split_info["label_vector_sha256"],
        "split_sha256": split_info["split_sha256"],
        "protocol": args.protocol,
        "model": args.model,
        "temporal_ablation": args.temporal_ablation,
        "input_transform": INPUT_TRANSFORM_ID,
        "optimizer": "AdamW",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "loss": "cross_entropy",
        "class_weighting": args.class_weighting,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "sfreq": args.sfreq,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "max_test_rows": args.max_test_rows,
        "early_stopping": bool(args.early_stopping),
        "inject_class_signal_snr": float(getattr(args, "inject_class_signal_snr", 0.0)),
    }
    if args.early_stopping:
        fingerprint.update(
            {
                "patience": int(args.patience),
                "max_epochs": int(args.max_epochs),
                "val_subjects": effective_val_subjects(args),
                "min_delta": float(args.min_delta),
            }
        )
    return fingerprint


def fingerprint_sha256(fingerprint: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()


def environment_versions() -> dict[str, str | None]:
    import importlib
    import platform

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "torch", "braindecode", "torcheeg"):
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", None)
        except Exception:
            versions[name] = None
    return versions


def is_smoke_run(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (args.max_folds, args.max_train_rows, args.max_val_rows, args.max_test_rows)
    )


def metadata_matches(payload: dict[str, object], args: argparse.Namespace, fold: int) -> bool:
    """A fold checkpoint is reusable only if its full scientific fingerprint matches.

    Checkpoints without a fingerprint are never reused.
    """
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    if int(metadata.get("fold", -1)) != int(fold):
        return False
    expected = getattr(args, "run_fingerprint", None)
    if expected is None:
        return False
    return metadata.get("run_fingerprint") == expected


def load_fold_checkpoint(args: argparse.Namespace, fold: int) -> dict[str, object] | None:
    path = fold_checkpoint_path(args, fold)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    fold_metrics = payload.get("fold_metrics")
    if not isinstance(fold_metrics, dict):
        return None
    if int(fold_metrics.get("fold", -1)) != int(fold):
        return None
    if not metadata_matches(payload, args, fold):
        return None
    return fold_metrics


def save_fold_checkpoint(args: argparse.Namespace, fold: int, fold_metrics: dict[str, object]) -> None:
    payload = {
        "metadata": {
            "dataset": args.dataset,
            "target": args.target,
            "protocol": args.protocol,
            "model": args.model,
            "temporal_ablation": args.temporal_ablation,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": int(args.seed),
            "fold": int(fold),
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "max_test_rows": args.max_test_rows,
            "early_stopping": bool(args.early_stopping),
            "patience": int(args.patience),
            "max_epochs": int(args.max_epochs),
            "val_subjects": effective_val_subjects(args),
            "min_delta": float(args.min_delta),
            "run_fingerprint": args.run_fingerprint,
            "run_fingerprint_sha256": fingerprint_sha256(args.run_fingerprint),
        },
        "fold_metrics": fold_metrics,
    }
    write_json_atomic(fold_checkpoint_path(args, fold), payload)


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


def training_class_weights(
    train: pd.DataFrame,
    label_to_id: dict[str, int],
    mode: str,
) -> list[float] | None:
    """Chunk-weighted 'balanced' weights n / (K * n_k) from the training split only."""
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unsupported class weighting: {mode}")
    counts = train["target_label"].astype(str).value_counts()
    n_classes = len(label_to_id)
    total = float(counts.sum())
    weights = [0.0] * n_classes
    for label, index in label_to_id.items():
        n_k = float(counts.get(label, 0))
        weights[index] = total / (n_classes * n_k) if n_k > 0 else 0.0
    return weights


def effective_val_subjects(args: argparse.Namespace) -> int:
    if args.val_subjects is not None:
        return int(args.val_subjects)
    return 1 if args.dataset == "seed_iv" else 2


def split_train_validation_subjects(
    fold_df: pd.DataFrame,
    args: argparse.Namespace,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_pool = fold_df[fold_df["split"].isin(["train", "val"])].copy()
    test_subjects = set(fold_df.loc[fold_df["split"] == "test", "subject_key"].astype(str))
    subjects = np.asarray(sorted(train_pool["subject_key"].astype(str).unique()))
    subjects = np.asarray([subject for subject in subjects if subject not in test_subjects])
    if len(subjects) < 2:
        raise ValueError("Early stopping requires at least two non-test subjects")

    rng = np.random.default_rng(args.seed + int(fold) + 1701)
    subjects = subjects[rng.permutation(len(subjects))]
    n_val = max(1, effective_val_subjects(args))
    n_val = min(n_val, len(subjects) - 1)
    val_subjects = set(subjects[:n_val].tolist())

    val = train_pool[train_pool["subject_key"].astype(str).isin(val_subjects)].copy()
    train = train_pool[~train_pool["subject_key"].astype(str).isin(val_subjects)].copy()
    if not len(train) or not len(val):
        raise ValueError("Early-stopping subject split produced an empty train or validation set")
    if set(train["subject_key"].astype(str)).intersection(set(val["subject_key"].astype(str))):
        raise AssertionError("early-stopping train/validation subject overlap detected")
    if set(val["subject_key"].astype(str)).intersection(test_subjects):
        raise AssertionError("early-stopping validation subject overlaps with test subject")
    return train.reset_index(drop=True), val.reset_index(drop=True), sorted(val_subjects)


def inject_class_signal(signal: np.ndarray, snr: float, sfreq: float, freq_hz: float = 10.0) -> np.ndarray:
    """Positive control: add a 10 Hz sinusoid scaled to ``snr`` x each channel's std.

    Applied only to class-id 0 samples, identically in train/val/test, so a
    functioning pipeline must reach clearly above-chance subject-held-out BA.
    Diagnostic only; never used for scientific results.
    """
    t = np.arange(signal.shape[-1], dtype=np.float32) / float(sfreq)
    wave = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
    return signal + snr * signal.std(axis=-1, keepdims=True) * wave


def phase_randomize(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase-randomized surrogate with one random phase sequence shared by all channels.

    Preserves each channel's full-window Fourier magnitude and the relative phase
    between channels at every frequency (hence cross-spectra); DC and, for even
    lengths, the Nyquist bin keep their original phase so the output is real.
    """
    n_times = signal.shape[-1]
    spectrum = np.fft.rfft(signal, axis=-1)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape[-1])
    phases[0] = 0.0
    if n_times % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(spectrum * np.exp(1j * phases), n=n_times, axis=-1).astype(signal.dtype)


class PreloadedSignals:
    """In-memory copy of the windows a job needs, read once from the TorchEEG cache.

    Returns exactly ``signal_from_sample(dataset_obj[row_index])`` for each row, so
    results are unchanged; only repeated per-epoch cache reads are avoided.
    """

    def __init__(self, dataset_obj, row_indices, progress_every: int = 20000):
        rows = np.unique(np.asarray(row_indices, dtype=np.int64))
        first = signal_from_sample(dataset_obj[int(rows[0])])
        self._position = {int(r): i for i, r in enumerate(rows)}
        self._data = np.empty((len(rows),) + first.shape, dtype=first.dtype)
        for i, row in enumerate(rows):
            self._data[i] = first if i == 0 else signal_from_sample(dataset_obj[int(row)])
            if progress_every and i and i % progress_every == 0:
                print(f"preloaded {i}/{len(rows)} windows", flush=True)

    def __getitem__(self, row_index: int):
        return (self._data[self._position[int(row_index)]],)

    def __len__(self) -> int:
        return len(self._position)


class ArraySignals:
    """Windows served from a verified export (scripts/export_window_arrays.py), memory-mapped."""

    def __init__(self, path: Path):
        self._data = np.load(path, mmap_mode="r")

    def __getitem__(self, row_index: int):
        return (np.asarray(self._data[int(row_index)]),)

    def __len__(self) -> int:
        return len(self._data)


def apply_temporal_ablation(signal: np.ndarray, mode: str, seed: int, row_index: int) -> np.ndarray:
    """Transforms applied identically in training and testing, fixed per sample
    (seeded by seed + row_index, so the same surrogate is reused every epoch), and
    before per-window z-scoring. ``time_shuffle`` applies one permutation to all
    channels; it keeps the per-channel amplitude multiset but not the spectrum.
    """
    if mode == "none":
        return signal
    if mode == "time_reverse":
        return signal[..., ::-1].copy()
    rng = np.random.default_rng(seed + int(row_index))
    if mode == "time_shuffle":
        order = rng.permutation(signal.shape[-1])
        return signal[..., order].copy()
    if mode == "phase_random":
        return phase_randomize(signal, rng)
    raise ValueError(f"Unsupported temporal ablation: {mode}")


class SplitTorchEEGDataset:
    def __init__(
        self,
        dataset_obj,
        frame: pd.DataFrame,
        label_to_id: dict[str, int],
        *,
        temporal_ablation: str,
        seed: int,
        return_index: bool = False,
        inject_snr: float = 0.0,
        sfreq: float = 128.0,
    ):
        self.dataset_obj = dataset_obj
        self.frame = frame.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.temporal_ablation = temporal_ablation
        self.seed = seed
        self.return_index = return_index
        self.inject_snr = float(inject_snr)
        self.sfreq = float(sfreq)

    def __len__(self) -> int:
        return len(self.frame)

    def with_index(self) -> "SplitTorchEEGDataset":
        """Same samples, but each item also carries its stable ``row_index``."""
        return SplitTorchEEGDataset(
            self.dataset_obj,
            self.frame,
            self.label_to_id,
            temporal_ablation=self.temporal_ablation,
            seed=self.seed,
            return_index=True,
            inject_snr=self.inject_snr,
            sfreq=self.sfreq,
        )

    def __getitem__(self, index: int):
        import torch

        row = self.frame.iloc[index]
        row_index = int(row["row_index"])
        signal = signal_from_sample(self.dataset_obj[row_index])
        signal = apply_temporal_ablation(signal, self.temporal_ablation, self.seed, row_index)
        signal = signal.astype("float32")
        label = self.label_to_id[str(row["target_label"])]
        if self.inject_snr > 0 and label == 0:
            signal = inject_class_signal(signal, self.inject_snr, self.sfreq)
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True) + 1e-6
        signal = (signal - mean) / std
        if self.return_index:
            return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long), torch.tensor(row_index)
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)


def logits_from_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def evaluate(model, loader, device: str, loss_fn=None) -> tuple[np.ndarray, np.ndarray, float | None]:
    import torch

    model.eval()
    y_true = []
    y_pred = []
    losses = []
    n_seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb_device = yb.to(device)
            logits = logits_from_output(model(xb))
            if loss_fn is not None:
                loss = loss_fn(logits, yb_device)
                losses.append(float(loss.detach().cpu()) * int(len(yb)))
                n_seen += int(len(yb))
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(pred)
    mean_loss = float(sum(losses) / n_seen) if losses and n_seen else None
    return np.asarray(y_true), np.asarray(y_pred), mean_loss


def _last_feature_module(model):
    import torch

    candidates = []
    for _, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            candidates.append(module)
    return candidates[-1] if candidates else None


EXTRACTION_VERSION = "keyed_deterministic_v2"


def extract_representations(model, loader, device: str):
    """Return (embeddings, label_ids, row_indices) in the order produced by ``loader``.

    ``loader`` must yield (x, y, row_index); row indices travel with every batch so
    that metadata is joined by key, never by position.
    """
    import torch

    module = _last_feature_module(model)
    captured = []

    def hook(_, inputs, __):
        value = inputs[0].detach().flatten(start_dim=1).cpu()
        captured.append(value)

    handle = module.register_forward_hook(hook) if module is not None else None
    embeddings = []
    labels = []
    row_indices = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                if len(batch) != 3:
                    raise ValueError("Representation extraction requires loaders yielding (x, y, row_index)")
                xb, yb, ib = batch
                captured.clear()
                xb = xb.to(device)
                logits = logits_from_output(model(xb))
                if captured:
                    emb = captured[-1]
                else:
                    emb = logits.detach().flatten(start_dim=1).cpu()
                embeddings.append(emb)
                labels.append(yb.detach().cpu())
                row_indices.append(ib.detach().cpu())
    finally:
        if handle is not None:
            handle.remove()
    return (
        torch.cat(embeddings).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(row_indices).numpy().astype(np.int64),
    )


def extraction_loader(dataset: SplitTorchEEGDataset, batch_size: int, num_workers: int):
    """Deterministic, non-dropping loader used only for representation export."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset.with_index(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )


def keyed_representation_metadata(
    split: str,
    frame: pd.DataFrame,
    label_ids: np.ndarray,
    row_indices: np.ndarray,
    label_to_id: dict[str, int],
) -> pd.DataFrame:
    """Join metadata to extracted rows by ``row_index`` and verify the join is exact."""
    frame = frame.reset_index(drop=True)
    if frame["row_index"].duplicated().any():
        raise AssertionError(f"{split}: duplicate row_index in frame")
    if pd.Series(row_indices).duplicated().any():
        raise AssertionError(f"{split}: extraction produced duplicate row_index values")
    if len(row_indices) != len(frame) or set(row_indices.tolist()) != set(frame["row_index"].astype(int)):
        raise AssertionError(
            f"{split}: extracted {len(row_indices)} rows but frame has {len(frame)}; row sets differ"
        )
    keyed = frame.set_index(frame["row_index"].astype(int)).loc[row_indices]
    expected_ids = keyed["target_label"].astype(str).map(label_to_id).to_numpy()
    if not np.array_equal(expected_ids, np.asarray(label_ids)):
        raise AssertionError(f"{split}: label ids from loader disagree with keyed metadata")
    return pd.DataFrame(
        {
            "split": split,
            "label_id": np.asarray(label_ids),
            "row_index": row_indices,
            "chunk_id": keyed["chunk_id"].astype(str).to_numpy() if "chunk_id" in keyed else None,
            "target_label": keyed["target_label"].astype(str).to_numpy(),
            "subject_key": keyed["subject_key"].astype(str).to_numpy(),
            "session_key": keyed["session_key"].astype(str).to_numpy(),
            "trial_key": keyed["trial_key"].astype(str).to_numpy(),
            "clip_id": keyed["clip_id"].astype(str).to_numpy(),
        }
    )


def save_representations(
    path: Path,
    model,
    datasets: dict[str, SplitTorchEEGDataset],
    frames: dict[str, pd.DataFrame],
    label_to_id: dict[str, int],
    device: str,
    *,
    batch_size: int,
    num_workers: int,
    provenance: dict[str, object],
) -> None:
    arrays = []
    metadata = []
    for split, dataset in datasets.items():
        loader = extraction_loader(dataset, batch_size, num_workers)
        embeddings, label_ids, row_indices = extract_representations(model, loader, device)
        arrays.append(embeddings)
        metadata.append(keyed_representation_metadata(split, frames[split], label_ids, row_indices, label_to_id))
    combined = pd.concat(metadata, ignore_index=True)
    combined["encoder_role"] = combined["split"]
    embeddings = np.vstack(arrays)
    if len(embeddings) != len(combined):
        raise AssertionError("embedding/metadata row count mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = path.with_name(path.stem + ".tmp.npz")
    tmp_csv = path.with_name(path.stem + ".tmp.csv")
    np.savez_compressed(tmp_npz, embeddings=embeddings, row_index=combined["row_index"].to_numpy())
    combined.to_csv(tmp_csv, index=False)
    tmp_npz.replace(path)
    tmp_csv.replace(path.with_suffix(".csv"))
    write_json_atomic(
        path.with_suffix(".provenance.json"),
        {
            "extraction_version": EXTRACTION_VERSION,
            "embedding_dim": int(embeddings.shape[1]),
            "n_rows": int(len(combined)),
            **provenance,
        },
    )


def run_fold(args: argparse.Namespace, fold_df: pd.DataFrame, fold: int) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    validate_fold(fold_df, args.protocol)
    if args.early_stopping:
        train_full, val_full, val_subject_keys = split_train_validation_subjects(fold_df, args, fold)
        train_full["split"] = "train"
        val_full["split"] = "val"
        train = sample_split(train_full, "train", args.max_train_rows, args.seed + fold)
        val = sample_split(val_full, "val", args.max_val_rows, args.seed + fold + 500)
    else:
        val_subject_keys = sorted(fold_df.loc[fold_df["split"] == "val", "subject_key"].astype(str).unique())
        train = sample_split(fold_df, "train", args.max_train_rows, args.seed + fold)
        val = sample_split(fold_df, "val", args.max_val_rows, args.seed + fold + 500)
    test = sample_split(fold_df, "test", args.max_test_rows, args.seed + fold + 1000)

    # Fixed policy-derived class mapping, never inferred from the classes present in a fold.
    label_to_id = class_mapping(args.label_policy)
    labels = sorted(label_to_id, key=label_to_id.get)
    unexpected = set(fold_df["target_label"].astype(str)) - set(labels)
    if unexpected:
        raise ValueError(f"fold {fold}: labels {sorted(unexpected)} not in policy {args.label_policy.policy_id}")
    dataset_obj = getattr(args, "preloaded_signals", None) or open_torcheeg_dataset(args)

    train_ds = SplitTorchEEGDataset(
        dataset_obj,
        train,
        label_to_id,
        temporal_ablation=args.temporal_ablation,
        seed=args.seed,
        inject_snr=args.inject_class_signal_snr,
        sfreq=args.sfreq or DATASET_SFREQ[args.dataset],
    )
    val_ds = SplitTorchEEGDataset(
        dataset_obj,
        val,
        label_to_id,
        temporal_ablation=args.temporal_ablation,
        seed=args.seed,
        inject_snr=args.inject_class_signal_snr,
        sfreq=args.sfreq or DATASET_SFREQ[args.dataset],
    )
    test_ds = SplitTorchEEGDataset(
        dataset_obj,
        test,
        label_to_id,
        temporal_ablation=args.temporal_ablation,
        seed=args.seed,
        inject_snr=args.inject_class_signal_snr,
        sfreq=args.sfreq or DATASET_SFREQ[args.dataset],
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed + fold)
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
    torch.manual_seed(args.seed + fold)
    model = build_braindecode_model(
        args.model,
        n_chans=DATASET_CHANNELS[args.dataset],
        n_times=DATASET_CHUNK_SIZE[args.dataset],
        n_outputs=len(labels),
        sfreq=args.sfreq or DATASET_SFREQ[args.dataset],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weight_values = training_class_weights(train, label_to_id, args.class_weighting)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=None
        if class_weight_values is None
        else torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    )

    history = []
    best_state = None
    best_epoch = None
    best_val_ba = -float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    n_epochs = int(args.max_epochs if args.early_stopping else args.epochs)
    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = logits_from_output(model(xb))
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if len(val_ds):
            y_val, pred_val, val_loss = evaluate(model, val_loader, device, loss_fn=loss_fn)
            val_metrics = classification_metrics(y_val, pred_val)
        else:
            val_loss = None
            val_metrics = {}
        val_ba = float(val_metrics.get("balanced_accuracy", float("-inf")))
        improved = bool(args.early_stopping and val_ba > best_val_ba + args.min_delta)
        if args.early_stopping:
            if improved:
                best_val_ba = val_ba
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if best_state is None:
                    best_val_ba = val_ba
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
            stopped_early = epochs_without_improvement >= args.patience
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "new_best": improved,
                "epochs_without_improvement": int(epochs_without_improvement),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        print(history[-1])
        if args.early_stopping and stopped_early:
            print(
                {
                    "event": "early_stopping",
                    "fold": int(fold),
                    "best_epoch": int(best_epoch),
                    "val_ba_at_best": float(best_val_ba),
                    "patience": int(args.patience),
                    "total_epochs_run": int(epoch),
                }
            )
            break

    if args.early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    if best_epoch is None:
        best_epoch = len(history)
    if not np.isfinite(best_val_ba):
        best_val_ba = float("nan")

    y_test, pred_test, test_loss = evaluate(model, test_loader, device, loss_fn=loss_fn)
    predictions_path = None
    if args.predictions_dir is not None:
        # Chunk-level out-of-fold predictions keyed by row_index, so trial- and
        # subject-level metrics and paired effects can be recomputed.
        predictions_path = (
            args.predictions_dir
            / fingerprint_sha256(args.run_fingerprint)[:12]
            / f"{args.dataset}_{args.protocol}_{args.target}_{args.model}_{args.temporal_ablation}_fold{fold}.csv"
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        frame = test.reset_index(drop=True)
        pd.DataFrame(
            {
                "row_index": frame["row_index"].astype(int),
                "chunk_id": frame["chunk_id"].astype(str),
                "subject_key": frame["subject_key"].astype(str),
                "trial_key": frame["trial_key"].astype(str),
                "y_true": y_test,
                "y_pred": pred_test,
            }
        ).to_csv(predictions_path, index=False)
    metrics = classification_metrics(y_test, pred_test, labels=list(range(len(labels))))
    if args.eval_train:
        # Training-fit diagnostic: BA on the (possibly row-capped) training split in eval mode.
        train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        y_train_eval, pred_train_eval, _ = evaluate(model, train_eval_loader, device)
        metrics["train_balanced_accuracy"] = classification_metrics(y_train_eval, pred_train_eval)["balanced_accuracy"]
    checkpoint_path = None
    representation_path = None
    if args.checkpoint_dir is not None:
        checkpoint_path = (
            args.checkpoint_dir
            / fingerprint_sha256(args.run_fingerprint)[:12]
            / f"{args.dataset}_{args.protocol}_{args.target}_{args.model}_{args.temporal_ablation}_fold{fold}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "labels": labels,
                "metadata": {
                    "dataset": args.dataset,
                    "target": args.target,
                    "protocol": args.protocol,
                    "fold": int(fold),
                    "model": args.model,
                    "temporal_ablation": args.temporal_ablation,
                    "early_stopping": bool(args.early_stopping),
                    "best_epoch": int(best_epoch),
                    "val_ba_at_best": float(best_val_ba),
                    "label_policy_id": args.label_policy.policy_id,
                    "run_fingerprint": args.run_fingerprint,
                    "run_fingerprint_sha256": fingerprint_sha256(args.run_fingerprint),
                },
            },
            checkpoint_path,
        )
    if args.representation_dir is not None:
        representation_path = (
            args.representation_dir
            / fingerprint_sha256(args.run_fingerprint)[:12]
            / f"{args.dataset}_{args.protocol}_{args.target}_{args.model}_{args.temporal_ablation}_fold{fold}.npz"
        )
        save_representations(
            representation_path,
            model,
            {"train": train_ds, "val": val_ds, "test": test_ds},
            {"train": train, "val": val, "test": test},
            label_to_id,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            provenance={
                "run_fingerprint": args.run_fingerprint,
                "run_fingerprint_sha256": fingerprint_sha256(args.run_fingerprint),
                "fold": int(fold),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "embedding_layer": "input_to_last_linear_or_conv_module",
            },
        )
    return {
        "dataset": args.dataset,
        "target": args.target,
        "protocol": args.protocol,
        "fold": int(fold),
        "model": args.model,
        "temporal_ablation": args.temporal_ablation,
        "seed": int(args.seed),
        "train_rows": int(len(train_ds)),
        "val_rows": int(len(val_ds)),
        "test_rows": int(len(test_ds)),
        "train_subjects": int(train["subject_key"].nunique()),
        "val_subjects": int(val["subject_key"].nunique()),
        "test_subjects": int(test["subject_key"].nunique()),
        "val_subject_keys": val_subject_keys,
        "early_stopping": bool(args.early_stopping),
        "best_epoch": int(best_epoch),
        "patience_used": int(args.patience if args.early_stopping else 0),
        "val_ba_at_best": float(best_val_ba),
        "stopped_early": bool(stopped_early),
        "total_epochs_run": int(len(history)),
        "label_policy_id": args.label_policy.policy_id,
        "class_weighting": args.class_weighting,
        "class_weights": class_weight_values,
        "test_class_support": {
            label: int(count) for label, count in test["target_label"].astype(str).value_counts().items()
        },
        "classes": labels,
        "history": history,
        "test_loss": test_loss,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "representation_path": str(representation_path) if representation_path is not None else None,
        "predictions_path": str(predictions_path) if predictions_path is not None else None,
        **metrics,
    }


def prepare_split(args: argparse.Namespace) -> tuple[Path, pd.DataFrame]:
    """Load the split and enforce the target contract before any model is built."""
    split_path = resolve_split_path(args.split_root, args.dataset, args.target, args.protocol, args.split_csv)
    assignments = pd.read_csv(split_path)
    if set(assignments["dataset"].astype(str)) != {args.dataset}:
        raise ValueError("Split dataset does not match requested dataset")
    if set(assignments["protocol"].astype(str)) != {args.protocol}:
        raise ValueError("Split protocol does not match requested protocol")
    policy = validate_assignments_target(assignments, args.dataset, args.target)
    unique_labels = assignments[["chunk_id", "target_label"]].drop_duplicates()
    split_info = {
        "label_policy_id": policy.policy_id,
        "label_vector_sha256": label_vector_sha256(unique_labels["chunk_id"], unique_labels["target_label"]),
        "split_sha256": file_sha256(split_path),
    }
    args.label_policy = policy
    args.split_info = split_info
    args.run_fingerprint = run_fingerprint(args, split_info)
    return split_path, assignments


def open_torcheeg_dataset(args: argparse.Namespace):
    return make_torcheeg_dataset(
        dataset=args.dataset,
        torcheeg_root=args.torcheeg_root,
        io_mode=args.io_mode,
        deap_root=args.deap_root,
        dreamer_mat=args.dreamer_mat,
        seed_iv_root=args.seed_iv_root,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    # CPU threads only affect host-side work (models run on the GPU); capping them avoids
    # oversubscription when several jobs share the machine.
    torch.set_num_threads(max(1, int(args.torch_threads)))
    split_path, assignments = prepare_split(args)

    folds = sorted(assignments["fold"].unique())
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    pending = [int(f) for f in folds if load_fold_checkpoint(args, int(f)) is None]
    if args.preload and pending:
        exported = args.window_array_root / f"{args.dataset}_windows.npy"
        if exported.with_suffix(".complete").exists():
            print(f"Using verified window array {exported}", flush=True)
            args.preloaded_signals = ArraySignals(exported)
        else:
            needed = assignments.loc[assignments["fold"].isin(pending), "row_index"]
            print(f"Preloading {needed.nunique()} windows into memory", flush=True)
            args.preloaded_signals = PreloadedSignals(open_torcheeg_dataset(args), needed.to_numpy())

    rows = []
    for fold in folds:
        fold = int(fold)
        cached = load_fold_checkpoint(args, fold)
        if cached is not None:
            print(
                f"Skipping checkpointed {args.dataset}/{args.protocol}/fold "
                f"{fold}/{args.model}/{args.temporal_ablation}"
            )
            rows.append(cached)
            continue
        print(f"Running {args.dataset}/{args.protocol}/fold {fold}/{args.model}/{args.temporal_ablation}")
        fold_metrics = run_fold(args, assignments[assignments["fold"] == fold].copy(), fold)
        save_fold_checkpoint(args, fold, fold_metrics)
        print(f"Wrote fold checkpoint {fold_checkpoint_path(args, fold)}")
        rows.append(fold_metrics)

    result = {
        "metadata": {
            "dataset": args.dataset,
            "target": args.target,
            "protocol": args.protocol,
            "model": args.model,
            "temporal_ablation": args.temporal_ablation,
            "split_path": str(split_path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": int(args.seed),
            "max_folds": args.max_folds,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "max_test_rows": args.max_test_rows,
            "early_stopping": bool(args.early_stopping),
            "patience": int(args.patience),
            "max_epochs": int(args.max_epochs),
            "val_subjects": effective_val_subjects(args),
            "min_delta": float(args.min_delta),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "class_weighting": args.class_weighting,
            "input_transform": INPUT_TRANSFORM_ID,
            "is_smoke": is_smoke_run(args),
            "n_folds_expected": int(assignments["fold"].nunique()),
            "n_folds_completed": int(len(rows)),
            "run_fingerprint": args.run_fingerprint,
            "run_fingerprint_sha256": fingerprint_sha256(args.run_fingerprint),
            "environment": environment_versions(),
        },
        "fold_metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
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
    parser.add_argument(
        "--model",
        choices=[
            "eegnet",
            "eegnetv4",
            "shallow_convnet",
            "deep_convnet",
            "eegconformer",
            "eeg_conformer",
            "tsception",
        ],
        default="eegnet",
    )
    parser.add_argument(
        "--temporal-ablation", choices=["none", "time_shuffle", "time_reverse", "phase_random"], default="none"
    )
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--torcheeg-root", type=Path, default=DEFAULT_TORCHEEG_ROOT)
    parser.add_argument("--deap-root", type=Path, default=DEFAULT_DEAP_ROOT)
    parser.add_argument("--dreamer-mat", type=Path, default=DEFAULT_DREAMER_MAT)
    parser.add_argument("--seed-iv-root", type=Path, default=DEFAULT_SEED_IV_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN_ROOT / "braindecode_ablation" / "metrics.json")
    parser.add_argument("--sfreq", type=float)
    parser.add_argument("--io-mode", default="lmdb")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--val-subjects", type=int)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="none",
        help="'none': unweighted loss; 'balanced': weights from training-split window counts",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--representation-dir", type=Path)
    parser.add_argument("--fold-checkpoint-dir", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--eval-train", action="store_true", help="also report training-split BA (diagnostic)")
    parser.add_argument(
        "--preload", action="store_true", help="read each needed window once into memory (I/O only; results unchanged)"
    )
    parser.add_argument("--window-array-root", type=Path, default=Path("data/processed/window_arrays"))
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--inject-class-signal-snr",
        type=float,
        default=0.0,
        help="POSITIVE CONTROL ONLY: add a 10 Hz sinusoid to class-0 samples at this SNR",
    )
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(f"Wrote {args.output}")
    print(json.dumps({"n_fold_metrics": len(result["fold_metrics"])}, indent=2))


if __name__ == "__main__":
    main()
