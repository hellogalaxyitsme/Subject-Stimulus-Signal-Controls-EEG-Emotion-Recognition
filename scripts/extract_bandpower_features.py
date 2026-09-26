"""Precompute reusable alpha/spectral bandpower features from TorchEEG caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eeg_emotion_validity.signals import (
    DATASET_SFREQ,
    DEFAULT_DEAP_ROOT,
    DEFAULT_DREAMER_MAT,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_SEED_IV_ROOT,
    DEFAULT_TORCHEEG_ROOT,
    feature_names,
    feature_vector,
    make_torcheeg_dataset,
    signal_from_sample,
)


def output_stem(dataset: str, feature_set: str, max_rows: int | None, suffix: str) -> str:
    stem = f"{dataset}_{feature_set}"
    if max_rows is not None:
        stem += f"_first{max_rows}"
    if suffix:
        stem += f"_{suffix}"
    return stem


def extract_feature_set(args: argparse.Namespace, feature_set: str) -> dict[str, object]:
    dataset_obj = make_torcheeg_dataset(
        dataset=args.dataset,
        torcheeg_root=args.torcheeg_root,
        io_mode=args.io_mode,
        deap_root=args.deap_root,
        dreamer_mat=args.dreamer_mat,
        seed_iv_root=args.seed_iv_root,
    )
    n_rows = len(dataset_obj)
    if args.max_rows is not None:
        n_rows = min(n_rows, args.max_rows)
    sfreq = args.sfreq or DATASET_SFREQ[args.dataset]

    first = feature_vector(signal_from_sample(dataset_obj[0]), sfreq, feature_set)
    names = feature_names(args.dataset, feature_set)
    if len(first) != len(names):
        names = [f"feature_{idx:04d}" for idx in range(len(first))]

    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.dataset, feature_set, args.max_rows, args.suffix)
    feature_path = args.output_root / f"{stem}.npy"
    row_index_path = args.output_root / f"{stem}_row_index.npy"
    metadata_path = args.output_root / f"{stem}_metadata.json"

    if feature_path.exists() and not args.overwrite:
        raise FileExistsError(f"{feature_path} exists; pass --overwrite to replace it")

    matrix = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, len(first)),
    )
    row_indices = np.arange(n_rows, dtype=np.int64)
    for row_index in range(n_rows):
        if row_index == 0:
            matrix[row_index] = first
        else:
            matrix[row_index] = feature_vector(
                signal_from_sample(dataset_obj[row_index]),
                sfreq,
                feature_set,
            )
        if args.progress_every and (row_index + 1) % args.progress_every == 0:
            print(f"{args.dataset}/{feature_set}: extracted {row_index + 1}/{n_rows}")

    matrix.flush()
    np.save(row_index_path, row_indices)
    metadata = {
        "dataset": args.dataset,
        "feature_set": feature_set,
        "feature_path": str(feature_path),
        "row_index_path": str(row_index_path),
        "rows": int(n_rows),
        "feature_dim": int(len(first)),
        "sfreq": float(sfreq),
        "feature_names": names,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("dataset", "feature_set", "rows", "feature_dim")}))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deap", "dreamer", "seed_iv"], required=True)
    parser.add_argument("--feature-set", choices=["alpha", "spectral", "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--torcheeg-root", type=Path, default=DEFAULT_TORCHEEG_ROOT)
    parser.add_argument("--deap-root", type=Path, default=DEFAULT_DEAP_ROOT)
    parser.add_argument("--dreamer-mat", type=Path, default=DEFAULT_DREAMER_MAT)
    parser.add_argument("--seed-iv-root", type=Path, default=DEFAULT_SEED_IV_ROOT)
    parser.add_argument("--sfreq", type=float)
    parser.add_argument("--io-mode", default="lmdb")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_sets = ["alpha", "spectral"] if args.feature_set == "all" else [args.feature_set]
    summary = [extract_feature_set(args, feature_set) for feature_set in feature_sets]
    summary_path = args.output_root / f"{args.dataset}_feature_extraction_summary.json"
    existing = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8")).get("features", [])
        except Exception:
            existing = []
    by_key = {(item["dataset"], item["feature_set"], item["feature_path"]): item for item in existing}
    by_key.update({(item["dataset"], item["feature_set"], item["feature_path"]): item for item in summary})
    summary_path.write_text(
        json.dumps({"features": list(by_key.values())}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
