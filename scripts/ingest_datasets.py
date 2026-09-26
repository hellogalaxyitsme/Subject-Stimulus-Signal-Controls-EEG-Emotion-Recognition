"""Build TorchEEG caches and manifests for DEAP, DREAMER, and SEED-IV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DEAP_ROOT = Path("data/raw/deap/data_preprocessed_python")
DEFAULT_DREAMER_MAT = Path("data/raw/dreamer/DREAMER.mat")
DEFAULT_SEED_IV_ROOT = Path("data/raw/seed_iv/eeg_raw_data")
DEFAULT_OUTPUT_ROOT = Path("data/processed/torcheeg")


def _info_to_frame(info: Any):
    import pandas as pd

    if isinstance(info, pd.DataFrame):
        return info.copy()
    if hasattr(info, "to_dataframe"):
        return info.to_dataframe()
    if hasattr(info, "to_pandas"):
        return info.to_pandas()
    return pd.DataFrame(info)


def _write_manifest(dataset: Any, manifest_path: Path) -> int:
    frame = _info_to_frame(dataset.info)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)
    return len(frame)


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            old = existing.get("ingestions", [])
            new = payload.get("ingestions", [])
            by_dataset = {item["dataset"]: item for item in old if "dataset" in item}
            by_dataset.update({item["dataset"]: item for item in new if "dataset" in item})
            payload = {"ingestions": list(by_dataset.values())}
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_deap(args: argparse.Namespace) -> dict[str, Any]:
    from torcheeg.datasets import DEAPDataset

    io_path = args.output_root / f"deap_chunk{args.deap_chunk_size}_overlap{args.overlap}"
    dataset = DEAPDataset(
        root_path=str(args.deap_root),
        chunk_size=args.deap_chunk_size,
        overlap=args.overlap,
        num_channel=32,
        io_path=str(io_path),
        io_mode=args.io_mode,
        num_worker=args.num_worker,
        verbose=args.verbose,
    )
    manifest_path = args.output_root / "manifests" / "deap_torcheeg_manifest.csv"
    n_rows = _write_manifest(dataset, manifest_path)
    return {
        "dataset": "deap",
        "raw_path": str(args.deap_root),
        "io_path": str(io_path),
        "manifest_path": str(manifest_path),
        "rows": n_rows,
        "chunk_size": args.deap_chunk_size,
        "overlap": args.overlap,
    }


def build_dreamer(args: argparse.Namespace) -> dict[str, Any]:
    from torcheeg.datasets import DREAMERDataset

    io_path = args.output_root / f"dreamer_chunk{args.dreamer_chunk_size}_overlap{args.overlap}"
    dataset = DREAMERDataset(
        mat_path=str(args.dreamer_mat),
        chunk_size=args.dreamer_chunk_size,
        overlap=args.overlap,
        num_channel=14,
        io_path=str(io_path),
        io_mode=args.io_mode,
        num_worker=args.num_worker,
        verbose=args.verbose,
    )
    manifest_path = args.output_root / "manifests" / "dreamer_torcheeg_manifest.csv"
    n_rows = _write_manifest(dataset, manifest_path)
    return {
        "dataset": "dreamer",
        "raw_path": str(args.dreamer_mat),
        "io_path": str(io_path),
        "manifest_path": str(manifest_path),
        "rows": n_rows,
        "chunk_size": args.dreamer_chunk_size,
        "overlap": args.overlap,
    }


def build_seed_iv(args: argparse.Namespace) -> dict[str, Any]:
    from torcheeg.datasets import SEEDIVDataset

    io_path = args.output_root / f"seed_iv_chunk{args.seed_iv_chunk_size}_overlap{args.overlap}"
    dataset = SEEDIVDataset(
        root_path=str(args.seed_iv_root),
        chunk_size=args.seed_iv_chunk_size,
        overlap=args.overlap,
        num_channel=62,
        io_path=str(io_path),
        io_mode=args.io_mode,
        num_worker=args.num_worker,
        verbose=args.verbose,
    )
    manifest_path = args.output_root / "manifests" / "seed_iv_torcheeg_manifest.csv"
    n_rows = _write_manifest(dataset, manifest_path)
    return {
        "dataset": "seed_iv",
        "raw_path": str(args.seed_iv_root),
        "io_path": str(io_path),
        "manifest_path": str(manifest_path),
        "rows": n_rows,
        "chunk_size": args.seed_iv_chunk_size,
        "overlap": args.overlap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "deap", "dreamer", "seed_iv"], default="all")
    parser.add_argument("--deap-root", type=Path, default=DEFAULT_DEAP_ROOT)
    parser.add_argument("--dreamer-mat", type=Path, default=DEFAULT_DREAMER_MAT)
    parser.add_argument("--seed-iv-root", type=Path, default=DEFAULT_SEED_IV_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--deap-chunk-size", type=int, default=128)
    parser.add_argument("--dreamer-chunk-size", type=int, default=128)
    parser.add_argument("--seed-iv-chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--io-mode", default="lmdb")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.verbose = not args.quiet
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    if args.dataset in {"all", "deap"}:
        summaries.append(build_deap(args))
    if args.dataset in {"all", "dreamer"}:
        summaries.append(build_dreamer(args))
    if args.dataset in {"all", "seed_iv"}:
        summaries.append(build_seed_iv(args))

    summary_path = args.output_root / "ingestion_summary.json"
    _write_summary(summary_path, {"ingestions": summaries})
    for summary in summaries:
        print(summary)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
