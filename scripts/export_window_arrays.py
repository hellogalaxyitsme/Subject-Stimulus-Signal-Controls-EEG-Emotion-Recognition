"""Export every TorchEEG cache window of a dataset to one memory-mappable .npy array.

Row i of the array equals ``signal_from_sample(dataset_obj[i])`` for manifest row i.
A sample of rows is re-read and compared exactly before the ``.complete`` marker is
written; the deep runner only uses arrays that carry this marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eeg_emotion_validity.signals import make_torcheeg_dataset, signal_from_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["deap", "dreamer", "seed_iv"])
    parser.add_argument("--torcheeg-root", type=Path, default=Path("data/processed/torcheeg"))
    parser.add_argument("--deap-root", type=Path, default=Path("data/raw/deap/data_preprocessed_python"))
    parser.add_argument("--dreamer-mat", type=Path, default=Path("data/raw/dreamer/DREAMER.mat"))
    parser.add_argument("--seed-iv-root", type=Path, default=Path("data/raw/seed_iv/eeg_raw_data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/window_arrays"))
    parser.add_argument("--verify-rows", type=int, default=500)
    args = parser.parse_args()

    ds = make_torcheeg_dataset(args.dataset, torcheeg_root=args.torcheeg_root, deap_root=args.deap_root,
                               dreamer_mat=args.dreamer_mat, seed_iv_root=args.seed_iv_root)
    first = signal_from_sample(ds[0])
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / f"{args.dataset}_windows.npy"
    array = np.lib.format.open_memmap(path, mode="w+", dtype=first.dtype, shape=(len(ds),) + first.shape)
    for i in range(len(ds)):
        array[i] = first if i == 0 else signal_from_sample(ds[i])
        if i and i % 10000 == 0:
            print(f"{args.dataset}: {i}/{len(ds)}", flush=True)
    array.flush()
    del array

    check = np.load(path, mmap_mode="r")
    rng = np.random.default_rng(20260926)
    rows = rng.choice(len(ds), size=min(args.verify_rows, len(ds)), replace=False)
    mismatches = [int(r) for r in rows if not np.array_equal(check[r], signal_from_sample(ds[int(r)]))]
    if mismatches:
        raise SystemExit(f"verification failed for rows {mismatches[:5]}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    info = {"dataset": args.dataset, "shape": list(check.shape), "dtype": str(check.dtype),
            "verified_rows": int(len(rows)), "sha256": digest.hexdigest()}
    path.with_suffix(".complete").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
