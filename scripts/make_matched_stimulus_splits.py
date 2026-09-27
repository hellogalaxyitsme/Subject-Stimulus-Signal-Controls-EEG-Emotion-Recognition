"""Class- and window-matched variant of the subject x stimulus holdout splits.

The stimulus-seen and stimulus-held-out conditions remove the same number of stimulus
clips, but clip durations and clip ratings differ, so the two conditions of a fold differ
in the number of training windows and in their class composition. For each fold and each
class, this script subsamples (without replacement) the train and val windows of both
conditions to the smaller of the two per-class counts, so both conditions have identical
per-class window counts. Test rows are left unchanged and are checked to be identical
between conditions. The output split files keep their names and protocols and are written
to a separate split root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOLS = ("loso_stimulus_seen", "loso_stimulus_heldout")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subsample(part: pd.DataFrame, quota: dict[str, int], rng: np.random.Generator) -> pd.DataFrame:
    """Subsample each class of ``part`` to ``quota[class]`` rows."""
    keep = []
    for label, g in part.groupby("target_label"):
        n = int(quota.get(label, 0))
        keep.append(g if n >= len(g) else g.iloc[np.sort(rng.choice(len(g), size=n, replace=False))])
    return pd.concat(keep).sort_values("row_index")


def match(frames: dict[str, pd.DataFrame], seed: int) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    out = {p: [] for p in PROTOCOLS}
    report = []
    for fold in sorted(frames[PROTOCOLS[0]]["fold"].unique()):
        parts = {p: frames[p][frames[p]["fold"] == fold] for p in PROTOCOLS}
        tests = [set(parts[p].loc[parts[p]["split"] == "test", "row_index"]) for p in PROTOCOLS]
        if tests[0] != tests[1]:
            raise AssertionError(f"fold {fold}: test rows differ between conditions")
        rng = np.random.default_rng(seed + int(fold))
        row = {"fold": int(fold), "test_rows": len(tests[0])}
        quotas = {}
        for split in ("train", "val"):
            counts = {p: parts[p].loc[parts[p]["split"] == split, "target_label"].value_counts() for p in PROTOCOLS}
            labels = sorted(set(counts[PROTOCOLS[0]].index) | set(counts[PROTOCOLS[1]].index))
            quotas[split] = {c: min(int(counts[p].get(c, 0)) for p in PROTOCOLS) for c in labels}
            for p in PROTOCOLS:
                row[f"{split}_rows_{p}_original"] = int(counts[p].sum())
                row[f"{split}_classes_{p}_original"] = json.dumps({c: int(counts[p].get(c, 0)) for c in labels})
            row[f"{split}_rows_matched"] = int(sum(quotas[split].values()))
            row[f"{split}_classes_matched"] = json.dumps(quotas[split])
        for p in PROTOCOLS:
            kept = [parts[p][parts[p]["split"] == "test"]]
            for split in ("train", "val"):
                kept.append(subsample(parts[p][parts[p]["split"] == split], quotas[split], rng))
            fold_df = pd.concat(kept)
            for split in ("train", "val"):
                got = fold_df.loc[fold_df["split"] == split, "target_label"].value_counts().to_dict()
                if {c: int(v) for c, v in got.items() if v} != {c: n for c, n in quotas[split].items() if n}:
                    raise AssertionError(f"fold {fold} {p} {split}: class counts not matched")
            out[p].append(fold_df)
        report.append(row)
    return {p: pd.concat(v, ignore_index=True) for p, v in out.items()}, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=Path("data/processed/splits/torcheeg"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/splits/torcheeg_window_matched"))
    parser.add_argument("--tasks", default="dreamer:valence,dreamer:arousal,seed_iv:emotion")
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for task in args.tasks.split(","):
        dataset, target = task.split(":")
        paths = {p: args.split_root / f"{dataset}__{target}__{p}.csv" for p in PROTOCOLS}
        frames = {p: pd.read_csv(path) for p, path in paths.items()}
        matched, report = match(frames, args.seed)
        for p, frame in matched.items():
            out = args.output_root / paths[p].name
            frame.to_csv(out, index=False)
            summary.append({"dataset": dataset, "target": target, "protocol": p, "path": str(out),
                            "source_split_sha256": file_sha256(paths[p]), "split_sha256": file_sha256(out)})
        pd.DataFrame(report).assign(dataset=dataset, target=target).to_csv(
            args.output_root / f"{dataset}__{target}__window_matching.csv", index=False)
        print(dataset, target, "folds", len(report))
    summary_path = args.output_root / "window_matched_summary.json"
    if summary_path.exists():  # keep entries of tasks generated in an earlier call
        written = {(s["dataset"], s["target"], s["protocol"]) for s in summary}
        summary = [s for s in json.loads(summary_path.read_text())
                   if (s["dataset"], s["target"], s["protocol"]) not in written] + summary
    summary_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
