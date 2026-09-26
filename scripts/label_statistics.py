"""Analyze label instability and stimulus-label determinism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.labels import require_verified_ratings


DEFAULT_MANIFEST_ROOT = Path("data/processed/torcheeg/manifests")
DEFAULT_OUTPUT_DIR = Path("results/label_instability")
NOMINAL_THRESHOLDS = {"deap": 5.0, "dreamer": 3.0}
SEED_LABEL_NAMES = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}


def unique_trials(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    cols = ["subject_id", "trial_id"]
    if "session_id" in frame:
        cols.append("session_id")
    keep = [col for col in frame.columns if col in set(cols + ["valence", "arousal", "emotion"])]
    return frame[keep].drop_duplicates(cols).reset_index(drop=True)


def binary_label(values: pd.Series, threshold: float, target: str) -> pd.Series:
    return np.where(values.astype(float) > threshold, f"high_{target}", f"low_{target}")


def entropy_from_counts(counts: pd.Series) -> float:
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def analyze_continuous_dataset(frame: pd.DataFrame, dataset: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trials = unique_trials(frame, dataset)
    nominal = NOMINAL_THRESHOLDS[dataset]
    values = trials[target].astype(float)

    thresholds = np.round(np.arange(nominal - 1.0, nominal + 1.0001, 0.25), 2)
    sweep_rows = []
    for threshold in thresholds:
        labels = binary_label(values, threshold, target)
        counts = pd.Series(labels).value_counts()
        sweep_rows.append(
            {
                "dataset": dataset,
                "target": target,
                "threshold": threshold,
                "n_trials": int(len(values)),
                "low_count": int(counts.get(f"low_{target}", 0)),
                "high_count": int(counts.get(f"high_{target}", 0)),
                "high_fraction": float(counts.get(f"high_{target}", 0) / len(values)),
            }
        )

    summary_rows = []
    nominal_labels = pd.Series(binary_label(values, nominal, target))
    nominal_counts = nominal_labels.value_counts()
    for delta in (0.25, 0.5, 1.0):
        lower_labels = pd.Series(binary_label(values, nominal - delta, target))
        upper_labels = pd.Series(binary_label(values, nominal + delta, target))
        # boundary_fraction(delta) = count(|r - tau| <= delta) / eligible trials.
        # Ratings exactly at tau are ties (assigned "low" by the '>' rule) and are
        # reported separately; on the integer DREAMER scale they form the whole
        # delta = 0.25 zone.
        boundary = values.between(nominal - delta, nominal + delta, inclusive="both")
        at_threshold = values == nominal
        summary_rows.append(
            {
                "dataset": dataset,
                "target": target,
                "nominal_threshold": nominal,
                "delta": delta,
                "boundary_rule": "abs(rating - threshold) <= delta",
                "tie_rule": "rating == threshold assigned low",
                "n_trials": int(len(values)),
                "n_unique_ratings": int(values.nunique()),
                "rating_mean": float(values.mean()),
                "rating_std": float(values.std()),
                "nominal_low_count": int(nominal_counts.get(f"low_{target}", 0)),
                "nominal_high_count": int(nominal_counts.get(f"high_{target}", 0)),
                "nominal_high_fraction": float(nominal_counts.get(f"high_{target}", 0) / len(values)),
                "at_threshold_count": int(at_threshold.sum()),
                "at_threshold_fraction": float(at_threshold.mean()),
                "boundary_excluding_ties_count": int((boundary & ~at_threshold).sum()),
                "boundary_count": int(boundary.sum()),
                "boundary_fraction": float(boundary.mean()),
                # policy_flip_fraction(tau-delta, tau+delta): trials whose label differs
                # between the two thresholds; a policy contrast, not a label-noise rate.
                "shift_disagreement_count": int((lower_labels != upper_labels).sum()),
                "shift_disagreement_fraction": float((lower_labels != upper_labels).mean()),
            }
        )

    agreement_rows = []
    for stimulus_id, group in trials.groupby("trial_id"):
        labels = pd.Series(binary_label(group[target].astype(float), nominal, target))
        counts = labels.value_counts()
        agreement_rows.append(
            {
                "dataset": dataset,
                "target": target,
                "stimulus_id": stimulus_id,
                "n_subject_trials": int(len(group)),
                "dominant_label": str(counts.idxmax()),
                "dominant_fraction": float(counts.max() / counts.sum()),
                "label_entropy_bits": entropy_from_counts(counts),
                "rating_mean": float(group[target].astype(float).mean()),
                "rating_std": float(group[target].astype(float).std()),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(sweep_rows), pd.DataFrame(agreement_rows)


def analyze_seed_iv(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = unique_trials(frame, "seed_iv")
    trials["emotion_name"] = trials["emotion"].astype(int).map(SEED_LABEL_NAMES)
    trials["stimulus_key"] = "session_" + trials["session_id"].astype(str) + "::trial_" + trials["trial_id"].astype(str)

    rows = []
    for stimulus_key, group in trials.groupby("stimulus_key"):
        counts = group["emotion_name"].value_counts()
        rows.append(
            {
                "dataset": "seed_iv",
                "stimulus_key": stimulus_key,
                "session_id": group["session_id"].iloc[0],
                "trial_id": group["trial_id"].iloc[0],
                "n_subject_trials": int(len(group)),
                "n_labels": int(counts.size),
                "dominant_label": str(counts.idxmax()),
                "dominant_fraction": float(counts.max() / counts.sum()),
                "label_entropy_bits": entropy_from_counts(counts),
            }
        )
    stimulus = pd.DataFrame(rows).sort_values(["session_id", "trial_id"]).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "dataset": "seed_iv",
                "target": "emotion",
                "n_trials": int(len(trials)),
                "n_subjects": int(trials["subject_id"].nunique()),
                "n_stimulus_keys": int(stimulus["stimulus_key"].nunique()),
                "mean_stimulus_dominant_fraction": float(stimulus["dominant_fraction"].mean()),
                "fully_deterministic_stimulus_fraction": float((stimulus["dominant_fraction"] == 1.0).mean()),
                "interpretation": "Stimulus-assigned labels are deterministic across subjects when dominant_fraction is 1.0.",
            }
        ]
    )
    return summary, stimulus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    sweeps = []
    agreements = []
    for dataset in ("deap", "dreamer"):
        frame = require_verified_ratings(pd.read_csv(args.manifest_root / f"{dataset}_torcheeg_manifest.csv"), dataset)
        for target in ("valence", "arousal"):
            summary, sweep, agreement = analyze_continuous_dataset(frame, dataset, target)
            summaries.append(summary)
            sweeps.append(sweep)
            agreements.append(agreement)

    seed_frame = pd.read_csv(args.manifest_root / "seed_iv_torcheeg_manifest.csv")
    seed_summary, seed_stimulus = analyze_seed_iv(seed_frame)
    summaries.append(seed_summary)

    summary_frame = pd.concat(summaries, ignore_index=True)
    sweep_frame = pd.concat(sweeps, ignore_index=True)
    agreement_frame = pd.concat(agreements, ignore_index=True)

    summary_path = args.output_dir / "label_instability_summary.csv"
    sweep_path = args.output_dir / "threshold_sweep.csv"
    agreement_path = args.output_dir / "stimulus_agreement.csv"
    seed_path = args.output_dir / "seed_iv_stimulus_label_determinism.csv"
    json_path = args.output_dir / "label_instability_summary.json"

    summary_frame.to_csv(summary_path, index=False)
    sweep_frame.to_csv(sweep_path, index=False)
    agreement_frame.to_csv(agreement_path, index=False)
    seed_stimulus.to_csv(seed_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "summary_rows": int(len(summary_frame)),
                "threshold_sweep_rows": int(len(sweep_frame)),
                "stimulus_agreement_rows": int(len(agreement_frame)),
                "seed_iv_stimulus_rows": int(len(seed_stimulus)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {summary_path} with {len(summary_frame)} rows")
    print(f"Wrote {sweep_path} with {len(sweep_frame)} rows")
    print(f"Wrote {agreement_path} with {len(agreement_frame)} rows")
    print(f"Wrote {seed_path} with {len(seed_stimulus)} rows")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
