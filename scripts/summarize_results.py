"""Subject-level summaries and paired comparisons from run outputs.

Reads every run JSON written by ``run_deep.py`` or ``run_classical.py`` under ``--runs``
and writes:

  fold_results.csv       one row per run x model x fold (fold = held-out subject for LOSO)
  subject_summary.csv    mean over subjects (seeds averaged within subject), Student-t 95% CI
  paired_comparisons.csv per-subject paired differences for the prespecified contrasts

Units: for subject-held-out protocols the held-out subject is the unit; intervals describe
between-subject variability for the evaluated stimulus set. Subjects whose test data
contain a single class are excluded from balanced-accuracy comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.stats import cohens_dz, t_interval

N_CLASSES = {"deap": 2, "dreamer": 2, "seed_iv": 4}
SUBJECT_PROTOCOLS = {"strict_loso", "loso", "loso_stimulus_seen", "loso_stimulus_heldout"}


def load_folds(runs: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs.rglob("*.json")):
        if "_fold_checkpoints" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = payload.get("metadata") or {}
        fingerprint = meta.get("run_fingerprint") or {}
        for fold in payload.get("fold_metrics") or []:
            rows.append({
                "run": path.stem,
                "dataset": fold.get("dataset", meta.get("dataset")),
                "target": fold.get("target", meta.get("target")),
                "protocol": fold.get("protocol", meta.get("protocol")),
                "model": fold.get("model", meta.get("model")),
                "transform": fold.get("temporal_ablation", meta.get("temporal_ablation", "none")),
                "label_variant": meta.get("label_variant", "primary"),
                "class_weighting": fingerprint.get("class_weighting", "none"),
                "early_stopping": bool(fold.get("early_stopping", False)),
                "seed": fold.get("seed", meta.get("seed")),
                "fold": int(fold["fold"]),
                "n_test_classes": fold.get("n_test_classes"),
                "balanced_accuracy": fold.get("balanced_accuracy"),
                "accuracy": fold.get("accuracy"),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["transform"] = frame["transform"].fillna("none")
        frame["model"] = frame["model"].fillna("unknown")
    return frame


def usable(frame: pd.DataFrame) -> pd.DataFrame:
    k = frame["dataset"].map(N_CLASSES)
    single = pd.to_numeric(frame["n_test_classes"], errors="coerce") < k
    return frame[~single.fillna(False)]


KEYS = ["dataset", "target", "protocol", "model", "transform", "label_variant", "class_weighting", "early_stopping"]


def subject_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = usable(frame[frame["protocol"].isin(SUBJECT_PROTOCOLS)])
    per_subject = frame.groupby(KEYS + ["fold"], dropna=False)["balanced_accuracy"].mean().reset_index()
    rows = []
    for key, group in per_subject.groupby(KEYS, dropna=False):
        interval = t_interval(group["balanced_accuracy"])
        rows.append({**dict(zip(KEYS, key)), "n_subjects": interval["n"], "ba_mean": interval["mean"],
                     "ba_t95_low": interval["low"], "ba_t95_high": interval["high"]})
    return pd.DataFrame(rows)


def paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    from scipy.stats import wilcoxon

    sa = usable(a).groupby("fold")["balanced_accuracy"].mean()
    sb = usable(b).groupby("fold")["balanced_accuracy"].mean()
    d = (sa - sb).dropna()
    interval = t_interval(d)
    nonzero = d[d != 0]
    p = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 5 else float("nan")
    return {"n_pairs": int(len(d)), "mean_a": float(sa.loc[d.index].mean()) if len(d) else np.nan,
            "mean_b": float(sb.loc[d.index].mean()) if len(d) else np.nan, "mean_diff": interval["mean"],
            "diff_t95_low": interval["low"], "diff_t95_high": interval["high"], "d_z": cohens_dz(d.to_numpy()),
            "p_wilcoxon": p}


def holm(p: pd.Series) -> pd.Series:
    order = p.dropna().sort_values()
    adjusted, running = {}, 0.0
    for rank, (idx, value) in enumerate(order.items()):
        running = max(running, min(1.0, (len(order) - rank) * value))
        adjusted[idx] = running
    return pd.Series(adjusted).reindex(p.index)


def paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = ["dataset", "target", "model"]
    for key, g in frame.groupby(base):
        info = dict(zip(base, key))
        seen = g[g["protocol"] == "loso_stimulus_seen"]
        held = g[g["protocol"] == "loso_stimulus_heldout"]
        seeds = set(seen["seed"]) & set(held["seed"])
        seen, held = seen[seen["seed"].isin(seeds)], held[held["seed"].isin(seeds)]
        if len(seen) and len(held):
            rows.append({"family": "stimulus_seen_minus_heldout", **info, **paired(seen, held)})
        loso = g[(g["protocol"] == "strict_loso") & ~g["early_stopping"] & (g["class_weighting"] == "none")
                 & (g["label_variant"] == "primary")]
        intact = loso[loso["transform"] == "none"]
        for transform in ("time_shuffle", "time_reverse", "phase_random"):
            other = loso[loso["transform"] == transform]
            if len(intact) and len(other):
                common = set(intact["seed"]) & set(other["seed"])
                rows.append({"family": "intact_minus_transform", "contrast": transform, **info,
                             **paired(intact[intact["seed"].isin(common)], other[other["seed"].isin(common)])})
        for variant in ("tie_high", "exclude_ties"):
            other = g[(g["protocol"] == "strict_loso") & (g["label_variant"] == variant)]
            primary = g[(g["protocol"] == "strict_loso") & (g["label_variant"] == "primary") & (g["transform"] == "none")]
            if len(other) and len(primary):
                rows.append({"family": "label_policy_minus_primary", "contrast": variant, **info, **paired(other, primary)})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = table.groupby("family")["p_wilcoxon"].transform(holm)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    args = parser.parse_args()
    folds = load_folds(args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "fold_results.csv", index=False)
    subject_summary(folds).to_csv(args.output_dir / "subject_summary.csv", index=False)
    paired_comparisons(folds).to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    print(f"wrote summaries for {len(folds)} fold rows to {args.output_dir}")


if __name__ == "__main__":
    main()
