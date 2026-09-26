"""Per-network, group-safe linear confound probes.

  * one probe per frozen encoder, in that encoder's own coordinate system;
  * probe split assigned by whole groups (default ``trial_key``) before fitting;
  * closed-set queries: every scored test class must have training support, and a
    query without at least two groups per class is NOT_ESTIMABLE (never "chance");
  * scaler/probe fitted on probe-training rows only;
  * counts of subjects, trials, chunks and encoders reported separately.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

NOT_ESTIMABLE = "NOT_ESTIMABLE"


@dataclass(frozen=True)
class ProbeQuery:
    """A closed-set probe question.

    split="within_class_groups": whole groups (default trials) are split within each
    class; answers "is the class recoverable from held-out trials of known classes?"
    (subject identity, recording-session identity).
    split="subject_holdout": whole subjects are held out and every class must occur in
    both halves; answers "is the class recoverable for unseen subjects?" (stimulus
    identity). Splitting trials within stimulus classes is not used for stimuli because
    the held-out subject's other trials would sit in training and bias predictions
    toward stimuli already seen with that subject (systematically below chance).
    """

    label_col: str  # e.g. subject_key, stimulus_key, session_key
    group_col: str = "trial_key"
    test_fraction: float = 0.3
    seed: int = 20260926
    min_groups_per_class: int = 2
    split: str = "within_class_groups"


def default_query(label_col: str, seed: int = 20260926) -> ProbeQuery:
    if label_col == "stimulus_key":
        return ProbeQuery(label_col, group_col="subject_key", seed=seed, split="subject_holdout")
    return ProbeQuery(label_col, seed=seed)


def subject_holdout_probe_split(meta: pd.DataFrame, query: ProbeQuery) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    subjects = _group_order(meta["subject_key"].astype(str).unique().tolist(), query.seed)
    n_test = min(max(1, int(round(len(subjects) * query.test_fraction))), len(subjects) - 1) if len(subjects) > 1 else 0
    test_subjects = set(subjects[:n_test])
    in_test = meta["subject_key"].astype(str).isin(test_subjects).to_numpy()
    labels = meta[query.label_col].astype(str)
    shared = set(labels[in_test]) & set(labels[~in_test])
    eligible = labels.isin(shared).to_numpy()
    train, test = eligible & ~in_test, eligible & in_test
    info = {
        "n_eligible_classes": len(shared),
        "n_skipped_classes": int(labels.nunique() - len(shared)),
        "n_train_groups": int(len(subjects) - n_test),
        "n_test_groups": int(n_test),
        "status": "OK" if len(shared) >= 2 and n_test >= 1 else NOT_ESTIMABLE,
    }
    return train, test, info


def _group_order(groups: list[str], seed: int) -> list[str]:
    return sorted(groups, key=lambda g: (zlib.crc32(f"{seed}:{g}".encode("utf-8")) & 0xFFFFFFFF, g))


def group_safe_probe_split(meta: pd.DataFrame, query: ProbeQuery) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return boolean (train, test) masks assigning whole groups to one side.

    Groups are split within each class so closed-set classes appear on both
    sides. Groups spanning several classes of ``label_col`` are rejected because
    they would make the query ill-defined.
    """
    for column in (query.label_col, query.group_col):
        if column not in meta.columns:
            raise KeyError(column)
    pairs = meta[[query.group_col, query.label_col]].astype(str).drop_duplicates()
    if pairs[query.group_col].duplicated().any():
        raise ValueError(f"{query.group_col} groups span multiple {query.label_col} classes")

    test_groups: set[str] = set()
    eligible_classes = []
    skipped_classes = []
    for label, part in pairs.groupby(query.label_col):
        groups = _group_order(part[query.group_col].tolist(), query.seed)
        if len(groups) < query.min_groups_per_class:
            skipped_classes.append(label)
            continue
        n_test = int(round(len(groups) * query.test_fraction))
        n_test = min(max(1, n_test), len(groups) - 1)
        test_groups.update(groups[:n_test])
        eligible_classes.append(label)

    labels = meta[query.label_col].astype(str)
    groups = meta[query.group_col].astype(str)
    eligible = labels.isin(eligible_classes).to_numpy()
    test = eligible & groups.isin(test_groups).to_numpy()
    train = eligible & ~test
    info = {
        "n_eligible_classes": len(eligible_classes),
        "n_skipped_classes": len(skipped_classes),
        "n_train_groups": int(groups[train].nunique()),
        "n_test_groups": int(groups[test].nunique()),
        "status": "OK" if len(eligible_classes) >= 2 else NOT_ESTIMABLE,
    }
    return train, test, info


def fit_linear_probe(x_train, y_train, x_test, y_test, seed: int = 20260926) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_classes = set(np.unique(y_train))
    if not set(np.unique(y_test)) <= train_classes:
        raise ValueError("closed-set probe: test contains classes without training support")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    n_classes = len(np.unique(y_test))
    ba = float(balanced_accuracy_score(y_test, pred))
    values, counts = np.unique(y_test, return_counts=True)
    return {
        "balanced_accuracy": ba,
        "accuracy": float(accuracy_score(y_test, pred)),
        "n_test_classes": int(n_classes),
        "uniform_chance": 1.0 / n_classes,
        "ba_over_uniform_chance": ba * n_classes,
        "test_majority_accuracy": float(counts.max() / counts.sum()),
        "converged": bool(getattr(model[-1], "n_iter_", [0])[0] < 2000),
    }


def probe_one_encoder(embeddings: np.ndarray, meta: pd.DataFrame, query: ProbeQuery) -> dict[str, object]:
    if len(embeddings) != len(meta):
        raise ValueError("embeddings and metadata row counts differ")
    if query.split == "subject_holdout":
        train, test, info = subject_holdout_probe_split(meta, query)
    else:
        train, test, info = group_safe_probe_split(meta, query)
    row: dict[str, object] = {
        "probe_query": query.label_col,
        "probe_split": query.split,
        "group_col": query.group_col,
        "n_chunks": int(len(meta)),
        "n_subjects": int(meta["subject_key"].nunique()) if "subject_key" in meta else None,
        "n_trials": int(meta["trial_key"].nunique()) if "trial_key" in meta else None,
        **info,
    }
    if info["status"] != "OK" or not test.any() or not train.any():
        row["status"] = NOT_ESTIMABLE
        return row
    y = meta[query.label_col].astype(str).to_numpy()
    row.update(fit_linear_probe(embeddings[train], y[train], embeddings[test], y[test], seed=query.seed))
    return row


def probe_encoders(encoders: dict[str, tuple[np.ndarray, pd.DataFrame]], query: ProbeQuery) -> pd.DataFrame:
    """One probe per encoder id; never concatenates coordinates across encoders."""
    rows = []
    for encoder_id, (embeddings, meta) in sorted(encoders.items()):
        row = probe_one_encoder(np.asarray(embeddings), meta.reset_index(drop=True), query)
        row["encoder_id"] = encoder_id
        rows.append(row)
    return pd.DataFrame(rows)
