"""Baseline models used before deep EEG models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BaselineResult:
    name: str
    metrics: dict[str, float]
    predictions: list[Any]


def classification_metrics(y_true, y_pred, labels=None) -> dict[str, object]:
    """Fold metrics.

    ``balanced_accuracy`` follows scikit-learn: mean recall over classes present
    in ``y_true``. When a test fold lacks a class (e.g. one DEAP arousal subject),
    that denominator shrinks. With ``labels`` given, the fixed class set, support,
    and confusion matrix are also returned so BA can be recomputed or pooled
    under an explicit convention.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if labels is not None:
        labels = list(labels)
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        support = matrix.sum(axis=1)
        metrics.update(
            {
                "metric_labels": [str(label) for label in labels],
                "confusion_matrix": matrix.astype(int).tolist(),
                "class_support": support.astype(int).tolist(),
                "n_test_classes": int((support > 0).sum()),
                "ba_class_denominator": int((support > 0).sum()),
            }
        )
    return metrics


def majority_baseline(y_train, y_test) -> BaselineResult:
    values, counts = np.unique(np.asarray(y_train), return_counts=True)
    label = values[np.argmax(counts)]
    pred = np.repeat(label, len(y_test))
    return BaselineResult("majority", classification_metrics(y_test, pred), pred.tolist())


def random_prior_baseline(y_train, y_test, seed: int = 20260530) -> BaselineResult:
    rng = np.random.default_rng(seed)
    values, counts = np.unique(np.asarray(y_train), return_counts=True)
    probs = counts / counts.sum()
    pred = rng.choice(values, size=len(y_test), p=probs)
    return BaselineResult("random_prior", classification_metrics(y_test, pred), pred.tolist())


def logistic_baseline(x_train, y_train, x_test, y_test, name: str = "logistic") -> BaselineResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return BaselineResult(name, classification_metrics(y_test, pred), pred.tolist())


def svm_baseline(x_train, y_train, x_test, y_test, name: str = "svm") -> BaselineResult:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", dual="auto", max_iter=5000),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return BaselineResult(name, classification_metrics(y_test, pred), pred.tolist())
