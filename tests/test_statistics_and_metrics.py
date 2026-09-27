"""Metrics, intervals, boundary policy, class weights, aliases."""

import numpy as np
import pandas as pd
import pytest

from conftest import load_script
from eeg_emotion_validity.baselines import classification_metrics
from eeg_emotion_validity.stats import cohens_dz, paired_differences, t_interval

splits = load_script("make_splits")
runner = load_script("run_deep")
instability = load_script("label_statistics")
summary = load_script("summarize_results")


def test_t_interval_uses_student_multiplier_for_five_seeds():
    values = [0.06, 0.065, 0.07, 0.066, 0.064]
    out = t_interval(values)
    assert out["multiplier"] == pytest.approx(2.7764, abs=1e-4)
    sd = np.std(values, ddof=1)
    assert out["high"] - out["mean"] == pytest.approx(2.7764 * sd / np.sqrt(5), abs=1e-5)
    assert np.isnan(t_interval([0.1])["low"])


def test_summarize_results_pairs_subjects_and_reports_t_interval():
    seen = [0.40, 0.42, 0.38, 0.45, 0.41, 0.39]
    held = [0.30, 0.33, 0.31, 0.36, 0.30, 0.32]
    base = {"kind": "deep", "dataset": "seed_iv", "target": "emotion", "label_policy_id": "p",
            "label_variant": "primary", "split_variant": "primary", "model": "m", "transform": "none", "epochs": 5,
            "batch_size": 128, "early_stopping": False, "class_weighting": "none", "inject_snr": 0.0, "seed": 1}
    runs = pd.DataFrame([{**base, "run_id": "a", "protocol": "loso_stimulus_seen", "split_partition_sha256": "s"},
                         {**base, "run_id": "b", "protocol": "loso_stimulus_heldout", "split_partition_sha256": "h"}])
    rows = pd.DataFrame([{"run_id": r, "seed": 1, "fold": i + 1, "balanced_accuracy": ba, "n_test_classes": 4}
                         for r, bas in (("a", seen), ("b", held)) for i, ba in enumerate(bas)])
    est = rows[["run_id", "seed", "fold"]].assign(estimate="stimulus_holdout:seed_iv:emotion:m",
                                                   analysis="stimulus_holdout", holm_family="stimulus_holdout",
                                                   role=lambda f: f["run_id"].map({"a": "A", "b": "B"}))
    table = summary.paired_estimates(summary.validate(runs, rows, est))
    row = table.iloc[0]
    diffs = [s - h for s, h in zip(seen, held)]
    assert row["n_pairs"] == 6
    assert row["mean_diff"] == pytest.approx(sum(diffs) / 6)
    assert row["diff_t95_low"] == pytest.approx(t_interval(diffs)["low"])
    assert "p_holm" in table

def test_paired_differences_reject_unmatched_units_and_dz():
    with pytest.raises(ValueError):
        paired_differences({"s1": 1.0, "s2": 2.0}, {"s1": 1.0, "s3": 2.0})
    diff = paired_differences({"a": 0.6, "b": 0.7, "c": 0.5}, {"a": 0.5, "b": 0.5, "c": 0.5})
    assert cohens_dz(diff) == pytest.approx(diff.mean() / diff.std(ddof=1))


def test_balanced_accuracy_hand_fixture_and_absent_class_denominator():
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 1, 1]
    metrics = classification_metrics(y_true, y_pred, labels=[0, 1])
    assert metrics["balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert metrics["confusion_matrix"] == [[2, 1], [0, 1]]
    single = classification_metrics([0, 0, 0], [0, 1, 0], labels=[0, 1])
    assert single["n_test_classes"] == 1 and single["ba_class_denominator"] == 1
    assert single["balanced_accuracy"] == pytest.approx(single["accuracy"])


def test_boundary_policy_ties_and_inclusive_endpoints():
    frame = pd.DataFrame(
        {"subject_id": range(6), "trial_id": 1, "valence": [3.0, 3.0, 2.75, 3.25, 4.0, 1.0], "arousal": 3.0}
    )
    summary, _, _ = instability.analyze_continuous_dataset(frame, "dreamer", "valence")
    row = summary[summary["delta"] == 0.25].iloc[0]
    assert row["at_threshold_count"] == 2
    assert row["boundary_count"] == 4  # |r - 3| <= 0.25 includes the 2.75 / 3.25 endpoints
    assert row["boundary_excluding_ties_count"] == 2
    assert row["nominal_high_count"] == 2  # 3.25 and 4.0; ties are low


def test_training_class_weights_match_hand_computation():
    train = pd.DataFrame({"target_label": ["a"] * 6 + ["b"] * 2})
    weights = runner.training_class_weights(train, {"a": 0, "b": 1}, "balanced")
    assert weights == pytest.approx([8 / (2 * 6), 8 / (2 * 2)])
    assert runner.training_class_weights(train, {"a": 0, "b": 1}, "none") is None


def test_loso_is_an_alias_of_strict_loso(rating_manifest):
    standard = splits.standardize_manifest(rating_manifest, "deap", "valence")
    loso = splits.build_assignments(standard, "loso", n_splits=2, seed=3)
    strict = splits.build_assignments(standard, "strict_loso", n_splits=2, seed=3)
    cols = ["fold", "split", "row_index"]
    pd.testing.assert_frame_equal(
        loso[cols].sort_values(cols).reset_index(drop=True),
        strict[cols].sort_values(cols).reset_index(drop=True),
    )
    assert splits.PROTOCOL_ALIASES["loso"] == "strict_loso"
    for _, fold in strict.groupby("fold"):
        test_subjects = set(fold.loc[fold["split"] == "test", "subject_key"])
        assert len(test_subjects) == 1
        assert not test_subjects & set(fold.loc[fold["split"] == "val", "subject_key"])
