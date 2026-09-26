"""Matched subject x stimulus holdout design."""

import pandas as pd
import pytest

from conftest import load_script

gen = load_script("make_stimulus_holdout_splits")
splits = load_script("make_splits")
runner = load_script("run_deep")


def seed_iv_manifest():
    rows = []
    labels = {1: [0, 1, 2, 3] * 6}
    for subject in range(1, 6):
        for trial in range(1, 25):
            for clip in range(2):
                rows.append({"subject_id": subject, "session_id": 1, "trial_id": trial,
                             "emotion": labels[1][trial - 1], "clip_id": f"{subject}_{trial}_{clip}",
                             "_record_id": f"r{subject}"})
    return pd.DataFrame(rows)


@pytest.fixture
def frame():
    return gen.add_stimulus_columns(splits.standardize_manifest(seed_iv_manifest(), "seed_iv", "emotion"), "seed_iv")


def test_groups_are_class_balanced_clips(frame):
    clips = frame.drop_duplicates("stimulus_key")
    counts = clips.groupby(["stimulus_group", "emotion"]).size().unstack()
    assert counts.shape == (6, 4) and (counts == 1).all().all()


def test_heldout_and_seen_share_test_units_and_training_size(frame):
    held = gen.build(frame, "loso_stimulus_heldout", seed=1)
    seen = gen.build(frame, "loso_stimulus_seen", seed=1)
    for fold in held["fold"].unique():
        h, s = held[held["fold"] == fold], seen[seen["fold"] == fold]
        assert set(h.loc[h["split"] == "test", "row_index"]) == set(s.loc[s["split"] == "test", "row_index"])
        assert (h["split"] == "train").sum() == (s["split"] == "train").sum()
        assert set(h.loc[h["split"] == "val", "subject_key"]) == set(s.loc[s["split"] == "val", "subject_key"])
        test_stim = set(h.loc[h["split"] == "test", "stimulus_key"])
        assert not test_stim & set(h.loc[h["split"] != "test", "stimulus_key"])
        assert test_stim <= set(s.loc[s["split"] != "test", "stimulus_key"])


def test_runner_rejects_stimulus_leak(frame):
    held = gen.build(frame, "loso_stimulus_heldout", seed=1)
    fold = held[held["fold"] == 1].copy()
    runner.validate_fold(fold, "loso_stimulus_heldout")
    leaked = fold.copy()
    test_stim = leaked.loc[leaked["split"] == "test", "stimulus_key"].iloc[0]
    idx = leaked.index[(leaked["split"] == "train")][0]
    leaked.loc[idx, "stimulus_key"] = test_stim
    with pytest.raises(AssertionError, match="stimulus leakage"):
        runner.validate_fold(leaked, "loso_stimulus_heldout")
