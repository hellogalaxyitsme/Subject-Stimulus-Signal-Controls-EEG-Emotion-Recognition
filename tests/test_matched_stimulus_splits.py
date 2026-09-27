"""Class- and window-matched subject x stimulus splits."""

import numpy as np
import pandas as pd
import pytest

from conftest import load_script

matched = load_script("make_matched_stimulus_splits")


def condition_frames(rng):
    """Two conditions of three folds whose train sets differ in size and class composition."""
    frames = {}
    for protocol, bias in (("loso_stimulus_seen", 0), ("loso_stimulus_heldout", 1)):
        rows, index = [], 0
        for fold in (1, 2, 3):
            for split, n in (("train", 200 + 30 * bias * fold), ("val", 50), ("test", 40)):
                labels = rng.choice(["high", "low"], size=n, p=[0.3 + 0.2 * bias, 0.7 - 0.2 * bias])
                for label in labels:
                    rows.append({"row_index": index if split != "test" else 10_000 + fold * 100 + len(rows) % 40,
                                 "fold": fold, "split": split, "target_label": label})
                    index += 1
        frames[protocol] = pd.DataFrame(rows)
    # test rows must be identical between the conditions
    frames["loso_stimulus_heldout"].loc[frames["loso_stimulus_heldout"]["split"] == "test", "row_index"] = \
        frames["loso_stimulus_seen"].loc[frames["loso_stimulus_seen"]["split"] == "test", "row_index"].to_numpy()
    return frames


def test_matched_conditions_have_identical_per_class_counts(rng):
    frames = condition_frames(rng)
    out, report = matched.match(frames, seed=1)
    for split in ("train", "val"):
        counts = {p: out[p][out[p]["split"] == split].groupby(["fold", "target_label"]).size() for p in out}
        assert counts["loso_stimulus_seen"].equals(counts["loso_stimulus_heldout"])
    for p in out:  # subsampling only removes rows; test rows are untouched
        assert set(out[p]["row_index"]) <= set(frames[p]["row_index"])
        assert out[p][out[p]["split"] == "test"]["row_index"].tolist() == \
            frames[p][frames[p]["split"] == "test"]["row_index"].tolist()
    assert len(report) == 3


def test_different_test_rows_are_rejected(rng):
    frames = condition_frames(rng)
    frames["loso_stimulus_heldout"].loc[frames["loso_stimulus_heldout"]["split"] == "test", "row_index"] += 1
    with pytest.raises(AssertionError, match="test rows differ"):
        matched.match(frames, seed=1)


def test_matching_is_deterministic(rng):
    frames = condition_frames(np.random.default_rng(3))
    a, _ = matched.match(frames, seed=7)
    b, _ = matched.match(frames, seed=7)
    for p in a:
        assert a[p]["row_index"].tolist() == b[p]["row_index"].tolist()
