"""Per-encoder, group-safe, closed-set confound probes."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from eeg_emotion_validity.probes import (  # noqa: E402
    NOT_ESTIMABLE,
    ProbeQuery,
    group_safe_probe_split,
    probe_encoders,
    probe_one_encoder,
)


def meta(n_subjects=6, n_trials=8, n_chunks=10):
    rows = []
    for s in range(n_subjects):
        for t in range(n_trials):
            for c in range(n_chunks):
                rows.append({"subject_key": f"s{s}", "trial_key": f"s{s}::t{t}", "stimulus_key": f"t{t}"})
    return pd.DataFrame(rows)


def subject_embeddings(m, rng, dim=8, signal=3.0):
    codes = m["subject_key"].str[1:].astype(int).to_numpy()
    centers = rng.normal(size=(codes.max() + 1, dim)) * signal
    return centers[codes] + rng.normal(size=(len(m), dim))


def test_no_trial_crosses_probe_split():
    m = meta()
    train, test, info = group_safe_probe_split(m, ProbeQuery("subject_key"))
    assert info["status"] == "OK"
    assert not set(m.loc[train, "trial_key"]) & set(m.loc[test, "trial_key"])
    assert set(m.loc[test, "subject_key"]) <= set(m.loc[train, "subject_key"])


def test_insufficient_support_is_not_estimable_not_chance(rng):
    m = meta(n_subjects=1)  # single subject -> no closed-set subject query
    row = probe_one_encoder(rng.normal(size=(len(m), 4)), m, ProbeQuery("subject_key"))
    assert row["status"] == NOT_ESTIMABLE and "balanced_accuracy" not in row
    m2 = meta(n_trials=1)  # one trial per subject -> cannot hold out a trial
    row2 = probe_one_encoder(rng.normal(size=(len(m2), 4)), m2, ProbeQuery("subject_key"))
    assert row2["status"] == NOT_ESTIMABLE


def test_synthetic_confound_recovered_within_each_encoder(rng):
    m = meta()
    encoders = {f"fold{k}": (subject_embeddings(m, rng), m) for k in range(3)}
    out = probe_encoders(encoders, ProbeQuery("subject_key"))
    assert (out["balanced_accuracy"] > 0.9).all()
    assert out["n_chunks"].eq(len(m)).all() and out["n_trials"].eq(48).all()


def test_rotated_encoders_pooling_vs_per_encoder(rng):
    """Independently rotated encoders: each is decodable; naive pooling degrades."""
    m = meta()
    base = subject_embeddings(m, rng, dim=8)
    encoders = {}
    for k in range(4):
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        encoders[f"fold{k}"] = (base @ q, m)
    per_encoder = probe_encoders(encoders, ProbeQuery("subject_key"))["balanced_accuracy"]
    pooled_x = np.vstack([e for e, _ in encoders.values()])
    pooled_m = pd.concat(
        [mm.assign(trial_key=mm["trial_key"] + f"@{k}") for k, (_, mm) in encoders.items()], ignore_index=True
    )
    pooled = probe_one_encoder(pooled_x, pooled_m, ProbeQuery("subject_key"))["balanced_accuracy"]
    assert per_encoder.min() > 0.9
    assert pooled < per_encoder.min() - 0.2


def test_raw_probe_runner_grouped_mode(rng):
    from conftest import load_script

    raw = load_script("run_feature_probes")
    manifest = pd.DataFrame(
        [{"subject_id": s, "trial_id": t, "clip": c} for s in range(5) for t in range(6) for c in range(4)]
    )
    frame = manifest.reset_index(drop=True)
    frame.insert(0, "row_index", frame.index)
    frame["subject_key"] = frame["subject_id"].astype(str)
    frame["trial_key"] = frame["subject_key"] + "::trial_" + frame["trial_id"].astype(str)
    features = subject_embeddings(frame.assign(subject_key="s" + frame["subject_key"]), rng)
    row = raw.run_probe_grouped(features, frame, "subject_key", seed=1)
    assert row["split_mode"] == "within_class_groups" and row["balanced_accuracy"] > 0.9


def test_stimulus_query_holds_out_subjects_and_recovers_shared_stimulus_signal(rng):
    from eeg_emotion_validity.probes import default_query

    m = meta()  # every subject watched every stimulus (t0..t7)
    query = default_query("stimulus_key")
    assert query.split == "subject_holdout"
    codes = m["stimulus_key"].str[1:].astype(int).to_numpy()
    centers = rng.normal(size=(codes.max() + 1, 8)) * 3
    subject_offset = rng.normal(size=(6, 8))[m["subject_key"].str[1:].astype(int)] * 3
    x = centers[codes] + subject_offset + rng.normal(size=(len(m), 8))
    row = probe_one_encoder(x, m, query)
    assert row["status"] == "OK" and row["n_test_groups"] >= 1
    assert row["balanced_accuracy"] > 3 * row["uniform_chance"]  # recovered despite unseen-subject offsets
    noise = probe_one_encoder(rng.normal(size=(len(m), 8)), m, query)
    assert abs(noise["balanced_accuracy"] - noise["uniform_chance"]) < 0.1


def test_shuffled_label_control_near_chance(rng):
    m = meta()
    emb = rng.normal(size=(len(m), 8))  # no subject information
    row = probe_one_encoder(emb, m, ProbeQuery("subject_key"))
    assert abs(row["balanced_accuracy"] - row["uniform_chance"]) < 0.1
