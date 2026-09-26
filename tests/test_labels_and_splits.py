"""The requested target must match split labels before any model is built."""

import argparse

import pandas as pd
import pytest

from conftest import load_script
from eeg_emotion_validity.labels import (
    TargetContractError,
    build_target_labels,
    get_policy,
    label_vector_sha256,
    resolve_split_path,
    split_filename,
    validate_assignments_target,
    verify_labels_against_source,
)

splits = load_script("make_splits")
runner = load_script("run_deep")


def write_split(tmp_path, manifest, dataset, target, protocol="strict_loso", name=None):
    standard = splits.standardize_manifest(manifest, dataset, target)
    assignments = splits.build_assignments(standard, protocol, n_splits=2, seed=1)
    path = tmp_path / (name or split_filename(dataset, target, protocol))
    assignments.to_csv(path, index=False)
    return path, assignments


def runner_args(tmp_path, target, split_csv=None):
    return argparse.Namespace(
        dataset="deap", target=target, protocol="strict_loso", split_root=tmp_path, split_csv=split_csv,
        model="eegnet", temporal_ablation="none", epochs=1, batch_size=8, lr=1e-3, weight_decay=1e-4,
        class_weighting="none", seed=1, sfreq=None, max_folds=None, max_train_rows=None,
        max_val_rows=None, max_test_rows=None, early_stopping=False, patience=7, max_epochs=50,
        val_subjects=None, min_delta=0.0,
    )


def test_valence_split_with_arousal_request_fails_before_training(tmp_path, rating_manifest):
    path, _ = write_split(tmp_path, rating_manifest, "deap", "valence")
    args = runner_args(tmp_path, "arousal", split_csv=path)
    with pytest.raises(TargetContractError, match="target"):
        runner.prepare_split(args)
    assert not hasattr(args, "run_fingerprint")


def test_matching_split_passes_and_fingerprints(tmp_path, rating_manifest):
    write_split(tmp_path, rating_manifest, "deap", "arousal")
    args = runner_args(tmp_path, "arousal")
    runner.prepare_split(args)
    assert args.run_fingerprint["label_policy_id"] == "deap_arousal_gt5_tie_low_v2"
    assert args.run_fingerprint["target"] == "arousal"


def test_valence_and_arousal_labels_differ_on_fixture(rating_manifest):
    valence = build_target_labels(rating_manifest, "deap", "valence")
    arousal = build_target_labels(rating_manifest, "deap", "arousal")
    assert set(valence) == {"high_valence", "low_valence"}
    assert set(arousal) == {"high_arousal", "low_arousal"}
    high_v = valence.eq("high_valence").to_numpy()
    high_a = arousal.eq("high_arousal").to_numpy()
    assert (high_v != high_a).any()


def test_split_without_declared_target_is_rejected(tmp_path, rating_manifest):
    path, assignments = write_split(tmp_path, rating_manifest, "deap", "valence", name="deap_strict_loso.csv")
    untagged = assignments.drop(columns=["target", "label_policy_id"])
    untagged.to_csv(path, index=False)
    with pytest.raises(FileNotFoundError):
        resolve_split_path(tmp_path, "deap", "arousal", "strict_loso")
    with pytest.raises(TargetContractError, match="missing required column"):
        validate_assignments_target(untagged, "deap", "valence")


def test_relabelled_header_with_wrong_classes_is_rejected(tmp_path, rating_manifest):
    """A split whose header says arousal but whose labels are valence must fail."""
    _, assignments = write_split(tmp_path, rating_manifest, "deap", "valence")
    forged = assignments.assign(target="arousal", label_policy_id=get_policy("deap", "arousal").policy_id)
    with pytest.raises(TargetContractError, match="not valid"):
        validate_assignments_target(forged, "deap", "arousal")


@pytest.mark.parametrize(
    "dataset,target,rating,expected",
    [
        ("deap", "valence", 5.0, "low_valence"),
        ("deap", "valence", 5.01, "high_valence"),
        ("dreamer", "arousal", 3.0, "low_arousal"),
        ("dreamer", "arousal", 4.0, "high_arousal"),
    ],
)
def test_threshold_ties_assigned_low(dataset, target, rating, expected):
    frame = pd.DataFrame({target: [rating], "rating_source": "deap_provider_participant_ratings"})
    assert build_target_labels(frame, dataset, target).iloc[0] == expected


def test_tie_policy_variants():
    frame = pd.DataFrame({"arousal": [2.0, 3.0, 3.0, 4.0]})
    primary = build_target_labels(frame, "dreamer", "arousal")
    tie_high = build_target_labels(frame, "dreamer", "arousal", "tie_high")
    excluded = build_target_labels(frame, "dreamer", "arousal", "exclude_ties")
    assert list(primary) == ["low_arousal", "low_arousal", "low_arousal", "high_arousal"]
    assert list(tie_high) == ["low_arousal", "high_arousal", "high_arousal", "high_arousal"]
    assert excluded.isna().tolist() == [False, True, True, False]
    assert get_policy("dreamer", "arousal", "tie_high").policy_id == "dreamer_arousal_gt3_tie_high_v1"
    with pytest.raises(TargetContractError):
        get_policy("seed_iv", "emotion", "tie_high")
    standard = splits.standardize_manifest(
        pd.DataFrame({"subject_id": [0, 0, 1, 1], "trial_id": [0, 1, 0, 1], "clip_id": list("abcd"),
                      "_record_id": "r", "valence": [3.0, 3.0, 4.0, 1.0], "arousal": 2.0}),
        "dreamer", "valence", "exclude_ties",
    )
    assert standard["row_index"].tolist() == [2, 3]  # original cache positions retained


def test_deap_labels_require_provider_rating_source():
    frame = pd.DataFrame({"valence": [7.0, 2.0]})
    with pytest.raises(TargetContractError, match="provider-rating manifest"):
        build_target_labels(frame, "deap", "valence")
    with pytest.raises(TargetContractError, match="provider-rating manifest"):
        build_target_labels(frame.assign(rating_source="redistributed_dat"), "deap", "valence")


def test_missing_ratings_invalid_codes_and_unsupported_targets():
    with pytest.raises(TargetContractError, match="missing"):
        build_target_labels(pd.DataFrame({"valence": [1.0, None], "rating_source": "deap_provider_participant_ratings"}), "deap", "valence")
    with pytest.raises(TargetContractError, match="Unknown"):
        build_target_labels(pd.DataFrame({"emotion": [0, 7]}), "seed_iv", "emotion")
    with pytest.raises(TargetContractError, match="Unsupported"):
        get_policy("seed_iv", "valence")
    with pytest.raises(TargetContractError, match="Unsupported"):
        get_policy("deap", "dominance")


def test_label_hash_is_order_independent_and_target_sensitive(rating_manifest):
    keys = rating_manifest["clip_id"]
    valence = build_target_labels(rating_manifest, "deap", "valence")
    arousal = build_target_labels(rating_manifest, "deap", "arousal")
    shuffled = rating_manifest.sample(frac=1.0, random_state=3)
    assert label_vector_sha256(keys, valence) == label_vector_sha256(
        shuffled["clip_id"], build_target_labels(shuffled, "deap", "valence")
    )
    assert label_vector_sha256(keys, valence) != label_vector_sha256(keys, arousal)


def test_source_label_join_survives_reordering_and_reports_mismatches(tmp_path, rating_manifest):
    _, assignments = write_split(tmp_path, rating_manifest, "deap", "arousal")
    source = rating_manifest.reset_index(drop=True).assign(row_index=lambda f: f.index)
    reordered = assignments.sample(frac=1.0, random_state=5)
    report = verify_labels_against_source(reordered, source.sample(frac=1.0, random_state=9), "deap", "arousal")
    assert report == {"n_checked": len(source), "n_missing_in_source": 0, "n_mismatched": 0}
    # Failure mode: an arousal job fed valence labels.
    _, valence_assign = write_split(tmp_path, rating_manifest, "deap", "valence")
    mislabelled = valence_assign.assign(
        target_label=valence_assign["target_label"].str.replace("valence", "arousal")
    )
    report = verify_labels_against_source(mislabelled, source, "deap", "arousal")
    assert report["n_mismatched"] > 0
