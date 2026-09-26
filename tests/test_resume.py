"""Stale or incompatible fold/run outputs must never be reused silently."""

import json

from conftest import load_script
from test_labels_and_splits import runner_args, write_split

runner = load_script("run_deep")
checker = load_script("check_run_complete")


def prepared(tmp_path, manifest, target="arousal", **overrides):
    write_split(tmp_path, manifest, "deap", target)
    args = runner_args(tmp_path, target)
    for key, value in overrides.items():
        setattr(args, key, value)
    runner.prepare_split(args)
    args.output = tmp_path / "out.json"
    args.fold_checkpoint_dir = tmp_path / "folds"
    return args


def test_fold_checkpoint_roundtrip_and_rejections(tmp_path, rating_manifest):
    args = prepared(tmp_path, rating_manifest)
    runner.save_fold_checkpoint(args, 1, {"fold": 1, "balanced_accuracy": 0.5})
    assert runner.load_fold_checkpoint(args, 1) is not None
    assert runner.load_fold_checkpoint(args, 2) is None

    for field, value in [("batch_size", 16), ("lr", 5e-4), ("class_weighting", "balanced"), ("seed", 2)]:
        other = prepared(tmp_path, rating_manifest, **{field: value})
        other.fold_checkpoint_dir = args.fold_checkpoint_dir
        assert runner.load_fold_checkpoint(other, 1) is None, field


def test_checkpoint_from_other_target_is_not_reused(tmp_path, rating_manifest):
    valence_args = prepared(tmp_path, rating_manifest, target="valence")
    runner.save_fold_checkpoint(valence_args, 1, {"fold": 1})
    arousal_args = prepared(tmp_path, rating_manifest, target="arousal")
    arousal_args.fold_checkpoint_dir = valence_args.fold_checkpoint_dir
    assert runner.load_fold_checkpoint(arousal_args, 1) is None


def test_fold_checkpoint_without_fingerprint_is_not_reused(tmp_path, rating_manifest):
    args = prepared(tmp_path, rating_manifest)
    path = runner.fold_checkpoint_path(args, 1)
    path.parent.mkdir(parents=True)
    old_metadata = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, type(None)))}
    old_metadata["fold"] = 1
    path.write_text(json.dumps({"metadata": old_metadata, "fold_metrics": {"fold": 1}}))
    assert runner.load_fold_checkpoint(args, 1) is None


def _run_json(tmp_path, metadata, folds):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"metadata": metadata, "fold_metrics": folds}))
    return path


def test_completion_checker(tmp_path, rating_manifest):
    args = prepared(tmp_path, rating_manifest)
    good_meta = {
        "run_fingerprint": args.run_fingerprint,
        "is_smoke": False,
        "n_folds_expected": 1,
        "n_folds_completed": 1,
    }
    good = _run_json(tmp_path, good_meta, [{"fold": 1, "classes": ["high_arousal", "low_arousal"]}])
    assert checker.reasons_incomplete(good, "deap", "arousal") == []

    mismatched = _run_json(
        tmp_path, {"target": "arousal"}, [{"fold": 1, "classes": ["high_valence", "low_valence"]}]
    )
    problems = checker.reasons_incomplete(mismatched, "deap", "arousal")
    assert any("without run_fingerprint" in p for p in problems) and any("classes" in p for p in problems)

    smoke = _run_json(tmp_path, {**good_meta, "is_smoke": True}, [{"fold": 1, "classes": []}])
    assert any("smoke" in p for p in checker.reasons_incomplete(smoke, "deap", "arousal"))

    partial = _run_json(tmp_path, {**good_meta, "n_folds_expected": 4}, [{"fold": 1, "classes": []}])
    assert any("folds" in p for p in checker.reasons_incomplete(partial, "deap", "arousal"))
    assert checker.reasons_incomplete(good, "deap", "valence")
