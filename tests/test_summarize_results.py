"""Aggregation from the approved-result registry: discovery, schema, duplicates, and condition mixing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, load_script

summ = load_script("summarize_results")

BASE = {"kind": "deep", "dataset": "dreamer", "target": "valence", "label_policy_id": "p", "label_variant": "primary",
        "protocol": "strict_loso", "split_variant": "primary", "model": "eegnet", "transform": "none", "epochs": 5,
        "batch_size": 128, "early_stopping": False, "class_weighting": "none", "inject_snr": 0.0,
        "split_partition_sha256": "abc"}


def make_registry(conditions: dict[str, dict], values: dict[str, dict[tuple[int, int], float]], roles: dict[str, str],
                  estimate: str = "e1", analysis: str = "signal_transforms"):
    runs = pd.DataFrame([{"run_id": r, **BASE, **c, "seed": 1} for r, c in conditions.items()])
    rows = pd.DataFrame([{"run_id": r, "seed": s, "fold": f, "balanced_accuracy": v, "n_test_classes": 2}
                         for r, vals in values.items() for (s, f), v in vals.items()])
    est = pd.DataFrame([{"estimate": estimate, "analysis": analysis, "holm_family": analysis, "role": roles[r],
                         "run_id": r, "seed": s, "fold": f} for r, vals in values.items() for (s, f) in vals])
    return runs, rows, est


def test_paired_estimate_averages_seeds_within_subject():
    a = {(1, f): 0.6 for f in range(1, 7)} | {(2, f): 0.8 for f in range(1, 7)}
    b = {(1, f): 0.5 for f in range(1, 7)} | {(2, f): 0.5 for f in range(1, 7)}
    runs, rows, est = make_registry({"a": {}, "b": {"transform": "phase_random"}}, {"a": a, "b": b},
                                    {"a": "A", "b": "B"})
    table = summ.paired_estimates(summ.validate(runs, rows, est))
    assert table.loc[0, "n_pairs"] == 6
    assert table.loc[0, "n_seeds"] == 2
    assert table.loc[0, "mean_diff"] == pytest.approx(0.2)


def test_mixing_conditions_within_a_role_is_rejected():
    """An injected-signal (positive-control) run must never be pooled with ordinary runs."""
    vals = {(1, f): 0.6 for f in range(1, 6)}
    runs, rows, est = make_registry(
        {"plain": {}, "control": {"inject_snr": 0.5}, "other": {"transform": "time_shuffle"}},
        {"plain": vals, "control": {(2, f): 0.9 for f in range(1, 6)}, "other": vals},
        {"plain": "A", "control": "A", "other": "B"})
    with pytest.raises(summ.RegistryError, match="different conditions"):
        summ.validate(runs, rows, est)


@pytest.mark.parametrize("field,value", [("epochs", 30), ("split_partition_sha256", "other-split"),
                                         ("label_policy_id", "p2"), ("early_stopping", True)])
def test_every_condition_field_separates_runs(field, value):
    vals = {(1, f): 0.6 for f in range(1, 6)}
    runs, rows, est = make_registry({"x": {}, "y": {field: value}, "z": {"transform": "time_reverse"}},
                                    {"x": vals, "y": {(2, f): 0.7 for f in range(1, 6)}, "z": vals},
                                    {"x": "A", "y": "A", "z": "B"})
    with pytest.raises(summ.RegistryError, match="different conditions"):
        summ.validate(runs, rows, est)


def test_duplicate_subject_seed_rows_are_rejected():
    vals = {(1, f): 0.6 for f in range(1, 6)}
    runs, rows, est = make_registry({"a": {}, "b": {"transform": "phase_random"}}, {"a": vals, "b": vals},
                                    {"a": "A", "b": "B"})
    rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(summ.RegistryError, match="duplicate"):
        summ.validate(runs, rows, est)


def test_same_subject_and_seed_from_two_runs_is_rejected():
    vals = {(1, f): 0.6 for f in range(1, 6)}
    runs, rows, est = make_registry({"a1": {}, "a2": {}, "b": {"transform": "phase_random"}},
                                    {"a1": vals, "a2": vals, "b": vals}, {"a1": "A", "a2": "A", "b": "B"})
    with pytest.raises(summ.RegistryError, match="several runs"):
        summ.validate(runs, rows, est)


def test_unlisted_runs_never_enter_a_summary():
    vals = {(1, f): 0.6 for f in range(1, 6)}
    runs, rows, est = make_registry({"a": {}, "b": {"transform": "phase_random"}}, {"a": vals, "b": vals},
                                    {"a": "A", "b": "B"})
    stray = pd.DataFrame([{"run_id": "a", "seed": 9, "fold": f, "balanced_accuracy": 0.99} for f in range(1, 6)])
    table = summ.paired_estimates(summ.validate(runs, pd.concat([rows, stray]), est))
    assert table.loc[0, "mean_a"] == pytest.approx(0.6)


def test_identical_conditions_cannot_form_a_contrast():
    vals = {(1, f): 0.6 for f in range(1, 6)}
    runs, rows, est = make_registry({"a": {}, "b": {}}, {"a": vals, "b": vals}, {"a": "A", "b": "B"})
    with pytest.raises(summ.RegistryError, match="identical conditions"):
        summ.validate(runs, rows, est)


def _write_run(path: Path, fold_metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": {"dataset": "dreamer", "target": "valence", "protocol": "strict_loso",
                                             "model": "eegnet", "seed": 1, "run_fingerprint": {}},
                                "fold_metrics": fold_metrics}))


def test_checkpoint_files_are_not_mistaken_for_runs(tmp_path):
    """The runner writes per-fold checkpoints to '<run>_fold_checkpoints/foldNNN.json'."""
    fold = {"fold": 1, "balanced_accuracy": 0.6, "seed": 1}
    ckpt = tmp_path / "runs" / "example_fold_checkpoints" / "fold001.json"
    _write_run(ckpt, fold)
    registry = tmp_path / "registry.csv"
    pd.DataFrame([{"run_id": "r", "path": str(ckpt), "split_partition_sha256": "abc"}]).to_csv(registry, index=False)
    with pytest.raises(summ.RegistryError, match="checkpoint"):
        summ.rows_from_run_files(registry)
    moved = tmp_path / "elsewhere" / "fold001.json"  # checkpoint content outside the checkpoint folder
    _write_run(moved, fold)
    pd.DataFrame([{"run_id": "r", "path": str(moved), "split_partition_sha256": "abc"}]).to_csv(registry, index=False)
    with pytest.raises(summ.RegistryError, match="non-empty list"):
        summ.rows_from_run_files(registry)


def test_finished_run_files_are_read(tmp_path):
    run = tmp_path / "runs" / "example.json"
    _write_run(run, [{"fold": f, "balanced_accuracy": 0.5 + f / 100, "seed": 1, "n_test_classes": 2} for f in (1, 2)])
    registry = tmp_path / "registry.csv"
    pd.DataFrame([{"run_id": "r", "path": str(run), "split_partition_sha256": "abc"}]).to_csv(registry, index=False)
    runs, rows = summ.rows_from_run_files(registry)
    assert list(rows["fold"]) == [1, 2] and runs.loc[0, "split_partition_sha256"] == "abc"


RESULTS = ROOT / "results"


@pytest.mark.skipif(not (RESULTS / "estimate_rows.csv").exists(), reason="released results not present")
def test_released_results_reproduce_reported_summaries():
    runs, rows, est = summ.load_results(RESULTS)
    joined = summ.validate(runs, rows, est)
    paired = summ.paired_estimates(joined).set_index("estimate")
    stim = pd.read_csv(RESULTS / "summary" / "stimulus_holdout.csv")
    for rec in stim.to_dict("records"):
        name = "stimulus_holdout" if rec.get("split_variant", "primary") == "primary" else "stimulus_holdout_matched"
        got = paired.loc[f"{name}:{rec['dataset']}:{rec['target']}:{rec['model']}"]
        assert got["mean_diff"] == pytest.approx(rec["mean_diff"], abs=1e-9)
        assert got["n_pairs"] == rec["n_pairs"]
        assert got["p_holm"] == pytest.approx(rec["p_holm"], abs=1e-9, nan_ok=True)
    single = summ.single_estimates(joined).set_index("estimate")
    above = pd.read_csv(RESULTS / "summary" / "above_chance.csv")
    for rec in above.to_dict("records"):
        got = single.loc[f"subject_held_out:{rec['dataset']}:{rec['target']}:{rec['model']}"]
        assert got["mean"] == pytest.approx(rec["ba_mean"], abs=1e-9)
        assert got["p_holm"] == pytest.approx(rec["p_holm"], abs=1e-9)
    trans = pd.read_csv(RESULTS / "summary" / "transforms.csv")
    for rec in trans.to_dict("records"):
        got = paired.loc[f"signal_transforms:{rec['dataset']}:{rec['target']}:{rec['model']}:{rec['transform']}"]
        assert got["mean_diff"] == pytest.approx(rec["mean_diff"], abs=1e-9)
    assert np.isfinite(paired["mean_diff"]).all()
