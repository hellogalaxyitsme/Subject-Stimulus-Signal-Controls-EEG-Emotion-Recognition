"""Regenerate the reported summaries from an explicit registry of approved results.

Inputs (all in ``--results``, see results/README.md):
  runs.csv                  approved runs and every condition-defining setting
  per_subject_*.csv         one row per (run_id, seed, fold)
  estimate_rows.csv         which rows form each reported estimate, their role, and Holm family
  transfer_per_subject.csv  per-target-subject transfer results (optional)

Nothing is discovered by scanning directories: a row enters a summary only if the estimate
registry lists it. The inputs are validated before any statistic is computed:
  * required columns and types are present, and metrics lie in [0, 1];
  * every estimate row refers to an approved run and to an existing (run_id, seed, fold) row;
  * no (run_id, seed, fold) row appears twice, and within one estimate role no (seed, fold)
    pair is contributed by two runs;
  * the runs pooled within one estimate role share every condition-defining setting
    (only the seed may differ), so seeds are averaged but conditions are never mixed;
  * the two roles of a paired estimate share dataset and target and differ in at least one
    condition-defining setting.

Run JSON files written by run_deep.py or run_classical.py can be converted into these inputs with
``--runs-registry``: a CSV listing ``run_id`` and ``path`` for each approved run file. Per-fold
checkpoint files (``*_fold_checkpoints/``) are rejected, since their ``fold_metrics`` is a single
fold dictionary rather than the list written to a finished run file.

Units: for subject-held-out protocols the held-out subject (fold) is the unit; seeds are averaged
within subject; intervals are Student-t 95% intervals across subjects.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_emotion_validity.stats import cohens_dz, t_interval

N_CLASSES = {"deap": 2, "dreamer": 2, "seed_iv": 4}
CHANCE = {"deap": 0.5, "dreamer": 0.5, "seed_iv": 0.25}
CONDITION = ["kind", "dataset", "target", "label_policy_id", "label_variant", "protocol", "split_variant", "model",
             "transform", "epochs", "batch_size", "early_stopping", "class_weighting", "inject_snr",
             "split_partition_sha256"]
RUN_COLUMNS = ["run_id"] + CONDITION + ["seed"]
ROW_COLUMNS = ["run_id", "seed", "fold", "balanced_accuracy"]
ESTIMATE_COLUMNS = ["estimate", "analysis", "holm_family", "role", "run_id", "seed", "fold"]


class RegistryError(ValueError):
    """The approved-result registry is inconsistent; no summary is produced."""


# ----------------------------------------------------------------------------- loading
def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RegistryError(f"{name}: missing columns {missing}")


def load_results(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(results / "runs.csv")
    rows = pd.concat([pd.read_csv(p) for p in sorted(results.glob("per_subject_*.csv"))], ignore_index=True)
    estimates = pd.read_csv(results / "estimate_rows.csv")
    return runs, rows, estimates


def validate(runs: pd.DataFrame, rows: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    """Validate the registry and return estimate rows joined to their runs and metrics."""
    _require(runs, RUN_COLUMNS, "runs.csv")
    _require(rows, ROW_COLUMNS, "per-subject results")
    _require(estimates, ESTIMATE_COLUMNS, "estimate_rows.csv")
    if runs["run_id"].duplicated().any():
        raise RegistryError(f"duplicate run_id in runs.csv: {runs.loc[runs['run_id'].duplicated(), 'run_id'].tolist()[:3]}")
    ba = pd.to_numeric(rows["balanced_accuracy"], errors="coerce")
    if ba.isna().any() or ((ba < 0) | (ba > 1)).any():
        raise RegistryError("balanced_accuracy must be a number in [0, 1] for every row")
    for frame in (rows, estimates):
        frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype("Int64")
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    dup = rows.duplicated(["run_id", "seed", "fold"])
    if dup.any():
        raise RegistryError(f"duplicate (run_id, seed, fold) rows: {rows.loc[dup, ['run_id', 'seed', 'fold']].head(3).to_dict('records')}")
    unknown = set(estimates["run_id"]) - set(runs["run_id"])
    if unknown:
        raise RegistryError(f"estimate rows refer to runs not in runs.csv: {sorted(unknown)[:3]}")
    joined = estimates.merge(rows, on=["run_id", "seed", "fold"], how="left", validate="many_to_one")
    if joined["balanced_accuracy"].isna().any():
        missing = joined.loc[joined["balanced_accuracy"].isna(), ["run_id", "seed", "fold"]].head(3)
        raise RegistryError(f"estimate rows without a per-subject result: {missing.to_dict('records')}")
    joined = joined.merge(runs[["run_id"] + CONDITION], on="run_id", how="left")
    for (estimate, role), g in joined.groupby(["estimate", "role"]):
        clash = g.duplicated(["seed", "fold"])
        if clash.any():
            raise RegistryError(f"{estimate} [{role}]: the same (seed, fold) is contributed by several runs")
        mixed = [c for c in CONDITION if g[c].astype(str).nunique() > 1]
        if mixed:
            raise RegistryError(f"{estimate} [{role}]: runs with different conditions would be pooled ({mixed})")
    for estimate, g in joined[joined["role"].isin(["A", "B"])].groupby("estimate"):
        if set(g["role"]) != {"A", "B"}:
            raise RegistryError(f"{estimate}: a paired estimate needs both roles A and B")
        a, b = g[g["role"] == "A"].iloc[0], g[g["role"] == "B"].iloc[0]
        if (a["dataset"], a["target"]) != (b["dataset"], b["target"]):
            raise RegistryError(f"{estimate}: roles A and B differ in dataset or target")
        if all(str(a[c]) == str(b[c]) for c in CONDITION):
            raise RegistryError(f"{estimate}: roles A and B have identical conditions")
    return joined


def partition_sha256(split_path: str | None) -> str:
    """SHA-256 of the partition a split file defines: its sorted (fold, split, row_index) triples."""
    import hashlib

    if not split_path or not Path(split_path).exists():
        raise RegistryError(f"split file {split_path!r} not found; add a split_partition_sha256 column")
    frame = pd.read_csv(split_path, usecols=["row_index", "fold", "split"])[["fold", "split", "row_index"]]
    frame = frame.sort_values(["fold", "split", "row_index"])
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def rows_from_run_files(registry: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build runs/per-subject tables from finished run JSON files listed in ``registry``."""
    listing = pd.read_csv(registry)
    _require(listing, ["run_id", "path"], str(registry))
    runs, rows = [], []
    for rec in listing.to_dict("records"):
        path = Path(rec["path"])
        if any(part.endswith("_fold_checkpoints") for part in path.parts):
            raise RegistryError(f"{path}: per-fold checkpoint files are not run results")
        payload = json.loads(path.read_text(encoding="utf-8"))
        folds = payload.get("fold_metrics")
        if not isinstance(folds, list) or not folds:
            raise RegistryError(f"{path}: 'fold_metrics' must be a non-empty list (is this a checkpoint file?)")
        meta = payload.get("metadata") or {}
        if meta.get("is_smoke"):
            raise RegistryError(f"{path}: smoke runs cannot be approved")
        fp = meta.get("run_fingerprint") or {}
        models = {f.get("model", meta.get("model")) for f in folds}
        if len(models) > 1 and "model" not in rec:
            raise RegistryError(f"{path}: several models in one file; list one run_id per model with a 'model' column")
        model = rec.get("model") or next(iter(models))
        runs.append({
            "run_id": rec["run_id"], "kind": "classical" if "models" in meta else "deep",
            "dataset": meta.get("dataset"), "target": meta.get("target"),
            "label_policy_id": fp.get("label_policy_id", meta.get("label_policy_id")),
            "label_variant": meta.get("label_variant", "primary"), "protocol": meta.get("protocol"),
            "split_variant": rec.get("split_variant", "primary"), "model": model,
            "transform": fp.get("temporal_ablation", meta.get("temporal_ablation", "none")),
            "epochs": fp.get("epochs", meta.get("epochs")), "batch_size": fp.get("batch_size", meta.get("batch_size")),
            "early_stopping": bool(fp.get("early_stopping", meta.get("early_stopping", False))),
            "class_weighting": fp.get("class_weighting", meta.get("class_weighting", "none")),
            "inject_snr": float(fp.get("inject_class_signal_snr", 0.0) or 0.0),
            "seed": meta.get("seed"),
            "split_partition_sha256": rec.get("split_partition_sha256") or partition_sha256(meta.get("split_path")),
        })
        for f in folds:
            if f.get("model", model) != model:
                continue
            rows.append({"run_id": rec["run_id"], "seed": f.get("seed", meta.get("seed")), "fold": int(f["fold"]),
                         "n_test_classes": f.get("n_test_classes"), "balanced_accuracy": f.get("balanced_accuracy"),
                         "train_balanced_accuracy": f.get("train_balanced_accuracy")})
    return pd.DataFrame(runs), pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics
def wilcoxon_p(values: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if len(values) < 5:
        return math.nan
    return float(wilcoxon(values, alternative="two-sided", method="exact" if len(values) <= 25 else "auto").pvalue)


def holm(p: pd.Series) -> pd.Series:
    order = p.dropna().sort_values()
    adjusted, running = {}, 0.0
    for rank, (idx, value) in enumerate(order.items()):
        running = max(running, min(1.0, (len(order) - rank) * value))
        adjusted[idx] = running
    return pd.Series(adjusted, dtype=float).reindex(p.index)


def per_subject(g: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.Series:
    """Seeds averaged within subject (fold)."""
    return g.groupby("fold")[metric].mean()


def _describe(g: pd.DataFrame) -> dict:
    first = g.iloc[0]
    return {"dataset": first["dataset"], "target": first["target"], "model": first["model"]}


def single_estimates(joined: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (estimate, analysis, family), g in joined[joined["role"] == "single"].groupby(
            ["estimate", "analysis", "holm_family"]):
        per = per_subject(g)
        interval = t_interval(per)
        row = {"estimate": estimate, "analysis": analysis, "holm_family": family, **_describe(g),
               "n_subjects": interval["n"], "mean": interval["mean"], "t95_low": interval["low"],
               "t95_high": interval["high"]}
        if analysis == "subject_held_out":
            row["chance"] = CHANCE[row["dataset"]]
            row["p_wilcoxon"] = wilcoxon_p(per.to_numpy() - row["chance"])
        if "train_balanced_accuracy" in g and g["train_balanced_accuracy"].notna().any():
            row["train_ba_mean"] = float(pd.to_numeric(g["train_balanced_accuracy"], errors="coerce").mean())
        out.append(row)
    table = pd.DataFrame(out)
    if "p_wilcoxon" in table:
        table["p_holm"] = np.nan
        tested = table["p_wilcoxon"].notna() | (table["analysis"] == "subject_held_out")
        table.loc[tested, "p_holm"] = table[tested].groupby("holm_family")["p_wilcoxon"].transform(holm)
    return table


def paired_estimates(joined: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (estimate, analysis, family), g in joined[joined["role"].isin(["A", "B"])].groupby(
            ["estimate", "analysis", "holm_family"]):
        a, b = g[g["role"] == "A"], g[g["role"] == "B"]
        seeds = set(a["seed"].dropna()) & set(b["seed"].dropna())
        if seeds:  # pair only seeds present in both conditions
            a, b = a[a["seed"].isin(seeds)], b[b["seed"].isin(seeds)]
        sa, sb = per_subject(a), per_subject(b)
        common = sa.index.intersection(sb.index)
        d = (sa.loc[common] - sb.loc[common]).dropna()
        interval = t_interval(d)
        out.append({"estimate": estimate, "analysis": analysis, "holm_family": family, **_describe(g),
                    "n_seeds": len(seeds) or 1, "n_pairs": int(len(d)), "mean_a": float(sa.loc[d.index].mean()),
                    "mean_b": float(sb.loc[d.index].mean()), "mean_diff": interval["mean"],
                    "diff_t95_low": interval["low"], "diff_t95_high": interval["high"],
                    "d_z": cohens_dz(d.to_numpy()), "n_positive": int((d > 0).sum()),
                    "p_wilcoxon": wilcoxon_p(d.to_numpy())})
    table = pd.DataFrame(out)
    if not table.empty:
        table["p_holm"] = table.groupby("holm_family")["p_wilcoxon"].transform(holm)
    return table


def protocol_estimates(joined: pd.DataFrame) -> pd.DataFrame:
    out = []
    part = joined[joined["analysis"] == "protocol_decomposition"]
    for estimate, g in part.groupby("estimate"):
        row = {"estimate": estimate, **_describe(g)}
        for protocol, p in g.groupby("role"):
            row[f"{protocol}_mean"] = float(p["balanced_accuracy"].mean())
            row[f"{protocol}_n"] = int(p["fold"].nunique())
        row["gap_segment_minus_trial"] = row.get("segment_random_mean", math.nan) - row.get("trial_within_subject_mean", math.nan)
        row["gap_trial_minus_loso"] = row.get("trial_within_subject_mean", math.nan) - row.get("strict_loso_mean", math.nan)
        out.append(row)
    return pd.DataFrame(out)


def transfer_estimates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    out = []
    for key, g in frame.groupby(["source", "target_dataset", "target", "pipeline"]):
        per = g.groupby("target_subject")["ba"].mean()
        interval = t_interval(per)
        out.append({**dict(zip(["source", "target_dataset", "target", "pipeline"], key)), "subj_n": interval["n"],
                    "subj_mean": interval["mean"], "subj_t95_low": interval["low"], "subj_t95_high": interval["high"],
                    "subj_p_wilcoxon": wilcoxon_p(per.to_numpy() - 0.5)})
    table = pd.DataFrame(out)
    table["subj_p_holm"] = holm(table["subj_p_wilcoxon"])
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--runs-registry", type=Path, default=None,
                        help="CSV of run_id,path for run JSON files; replaces runs.csv and per_subject_*.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary_check"))
    args = parser.parse_args()
    runs, rows, estimates = load_results(args.results)
    if args.runs_registry is not None:
        runs, rows = rows_from_run_files(args.runs_registry)
    joined = validate(runs, rows, estimates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"single_estimates": single_estimates(joined), "paired_estimates": paired_estimates(joined),
               "protocol_decomposition": protocol_estimates(joined),
               "transfer": transfer_estimates(args.results / "transfer_per_subject.csv")}
    for name, table in outputs.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows")


if __name__ == "__main__":
    main()
