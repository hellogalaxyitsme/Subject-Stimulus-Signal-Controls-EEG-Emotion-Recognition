"""Per-network, group-safe confound probes on exported representations.

Reads representation exports written by ``run_deep.py --representation-dir`` and uses
only rows of validation subjects, which were excluded from that network's training
(encoder-unseen subjects). For each network and query (subject, stimulus, session), a
linear probe is fitted in that network's own coordinates; spectral features for exactly
the same rows and split are probed alongside for a matched comparison.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_emotion_validity.probes import default_query, probe_one_encoder

NAME = re.compile(
    r"^(?P<dataset>deap|dreamer|seed_iv)_(?P<protocol>strict_loso)_(?P<target>valence|arousal|emotion)_"
    r"(?P<model>eegnet|shallow_convnet|deep_convnet)_(?P<ablation>none|time_shuffle)_fold(?P<fold>\d+)$"
)


def stimulus_keys(meta: pd.DataFrame, dataset: str) -> pd.Series:
    trial = meta["trial_key"].astype(str).str.extract(r"trial_(\d+)$", expand=False)
    if dataset == "seed_iv":
        return meta["session_key"].astype(str) + "::trial_" + trial
    return "trial_" + trial


def probe_file(path: Path, features: dict[str, np.ndarray], queries: list[str], seed: int) -> list[dict]:
    info = NAME.match(path.stem).groupdict()
    meta = pd.read_csv(path.with_suffix(".csv"))
    embeddings = np.load(path)["embeddings"]
    keep = (meta["split"] == "val").to_numpy()
    meta = meta[keep].reset_index(drop=True)
    embeddings = embeddings[keep]
    meta["stimulus_key"] = stimulus_keys(meta, info["dataset"])
    rows = []
    for query in queries:
        if query == "session_key" and meta["session_key"].nunique() < 2:
            continue
        spaces = {"learned": embeddings, "raw_spectral": features[info["dataset"]][meta["row_index"].astype(int).to_numpy()]}
        for space, x in spaces.items():
            row = probe_one_encoder(np.asarray(x, dtype=np.float32), meta, default_query(query, seed))
            row.update(info)
            row.update({"feature_space": space, "encoder_rows": "val_encoder_unseen_subjects",
                        "embedding_dim": int(x.shape[1]), "source_file": path.name})
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=Path("data/processed/features/bandpower"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/per_encoder_probes"))
    parser.add_argument("--queries", default="subject_key,stimulus_key,session_key")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    paths = [p for p in sorted(args.representation_root.rglob("*_fold*.npz"))
             if NAME.match(p.stem) and NAME.match(p.stem)["target"] != "arousal"]
    datasets = sorted({NAME.match(p.stem)["dataset"] for p in paths})
    features = {d: np.load(args.feature_root / f"{d}_spectral.npy", mmap_mode="r") for d in datasets}
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(probe_file)(p, features, queries, args.seed) for p in paths
    )
    frame = pd.DataFrame([row for rows in results for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_encoder_probe_metrics.csv", index=False)
    ok = frame[frame["status"] == "OK"]
    summary = (
        ok.groupby(["dataset", "target", "model", "ablation", "probe_query", "feature_space"])
        .agg(n_encoders=("fold", "nunique"), ba_mean=("balanced_accuracy", "mean"), ba_sd=("balanced_accuracy", "std"),
             chance_mean=("uniform_chance", "mean"), ratio_mean=("ba_over_uniform_chance", "mean"),
             n_test_classes_mean=("n_test_classes", "mean"))
        .reset_index()
    )
    summary.to_csv(args.output_dir / "per_encoder_probe_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(frame["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
