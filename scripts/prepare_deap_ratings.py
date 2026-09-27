"""Build the DEAP manifest with valence/arousal from the provider rating table.

Valence and arousal are taken from Metadata/participant_ratings.xls, joined on participant
and Experiment_id (the preprocessed trials are in Experiment_id order). The join is verified
through the dominance and liking ratings, the values stored in the preprocessed files are kept
as ``valence_preprocessed``/``arousal_preprocessed`` for reference, and ``rating_source`` is set
so that downstream code can require it. DREAMER and SEED-IV manifests are linked unchanged
into the same directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RATING_SOURCE = "deap_provider_participant_ratings"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/processed/torcheeg/manifests"))
    parser.add_argument("--ratings-xls", type=Path,
                        default=Path("data/raw/deap/Metadata/participant_ratings.xls"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/manifests"))
    args = parser.parse_args()

    original = pd.read_csv(args.manifest_root / "deap_torcheeg_manifest.csv")
    ratings = pd.read_excel(args.ratings_xls)
    ratings = ratings.assign(subject_id=ratings["Participant_id"].map(lambda p: f"s{int(p):02d}.dat"),
                             trial_id=ratings["Experiment_id"].astype(int) - 1)
    ratings = ratings[["subject_id", "trial_id", "Valence", "Arousal", "Dominance", "Liking"]]
    merged = original.merge(ratings, on=["subject_id", "trial_id"], how="left", validate="many_to_one")
    if merged["Valence"].isna().any():
        raise SystemExit("some manifest rows have no provider rating")
    if not (np.allclose(merged["dominance"], merged["Dominance"], atol=0.011)
            and np.allclose(merged["liking"], merged["Liking"], atol=0.011)):
        raise SystemExit("dominance/liking do not match the rating table: trial mapping is not verified")
    manifest = original.copy()
    manifest["valence_preprocessed"] = original["valence"]
    manifest["arousal_preprocessed"] = original["arousal"]
    manifest["valence"] = merged["Valence"].to_numpy()
    manifest["arousal"] = merged["Arousal"].to_numpy()
    manifest["rating_source"] = RATING_SOURCE
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / "deap_torcheeg_manifest.csv"
    manifest.to_csv(out, index=False)
    for other in ("dreamer_torcheeg_manifest.csv", "seed_iv_torcheeg_manifest.csv"):
        link = args.output_root / other
        if not link.exists():
            os.symlink((args.manifest_root / other).resolve(), link)
    trials = manifest.drop_duplicates(["subject_id", "trial_id"])
    info = {
        "rating_source": RATING_SOURCE,
        "input_manifest_sha256": sha256(args.manifest_root / "deap_torcheeg_manifest.csv"),
        "ratings_xls_sha256": sha256(args.ratings_xls),
        "output_manifest_sha256": sha256(out),
        "n_trials": int(len(trials)),
        "valence_range": [float(trials["valence"].min()), float(trials["valence"].max())],
        "high_valence_fraction_gt5": float((trials["valence"] > 5).mean()),
        "high_arousal_fraction_gt5": float((trials["arousal"] > 5).mean()),
    }
    (args.output_root / "deap_rating_provenance.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
