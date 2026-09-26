"""Take DEAP valence/arousal from the provider rating table (Metadata/participant_ratings.xls).

The redistributed preprocessed DEAP files used by this project store 9 - r instead of
the provider rating r for valence and arousal on every trial in the provider's
high-valence/high-arousal quadrant (439 of 1,280 trials), producing out-of-range
values; dominance and liking are unaltered.
This script writes a manifest whose valence/arousal come from
Metadata/participant_ratings.xls (joined on participant and Experiment_id; the
preprocessed trials are in Experiment_id order), keeps the original values as
``valence_dat``/``arousal_dat``, verifies the join through the unaltered dominance and
liking ratings, and marks ``rating_source`` so downstream code can require it.
DREAMER and SEED-IV manifests are linked unchanged into the same directory.
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
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/manifests_corrected"))
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
        raise SystemExit("dominance/liking disagree: trial mapping is not verified")
    corrected = original.copy()
    corrected["valence_dat"] = original["valence"]
    corrected["arousal_dat"] = original["arousal"]
    corrected["valence"] = merged["Valence"].to_numpy()
    corrected["arousal"] = merged["Arousal"].to_numpy()
    corrected["rating_source"] = RATING_SOURCE
    changed = ~np.isclose(corrected["valence"], corrected["valence_dat"], atol=0.011)
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / "deap_torcheeg_manifest.csv"
    corrected.to_csv(out, index=False)
    for other in ("dreamer_torcheeg_manifest.csv", "seed_iv_torcheeg_manifest.csv"):
        link = args.output_root / other
        if not link.exists():
            os.symlink((args.manifest_root / other).resolve(), link)
    trials = corrected.drop_duplicates(["subject_id", "trial_id"])
    info = {
        "rating_source": RATING_SOURCE,
        "input_manifest_sha256": sha256(args.manifest_root / "deap_torcheeg_manifest.csv"),
        "ratings_xls_sha256": sha256(args.ratings_xls),
        "output_manifest_sha256": sha256(out),
        "windows_changed": int(changed.sum()),
        "trials_changed": int((~np.isclose(trials["valence"], trials["valence_dat"], atol=0.011)).sum()),
        "valence_range": [float(trials["valence"].min()), float(trials["valence"].max())],
        "high_valence_fraction_gt5": float((trials["valence"] > 5).mean()),
        "high_arousal_fraction_gt5": float((trials["arousal"] > 5).mean()),
    }
    (args.output_root / "deap_correction_provenance.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
