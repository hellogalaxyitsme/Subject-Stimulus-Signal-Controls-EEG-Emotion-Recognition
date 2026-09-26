"""Target and label-policy contract shared by split generation, training, and analysis.

Split files declare their target and label policy, and every runner validates them
against the requested target before a model is built. Label construction always
starts from source ratings or codes, never from a precomputed label column.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEED_IV_LABEL_NAMES = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}


@dataclass(frozen=True)
class LabelPolicy:
    policy_id: str
    dataset: str
    target: str
    source_column: str
    kind: str  # "binary_threshold" or "categorical"
    threshold: float | None = None
    classes: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.kind == "binary_threshold":
            return (
                f"high_{self.target} if {self.source_column} > {self.threshold:g}; "
                f"ties (== {self.threshold:g}) assigned low_{self.target}"
            )
        return f"{self.source_column} integer codes mapped to {list(self.classes)}"


# Policy ids are versioned: change the id whenever the rule changes so hashes,
# checkpoints, and results produced under a different rule can never be reused.
# DEAP valence/arousal are taken from the provider rating table (see
# scripts/prepare_deap_ratings.py); the manifest must carry rating_source.
DEAP_RATING_SOURCE = "deap_provider_participant_ratings"

LABEL_POLICIES: dict[tuple[str, str], LabelPolicy] = {
    ("deap", "valence"): LabelPolicy(
        "deap_valence_gt5_tie_low_v2", "deap", "valence", "valence", "binary_threshold", 5.0,
        ("high_valence", "low_valence"),
    ),
    ("deap", "arousal"): LabelPolicy(
        "deap_arousal_gt5_tie_low_v2", "deap", "arousal", "arousal", "binary_threshold", 5.0,
        ("high_arousal", "low_arousal"),
    ),
    ("dreamer", "valence"): LabelPolicy(
        "dreamer_valence_gt3_tie_low_v1", "dreamer", "valence", "valence", "binary_threshold", 3.0,
        ("high_valence", "low_valence"),
    ),
    ("dreamer", "arousal"): LabelPolicy(
        "dreamer_arousal_gt3_tie_low_v1", "dreamer", "arousal", "arousal", "binary_threshold", 3.0,
        ("high_arousal", "low_arousal"),
    ),
    ("seed_iv", "emotion"): LabelPolicy(
        "seed_iv_emotion_codes_v1", "seed_iv", "emotion", "emotion", "categorical", None,
        tuple(sorted(SEED_IV_LABEL_NAMES.values())),
    ),
}


PRIMARY = "primary"
# Sensitivity variants. tie_high: rating >= threshold is high.
# exclude_ties: ratings exactly at the threshold are removed (changes the population).
TIE_VARIANTS = ("tie_high", "exclude_ties")


def _variant_policy(base: LabelPolicy, variant: str) -> LabelPolicy:
    stem = base.policy_id.rsplit("_tie_low_v1", 1)[0]
    suffix = {"tie_high": "_tie_high_v1", "exclude_ties": "_ties_excluded_v1"}[variant]
    return LabelPolicy(stem + suffix, base.dataset, base.target, base.source_column, f"binary_threshold:{variant}",
                       base.threshold, base.classes)


class TargetContractError(ValueError):
    """Raised when requested target/label policy and data disagree."""


def get_policy(dataset: str, target: str, variant: str = PRIMARY) -> LabelPolicy:
    if variant != PRIMARY:
        base = get_policy(dataset, target)
        if base.kind != "binary_threshold" or variant not in TIE_VARIANTS:
            raise TargetContractError(f"Label variant {variant!r} unsupported for {dataset}/{target}")
        return _variant_policy(base, variant)
    try:
        return LABEL_POLICIES[(dataset, target)]
    except KeyError as exc:
        supported = sorted(t for d, t in LABEL_POLICIES if d == dataset)
        raise TargetContractError(
            f"Unsupported dataset/target combination {dataset!r}/{target!r}; supported targets: {supported}"
        ) from exc


def require_verified_ratings(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Guard for scripts that read ratings directly from a manifest (transfer, label statistics)."""
    if dataset == "deap" and set(frame.get("rating_source", pd.Series(dtype=object)).unique()) != {DEAP_RATING_SOURCE}:
        raise TargetContractError(
            "DEAP manifest lacks provider ratings; run scripts/prepare_deap_ratings.py and pass its output as --manifest-root"
        )
    return frame


def build_target_labels(frame: pd.DataFrame, dataset: str, target: str, variant: str = PRIMARY) -> pd.Series:
    """Construct labels from source ratings/codes; never from an existing label column.

    For ``exclude_ties`` the returned Series is NaN at ties; callers must drop those rows.
    """
    policy = get_policy(dataset, target, variant)
    if dataset == "deap" and set(frame.get("rating_source", pd.Series(dtype=object)).unique()) != {DEAP_RATING_SOURCE}:
        raise TargetContractError(
            "DEAP labels must come from the provider-rating manifest "
            "(scripts/prepare_deap_ratings.py); the preprocessed-file values disagree with the provider rating table"
        )
    if policy.source_column not in frame.columns:
        raise TargetContractError(f"Source column {policy.source_column!r} missing for {dataset}/{target}")
    values = frame[policy.source_column]
    if values.isna().any():
        raise TargetContractError(
            f"{int(values.isna().sum())} missing {policy.source_column!r} values for {dataset}/{target}; "
            "missing ratings must be excluded explicitly before label construction"
        )
    if policy.kind.startswith("binary_threshold"):
        numeric = pd.to_numeric(values, errors="raise").astype(float)
        high = numeric >= float(policy.threshold) if variant == "tie_high" else numeric > float(policy.threshold)
        labels = pd.Series(np.where(high, f"high_{target}", f"low_{target}"), index=frame.index,
                           name="target_label", dtype=object)
        if variant == "exclude_ties":
            labels[numeric == float(policy.threshold)] = np.nan
        return labels
    codes = pd.to_numeric(values, errors="raise")
    if not np.all(np.equal(np.mod(codes, 1), 0)):
        raise TargetContractError(f"Non-integer {policy.source_column!r} codes for {dataset}")
    mapped = codes.astype(int).map(SEED_IV_LABEL_NAMES)
    if mapped.isna().any():
        bad = sorted(codes[mapped.isna()].unique().tolist())
        raise TargetContractError(f"Unknown {dataset} emotion codes {bad}")
    return mapped.rename("target_label")


def label_vector_sha256(sample_keys: pd.Series, labels: pd.Series) -> str:
    """Order-independent hash of the (sample key -> label) mapping."""
    pairs = pd.DataFrame({"key": sample_keys.astype(str).to_numpy(), "label": labels.astype(str).to_numpy()})
    if pairs["key"].duplicated().any():
        raise TargetContractError("Duplicate sample keys; label hash requires unique keys")
    pairs = pairs.sort_values("key", kind="mergesort")
    payload = "\n".join(pairs["key"] + "\t" + pairs["label"]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_filename(dataset: str, target: str, protocol: str, variant: str = PRIMARY) -> str:
    target_part = target if variant == PRIMARY else f"{target}@{variant}"
    return f"{dataset}__{target_part}__{protocol}.csv"



def resolve_split_path(
    split_root: Path, dataset: str, target: str, protocol: str, split_csv: Path | None = None, variant: str = PRIMARY
) -> Path:
    """Return the target-specific split path."""
    if split_csv is not None:
        return Path(split_csv)
    path = Path(split_root) / split_filename(dataset, target, protocol, variant)
    if path.exists():
        return path
    raise FileNotFoundError(path)


def validate_assignments_target(
    assignments: pd.DataFrame, dataset: str, target: str, variant: str = PRIMARY
) -> LabelPolicy:
    """Reject a split whose declared or implied target differs from the request.

    Checks, in order: declared ``target`` / ``label_policy_id`` columns (required),
    and the class inventory of ``target_label`` against the policy.
    """
    policy = get_policy(dataset, target, variant)
    for column in ("dataset", "target", "label_policy_id", "target_label"):
        if column not in assignments.columns:
            raise TargetContractError(
                f"Split is missing required column {column!r}; splits must declare their target "
                "and label policy (regenerate them with scripts/make_splits.py)"
            )
    declared = {
        "dataset": set(assignments["dataset"].astype(str).unique()),
        "target": set(assignments["target"].astype(str).unique()),
        "label_policy_id": set(assignments["label_policy_id"].astype(str).unique()),
    }
    expected = {"dataset": dataset, "target": target, "label_policy_id": policy.policy_id}
    for column, value in expected.items():
        if declared[column] != {value}:
            raise TargetContractError(
                f"Split declares {column}={sorted(declared[column])} but {value!r} was requested"
            )
    observed = set(assignments["target_label"].astype(str).unique())
    unexpected = observed - set(policy.classes)
    if unexpected:
        raise TargetContractError(
            f"Split target_label classes {sorted(unexpected)} are not valid for {policy.policy_id} "
            f"(allowed {list(policy.classes)})"
        )
    return policy


def verify_labels_against_source(
    assignments: pd.DataFrame,
    source: pd.DataFrame,
    dataset: str,
    target: str,
    key: str = "row_index",
) -> dict[str, int]:
    """Recompute labels from source ratings and join by stable key; report mismatches."""
    rebuilt = build_target_labels(source, dataset, target)
    reference = pd.DataFrame({key: source[key].to_numpy(), "expected_label": rebuilt.to_numpy()})
    if reference[key].duplicated().any():
        raise TargetContractError(f"Source key {key!r} is not unique")
    unique_rows = assignments[[key, "target_label"]].drop_duplicates()
    if unique_rows[key].duplicated().any():
        raise TargetContractError(f"Assignments carry conflicting labels for the same {key!r}")
    joined = unique_rows.merge(reference, on=key, how="left", validate="one_to_one")
    missing = int(joined["expected_label"].isna().sum())
    mismatched = int((joined["expected_label"].notna() & (joined["target_label"] != joined["expected_label"])).sum())
    return {"n_checked": int(len(joined)), "n_missing_in_source": missing, "n_mismatched": mismatched}


def class_mapping(policy: LabelPolicy) -> dict[str, int]:
    """Fixed class->index mapping from the policy, independent of which classes a fold contains."""
    return {label: index for index, label in enumerate(sorted(policy.classes))}
