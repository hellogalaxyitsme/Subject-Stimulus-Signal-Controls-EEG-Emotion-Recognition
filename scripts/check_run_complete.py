"""Semantic completion check for batch launchers that skip finished runs.

Exit 0 only if the run JSON was produced under the current run contract, for the
requested dataset/target, trained on classes of that target, is not a smoke run,
and completed every expected fold. Anything else exits 1 so the run is redone
rather than silently skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eeg_emotion_validity.labels import get_policy


def reasons_incomplete(path: Path, dataset: str, target: str) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return ["missing or empty"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"unreadable JSON: {exc}"]
    metadata = payload.get("metadata") or {}
    folds = payload.get("fold_metrics") or []
    problems = []
    fingerprint = metadata.get("run_fingerprint")
    if not isinstance(fingerprint, dict):
        problems.append("output without run_fingerprint")
    else:
        if fingerprint.get("dataset") != dataset or fingerprint.get("target") != target:
            problems.append("fingerprint dataset/target differs from request")
        if fingerprint.get("label_policy_id") != get_policy(dataset, target).policy_id:
            problems.append("label policy differs")
    if metadata.get("is_smoke", True):
        problems.append("smoke or unknown run type")
    if not folds or metadata.get("n_folds_completed") != metadata.get("n_folds_expected"):
        problems.append("not all folds completed")
    allowed = set(get_policy(dataset, target).classes)
    for fold in folds:
        if not set(map(str, fold.get("classes") or [])) <= allowed:
            problems.append(f"fold {fold.get('fold')} trained on classes {fold.get('classes')}")
            break
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    problems = reasons_incomplete(args.path, args.dataset, args.target)
    if problems:
        print(f"NOT COMPLETE {args.path}: {'; '.join(problems)}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
