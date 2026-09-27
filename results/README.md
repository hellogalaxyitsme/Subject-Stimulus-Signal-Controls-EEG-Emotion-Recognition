# Results

Results reported in the article. Every value in the manuscript tables can be regenerated from these
files with `python scripts/summarize_results.py --results results --output-dir results/summary_check`;
no EEG data are needed for this step.

- `runs.csv`: one row per run that contributes to a reported value. `run_id` identifies the run;
  the remaining columns are every setting that defines its condition (dataset, target, label policy,
  protocol, split variant, model, signal transform, epochs, batch size, early stopping, loss weighting,
  injected-signal amplitude), the seed, the split file with its SHA-256 hash, the SHA-256 of the
  partition it defines (sorted fold, split, and row-index triples; identical for files that assign
  the same rows to the same folds), and, where the runner recorded them, the SHA-256 of the label
  vector and of the full training configuration. `condition_sha256` hashes the condition columns
  and the partition hash.
- `per_subject_deep.csv`, `per_subject_classical.csv`: one row per (`run_id`, `seed`, `fold`). For
  leave-one-subject-out protocols `fold` indexes the held-out subject; for `segment_random` and
  `trial_within_subject` it is a cross-validation fold.
- `estimate_rows.csv`: the rows that form each reported estimate. `role` is `single` for a
  one-condition estimate, `A`/`B` for the two conditions of a paired contrast (A minus B), or the
  protocol name for the protocol decomposition; `holm_family` is the family for Holm correction.
- `transfer_per_subject.csv`: per-target-subject balanced accuracy for DEAP-DREAMER transfer
  (`seed_index` distinguishes EEGNet seeds).
- `probes_per_network.csv`: linear-probe balanced accuracy for each network (fold) and query,
  in the learned representation and in spectral features of the same validation windows.
- `summary/`: the summaries as reported (Student-t intervals across held-out subjects, paired
  differences, Wilcoxon signed-rank tests with Holm correction).

Split files are not redistributed; `scripts/make_splits.py`, `scripts/make_stimulus_holdout_splits.py`,
and `scripts/make_matched_stimulus_splits.py` regenerate them from the dataset manifests with the
seeds in their defaults, and the hashes in `runs.csv` identify the files used. Runs marked
`split_variant = class_window_matched` use the class- and window-matched stimulus-holdout splits.
DEAP valence and arousal use the provider rating table (`label_policy_id` ending in `_v2`).
