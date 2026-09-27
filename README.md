# Subject-Stimulus-Signal-Controls-EEG-Emotion-Recognition

The code evaluates EEG emotion recognition models on DEAP, DREAMER, and SEED-IV under
controls that separate what a benchmark score depends on:

- **Evaluation protocols**: segment-random, trial-held-out, and leave-one-subject-out (LOSO)
  splits, plus a matched **subject × stimulus holdout** in which test stimuli are either
  withheld from, or available in, the training data of other participants.
- **Label construction**: explicit, versioned label policies (threshold and tie rule) with
  sensitivity variants.
- **Signal transforms**: sample shuffling, time reversal, and multichannel phase-randomized
  surrogates with verified preservation properties.
- **Confound probes**: linear probes of subject, session, and stimulus identity in spectral
  features and in each trained network, using group-safe splits.
- **Cross-dataset transfer** between DEAP and DREAMER without adaptation and with label-free
  alignment (z-normalization, CORAL).
- **Controls**: an injected-signal positive control, training-fit diagnostics, and a
  label-permutation null.

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[deep,test]"      # omit "deep" for the classical pipeline only
pytest                             # unit and synthetic tests; no EEG data required
```

## Data

The datasets are not redistributed. Obtain them from their providers under their licence
agreements and place them as follows (paths can be overridden on the command line):

```text
data/raw/deap/data_preprocessed_python/s01.dat ... s32.dat
data/raw/deap/Metadata/participant_ratings.xls
data/raw/dreamer/DREAMER.mat
data/raw/seed_iv/eeg_raw_data/{1,2,3}/*.mat
```

**DEAP ratings.** Valence and arousal are taken from the provider rating table
(`Metadata/participant_ratings.xls`), joined on participant and video.
`scripts/prepare_deap_ratings.py` builds the DEAP manifest used by all later steps and checks
the join through the dominance and liking ratings.

## Reproducing the analyses

Commands are run from the repository root with `PYTHONPATH=src` (or after `pip install -e .`).

```bash
# 1. Window the recordings (1-s windows for DEAP/DREAMER, 4-s for SEED-IV) and build manifests
python scripts/ingest_datasets.py --dataset all --output-root data/processed/torcheeg
python scripts/prepare_deap_ratings.py --output-root data/processed/manifests
#    (copies/links the DREAMER and SEED-IV manifests next to the DEAP manifest)

# 2. Splits: every file declares its target and label policy
python scripts/make_splits.py --manifest-root data/processed/manifests --output-root data/processed/splits
python scripts/make_stimulus_holdout_splits.py --manifest-root data/processed/manifests \
    --output-root data/processed/splits
python scripts/make_splits.py --dataset dreamer --protocol strict_loso --label-variant tie_high \
    --manifest-root data/processed/manifests --output-root data/processed/splits

# 3. Band-power features and classical models
python scripts/extract_bandpower_features.py --dataset deap --output-root data/processed/features
python scripts/run_classical.py --dataset deap --target valence --protocol strict_loso \
    --split-root data/processed/splits --feature-root data/processed/features \
    --output runs/classical/deap__valence__strict_loso.json --predictions-dir runs/classical/predictions

# 4. Deep models (Braindecode). Optional: export windows once for faster I/O
python scripts/export_window_arrays.py --dataset seed_iv
python scripts/run_deep.py --dataset seed_iv --target emotion --protocol strict_loso \
    --model shallow_convnet --temporal-ablation none --epochs 5 --batch-size 128 --seed 20260530 \
    --split-root data/processed/splits --preload --predictions-dir runs/deep/predictions \
    --output runs/deep/seed_iv_shallow_none.json
#    --temporal-ablation {time_shuffle,time_reverse,phase_random}
#    --protocol {loso_stimulus_seen,loso_stimulus_heldout}
#    --inject-class-signal-snr 0.5 --eval-train        (positive control)
#    --representation-dir runs/deep/representations     (for network probes)

# 5. Probes, transfer, label statistics
python scripts/run_feature_probes.py --feature-root data/processed/features
python scripts/run_network_probes.py --representation-root runs/deep/representations
python scripts/run_transfer_pooled.py --target valence --manifest-root data/processed/manifests
python scripts/run_transfer_aligned.py --source-dataset deap --target-dataset dreamer \
    --manifest-root data/processed/manifests
python scripts/run_transfer_deep.py --source-dataset dreamer --target-dataset deap --target valence \
    --manifest-root data/processed/manifests
python scripts/label_statistics.py --manifest-root data/processed/manifests

# 6. Subject-level summaries and paired comparisons
python scripts/summarize_results.py --runs runs --output-dir results/summary
```

Every run writes a JSON file with per-fold metrics, class support, confusion matrices, the
full training configuration, package versions, and a fingerprint of the target, label
policy, split file, and settings. Fold results are resumed only when this fingerprint
matches, and `scripts/check_run_complete.py` verifies that a run is complete before a batch
launcher skips it.

## Reported results

`results/` contains the per-subject balanced accuracies and the subject-level summaries reported in
the article (see `results/README.md`).

## Evaluation units

For subject-held-out protocols the held-out subject is the unit of analysis: balanced
accuracy is computed over that subject's test windows, averaged over seeds, and summarized
across subjects with Student-t intervals. Paired comparisons use per-subject differences and
two-sided Wilcoxon signed-rank tests with Holm correction within a family of comparisons.
Subjects whose test data contain a single class are excluded from balanced-accuracy
comparisons.

## Repository layout

```text
src/eeg_emotion_validity/   labels, probes, statistics, models, features, signal loading
scripts/                    data preparation, splits, training, probes, transfer, summaries
tests/                      unit and synthetic tests of the contracts above
results/                    per-subject and summary results reported in the article
```
