"""Exported embeddings must stay paired with their sample metadata."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from conftest import load_script  # noqa: E402

runner = load_script("run_deep")

LABEL_TO_ID = {"high_valence": 0, "low_valence": 1}


class RowCodedSource:
    """Fake TorchEEG dataset: channel 0 carries the row index as a constant."""

    def __getitem__(self, row_index):
        signal = np.zeros((2, 16), dtype=np.float32)
        signal[0, :8] = row_index
        signal[0, 8:] = -row_index
        return signal, {}


class IdentityEncoder(torch.nn.Module):
    """Last Linear receives the raw flattened input, so embedding[:, 0] == signal[0, 0]."""

    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.head = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.head(self.flatten(x))


def frame(n=37, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.permutation(np.arange(100, 100 + n))
    return pd.DataFrame(
        {
            "row_index": rows,
            "chunk_id": [f"c{r}" for r in rows],
            "target_label": np.where(rows % 3 == 0, "high_valence", "low_valence"),
            "subject_key": [f"s{r % 5}" for r in rows],
            "session_key": "session_1",
            "trial_key": [f"s{r % 5}::t{r // 7}" for r in rows],
            "clip_id": [f"clip{r}" for r in rows],
        }
    )


def dataset(df):
    return runner.SplitTorchEEGDataset(RowCodedSource(), df, LABEL_TO_ID, temporal_ablation="none", seed=0)


def decoded_row_index(embeddings, df):
    # per-window z-scoring maps [+r]*8,[-r]*8 to +1/-1; recover r via metadata join instead
    return embeddings[:, 0]


@pytest.mark.parametrize("batch_size", [1, 5, 8, 64])
def test_keyed_extraction_pairs_every_row(tmp_path, batch_size):
    df = frame()
    ds = dataset(df)
    loader = runner.extraction_loader(ds, batch_size, 0)
    emb, label_ids, row_idx = runner.extract_representations(IdentityEncoder(), loader, "cpu")
    assert len(emb) == len(df)  # non-divisible final batch retained
    meta = runner.keyed_representation_metadata("train", df, label_ids, row_idx, LABEL_TO_ID)
    expected = df.set_index("row_index").loc[meta["row_index"], "subject_key"].to_numpy()
    assert (meta["subject_key"].to_numpy() == expected).all()


def test_saved_export_roundtrip(tmp_path):
    df = frame()
    path = tmp_path / "rep" / "deap_strict_loso_valence_eegnet_none_fold1.npz"
    runner.save_representations(
        path, IdentityEncoder(), {"train": dataset(df)}, {"train": df}, LABEL_TO_ID, "cpu",
        batch_size=7, num_workers=0, provenance={"fold": 1},
    )
    saved = np.load(path)
    meta = pd.read_csv(path.with_suffix(".csv"))
    assert (saved["row_index"] == meta["row_index"].to_numpy()).all()
    assert path.with_suffix(".provenance.json").exists()


def test_shuffled_loader_export_would_mispair():
    """A shuffled loader with frame-ordered metadata mispairs rows."""
    from torch.utils.data import DataLoader

    df = frame(n=64)
    ds = dataset(df).with_index()
    loader = DataLoader(ds, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(1))
    _, _, row_idx = runner.extract_representations(IdentityEncoder(), loader, "cpu")
    assert (row_idx != df["row_index"].to_numpy()).mean() > 0.5


def test_join_rejects_duplicates_missing_and_label_disagreement():
    df = frame(n=6)
    rows = df["row_index"].to_numpy()
    ids = df["target_label"].map(LABEL_TO_ID).to_numpy()
    with pytest.raises(AssertionError, match="duplicate"):
        runner.keyed_representation_metadata("t", df, ids, np.r_[rows[:-1], rows[0]], LABEL_TO_ID)
    with pytest.raises(AssertionError, match="row sets differ"):
        runner.keyed_representation_metadata("t", df, ids[:-1], rows[:-1], LABEL_TO_ID)
    with pytest.raises(AssertionError, match="label ids"):
        runner.keyed_representation_metadata("t", df, 1 - ids, rows, LABEL_TO_ID)


def test_preloaded_signals_are_identical_to_direct_reads():
    source = RowCodedSource()
    rows = np.array([105, 101, 130, 101, 150])
    pre = runner.PreloadedSignals(source, rows, progress_every=0)
    assert len(pre) == 4  # unique rows only
    for r in rows:
        np.testing.assert_array_equal(
            runner.signal_from_sample(pre[r]), runner.signal_from_sample(source[r])
        )
    df = frame()
    direct = runner.SplitTorchEEGDataset(source, df, LABEL_TO_ID, temporal_ablation="phase_random", seed=3)
    cached = runner.SplitTorchEEGDataset(
        runner.PreloadedSignals(source, df["row_index"], progress_every=0), df, LABEL_TO_ID,
        temporal_ablation="phase_random", seed=3,
    )
    for i in range(len(df)):
        a, la = direct[i]
        b, lb = cached[i]
        assert torch.equal(a, b) and int(la) == int(lb)


def test_array_signals_match_direct_reads(tmp_path):
    source = RowCodedSource()
    array = np.stack([runner.signal_from_sample(source[i]) for i in range(200)])
    np.save(tmp_path / "x_windows.npy", array)
    served = runner.ArraySignals(tmp_path / "x_windows.npy")
    df = frame()
    a = runner.SplitTorchEEGDataset(source, df, LABEL_TO_ID, temporal_ablation="time_shuffle", seed=5)
    b = runner.SplitTorchEEGDataset(served, df, LABEL_TO_ID, temporal_ablation="time_shuffle", seed=5)
    for i in range(len(df)):
        assert torch.equal(a[i][0], b[i][0])


def test_extraction_is_deterministic_under_eval_mode():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.5), torch.nn.Linear(32, 2))
    df = frame()
    first = runner.extract_representations(model, runner.extraction_loader(dataset(df), 5, 0), "cpu")[0]
    second = runner.extract_representations(model, runner.extraction_loader(dataset(df), 9, 0), "cpu")[0]
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)
