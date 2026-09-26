"""Each signal transform preserves exactly what its name claims, and no more."""

import numpy as np

from conftest import load_script

runner = load_script("run_deep")


def eeg_like(rng, channels=4, n=256):
    t = np.arange(n) / 128.0
    base = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 3 * t + 1.0)
    mix = rng.normal(size=(channels, 1))
    return (mix * base + 0.3 * rng.normal(size=(channels, n))).astype(np.float64)


def magnitude(x):
    return np.abs(np.fft.rfft(x, axis=-1))


def test_time_reverse_preserves_magnitude_and_amplitudes(rng):
    x = eeg_like(rng)
    y = runner.apply_temporal_ablation(x, "time_reverse", 0, 0)
    np.testing.assert_allclose(magnitude(y), magnitude(x), atol=1e-9)
    np.testing.assert_allclose(np.sort(y, axis=-1), np.sort(x, axis=-1))


def test_phase_random_preserves_magnitude_and_cross_channel_phase(rng):
    x = eeg_like(rng)
    y = runner.apply_temporal_ablation(x, "phase_random", 3, 11)
    np.testing.assert_allclose(magnitude(y), magnitude(x), rtol=1e-8, atol=1e-8)
    fx, fy = np.fft.rfft(x, axis=-1), np.fft.rfft(y, axis=-1)
    cross_x = fx[0] * np.conj(fx[1])
    cross_y = fy[0] * np.conj(fy[1])
    np.testing.assert_allclose(cross_y, cross_x, rtol=1e-7, atol=1e-7)
    assert not np.allclose(y, x)
    assert np.isrealobj(y)


def test_phase_random_is_fixed_per_sample_and_varies_across_samples(rng):
    x = eeg_like(rng)
    a = runner.apply_temporal_ablation(x, "phase_random", 3, 11)
    b = runner.apply_temporal_ablation(x, "phase_random", 3, 11)
    c = runner.apply_temporal_ablation(x, "phase_random", 3, 12)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_time_shuffle_keeps_amplitudes_but_not_spectrum(rng):
    x = eeg_like(rng)
    y = runner.apply_temporal_ablation(x, "time_shuffle", 0, 5)
    np.testing.assert_allclose(np.sort(y, axis=-1), np.sort(x, axis=-1))
    rel = np.abs(magnitude(y) - magnitude(x)).sum() / magnitude(x).sum()
    assert rel > 0.3  # shuffling redistributes power; it is not spectrum-preserving


def test_odd_length_phase_random_is_real_and_magnitude_preserving(rng):
    x = eeg_like(rng, n=255)
    y = runner.apply_temporal_ablation(x, "phase_random", 1, 1)
    assert y.shape == x.shape
    np.testing.assert_allclose(magnitude(y), magnitude(x), rtol=1e-8, atol=1e-8)
