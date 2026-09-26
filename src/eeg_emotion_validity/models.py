"""Braindecode model factory for canonical EEG deep architectures."""

from __future__ import annotations


MODEL_ALIASES = {
    "eegnet": ("EEGNet", "EEGNetv4"),
    "eegnetv4": ("EEGNetv4", "EEGNet"),
    "shallow_convnet": ("ShallowFBCSPNet",),
    "shallow_fbcspnet": ("ShallowFBCSPNet",),
    "deep_convnet": ("Deep4Net",),
    "deep4net": ("Deep4Net",),
    "eegconformer": ("EEGConformer",),
    "eeg_conformer": ("EEGConformer",),
    "tsception": ("TSception", "TSceptionV1"),
}


def build_braindecode_model(
    name: str,
    *,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: float | None = None,
):
    """Instantiate a Braindecode model by project alias.

    The project deliberately delegates these architectures to Braindecode so
    architecture details stay aligned with the maintained EEG model zoo.
    """

    try:
        import braindecode.models as models
    except ImportError as exc:
        raise ImportError(
            "Braindecode is required for canonical deep EEG models. "
            "Install the optional deep-learning dependencies with `pip install -e .[deep]`."
        ) from exc

    aliases = MODEL_ALIASES.get(name.lower())
    if aliases is None:
        raise ValueError(f"Unsupported Braindecode model: {name}")

    model_cls = None
    for alias in aliases:
        model_cls = getattr(models, alias, None)
        if model_cls is not None:
            break
    if model_cls is None:
        available = ", ".join(aliases)
        raise ValueError(f"None of the expected Braindecode classes are available: {available}")

    base_kwargs = {"n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times}
    if name.lower() in {"tsception", "eegconformer", "eeg_conformer"} and sfreq is not None:
        base_kwargs["sfreq"] = sfreq
        base_kwargs["input_window_seconds"] = n_times / sfreq

    attempts = [
        base_kwargs,
        {"in_chans": n_chans, "n_classes": n_outputs, "input_window_samples": n_times},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return model_cls(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise TypeError(f"Could not instantiate {model_cls.__name__}: {last_error}")
