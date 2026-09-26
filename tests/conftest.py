import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rating_manifest():
    """DEAP-like manifest where valence and arousal labels deliberately differ."""
    rows = []
    for subject in range(1, 5):
        for trial in range(1, 7):
            valence = 7.0 if trial % 2 else 3.0
            arousal = 7.0 if trial <= 3 else 2.0
            for clip in range(3):
                rows.append(
                    {
                        "subject_id": subject,
                        "trial_id": trial,
                        "clip_id": f"s{subject}_t{trial}_c{clip}",
                        "_record_id": f"rec{subject}",
                        "valence": valence,
                        "arousal": arousal,
                        "rating_source": "deap_provider_participant_ratings",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
