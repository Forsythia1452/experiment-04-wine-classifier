import json
from pathlib import Path

import joblib
import pytest
from sklearn.datasets import load_wine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dataset():
    return load_wine(as_frame=True)


@pytest.fixture(scope="session")
def meta():
    return json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def evaluation():
    return json.loads((ROOT / "models" / "evaluation.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def model():
    return joblib.load(ROOT / "models" / "wine_pipeline.joblib")

