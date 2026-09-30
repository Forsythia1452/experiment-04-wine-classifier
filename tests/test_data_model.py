import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.train import RANDOM_STATE, build_candidates

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_shape(dataset):
    assert dataset.data.shape == (178, 13)


def test_dataset_has_three_labels(dataset):
    assert sorted(dataset.target.unique().tolist()) == [0, 1, 2]


def test_dataset_has_no_missing_values(dataset):
    assert not dataset.data.isna().any().any()


def test_split_is_reproducible(dataset):
    args = dict(test_size=0.20, random_state=RANDOM_STATE, stratify=dataset.target)
    first = train_test_split(dataset.data, dataset.target, **args)
    second = train_test_split(dataset.data, dataset.target, **args)
    assert first[0].index.tolist() == second[0].index.tolist()
    assert first[1].index.tolist() == second[1].index.tolist()


def test_train_and_test_indices_do_not_overlap(dataset):
    train, test, _, _ = train_test_split(dataset.data, dataset.target, test_size=.20, random_state=RANDOM_STATE, stratify=dataset.target)
    assert set(train.index).isdisjoint(test.index)


def test_split_preserves_all_classes(dataset):
    _, test, _, y_test = train_test_split(dataset.data, dataset.target, test_size=.20, random_state=RANDOM_STATE, stratify=dataset.target)
    assert len(test) == 36
    assert y_test.value_counts().to_dict() == {1: 14, 0: 12, 2: 10}


def test_two_models_use_pipeline():
    candidates = build_candidates()
    assert set(candidates) == {"logistic_regression", "random_forest"}
    assert all(hasattr(model, "steps") for model in candidates.values())


def test_model_comparison_has_same_protocol(evaluation):
    rows = evaluation["comparison"]
    assert len(rows) == 2
    assert all("cv_f1_macro_mean" in row and "cv_f1_macro_std" in row for row in rows)
    assert evaluation["selected_model"] == max(rows, key=lambda row: row["cv_f1_macro_mean"])["model"]


def test_final_metrics_are_complete(evaluation):
    result = evaluation["final_test"]
    assert result["evaluated_once"] is True
    assert set(["accuracy", "precision_macro", "recall_macro", "f1_macro"]).issubset(result)
    assert np.asarray(result["confusion_matrix"]).shape == (3, 3)
    assert sum(sum(row) for row in result["confusion_matrix"]) == 36


def test_model_sha256_matches_metadata(meta):
    path = ROOT / "models" / "wine_pipeline.joblib"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["model_sha256"]

