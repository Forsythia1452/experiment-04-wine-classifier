from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "wine_pipeline.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
RESULT_PATH = MODEL_DIR / "evaluation.json"
RANDOM_STATE = 42


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_stats(frame: pd.DataFrame) -> dict:
    stats = {}
    for name in frame.columns:
        values = frame[name]
        stats[name] = {
            "min": float(values.min()), "max": float(values.max()),
            "q1": float(values.quantile(0.25)), "q3": float(values.quantile(0.75)),
            "median": float(values.median()), "mean": float(values.mean()),
        }
    return stats


def build_candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }


def train_and_save() -> dict:
    dataset = load_wine(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy", "precision_macro": "precision_macro",
        "recall_macro": "recall_macro", "f1_macro": "f1_macro",
    }
    comparisons = []
    candidates = build_candidates()
    for name, pipeline in candidates.items():
        start = time.perf_counter()
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        comparisons.append({
            "model": name,
            **{f"cv_{metric}_mean": float(np.mean(scores[f"test_{metric}"])) for metric in scoring},
            **{f"cv_{metric}_std": float(np.std(scores[f"test_{metric}"])) for metric in scoring},
            "cv_fit_time_mean_seconds": float(np.mean(scores["fit_time"])),
            "comparison_wall_seconds": float(time.perf_counter() - start),
        })
    best_name = max(comparisons, key=lambda row: row["cv_f1_macro_mean"])["model"]
    best = candidates[best_name]
    best.fit(X_train, y_train)

    # The held-out test set is evaluated once, after model selection.
    test_prediction = best.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_prediction, average="macro", zero_division=0,
    )
    final_metrics = {
        "accuracy": float(accuracy_score(y_test, test_prediction)),
        "precision_macro": float(precision), "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_test, test_prediction).tolist(),
        "test_samples": int(len(y_test)), "evaluated_once": True,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH, compress=3)
    model_hash = file_sha256(MODEL_PATH)
    meta = {
        "model_version": "1.0.0", "model_name": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": "sklearn-load_wine-1.7.2", "data_samples": int(len(X)),
        "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
        "random_state": RANDOM_STATE, "split_strategy": "stratified 80/20",
        "selection_protocol": "5-fold stratified cross-validation on training set",
        "primary_metric": "macro F1", "feature_names": list(X.columns),
        "class_names": [f"cultivar_{i}" for i in range(len(dataset.target_names))],
        "training_stats": training_stats(X_train),
        "sklearn_version": sklearn.__version__, "git_commit": git_commit(),
        "model_file": MODEL_PATH.name, "model_sha256": model_hash,
        "final_metrics": final_metrics,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"comparison": comparisons, "selected_model": best_name, "final_test": final_metrics}
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(train_and_save(), ensure_ascii=False, indent=2))

