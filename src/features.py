from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """Raised when inference data does not match the model schema."""


@dataclass(frozen=True)
class DriftResult:
    level: str
    count: int
    details: list[str]


def validate_frame(frame: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    expected = list(meta["feature_names"])
    missing = [name for name in expected if name not in frame.columns]
    extra = [name for name in frame.columns if name not in expected]
    if missing:
        raise SchemaError(f"缺少特征：{', '.join(missing)}")
    if extra:
        raise SchemaError(f"存在未知特征：{', '.join(extra)}")
    if frame.empty:
        raise SchemaError("至少需要一行数据")

    clean = frame.loc[:, expected].copy()
    for name in expected:
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
        if clean[name].isna().any():
            raise SchemaError(f"特征 {name} 含空值或非数字")
        if not np.isfinite(clean[name].to_numpy()).all():
            raise SchemaError(f"特征 {name} 必须是有限数值")
        stat = meta["training_stats"][name]
        span = max(float(stat["max"]) - float(stat["min"]), 1e-9)
        hard_low = float(stat["min"]) - span
        hard_high = float(stat["max"]) + span
        if ((clean[name] < hard_low) | (clean[name] > hard_high)).any():
            raise SchemaError(f"特征 {name} 超出可接受范围 {hard_low:.3g} 至 {hard_high:.3g}")
    return clean


def assess_distribution(row: pd.Series, meta: dict[str, Any]) -> DriftResult:
    details: list[str] = []
    severe = 0
    for name in meta["feature_names"]:
        value = float(row[name])
        stat = meta["training_stats"][name]
        train_min, train_max = float(stat["min"]), float(stat["max"])
        q1, q3 = float(stat["q1"]), float(stat["q3"])
        iqr = max(q3 - q1, 1e-9)
        if value < train_min or value > train_max:
            severe += 1
            details.append(f"{name}={value:g} 超出训练范围 [{train_min:g}, {train_max:g}]")
        elif value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr:
            details.append(f"{name}={value:g} 位于训练分布尾部")
    count = len(details)
    level = "high" if severe >= 2 or count >= 4 else ("medium" if count else "normal")
    return DriftResult(level=level, count=count, details=details)


def predict_rows(model: Any, frame: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    clean = validate_frame(frame, meta)
    prediction = model.predict(clean)
    probabilities = model.predict_proba(clean)
    labels = meta["class_names"]
    rows = []
    for index, (pred, probs) in enumerate(zip(prediction, probabilities)):
        drift = assess_distribution(clean.iloc[index], meta)
        rows.append({
            "prediction": labels[int(pred)],
            "confidence": float(np.max(probs)),
            "distribution_level": drift.level,
            "distribution_warnings": "；".join(drift.details),
        })
    return pd.DataFrame(rows, index=frame.index)

