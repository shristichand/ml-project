from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    AVERAGE_RISK_HIGH,
    AVERAGE_RISK_WATCH,
    MATH_RISK_HIGH,
    MATH_RISK_WATCH,
)


def math_risk_band(predicted_math: float) -> Dict:
    """Map a predicted math score to a monitoring band. Same-exam, not early warning."""
    if predicted_math < MATH_RISK_HIGH:
        band = "high"
        label = "High math risk"
    elif predicted_math < MATH_RISK_WATCH:
        band = "watch"
        label = "Watch — borderline math"
    else:
        band = "low"
        label = "Low math risk"
    return {
        "band": band,
        "label": label,
        "threshold_high": MATH_RISK_HIGH,
        "threshold_watch": MATH_RISK_WATCH,
        "basis": "predicted_math_score",
        "early_warning": False,
    }


def average_risk_band(predicted_average: float) -> Dict:
    """Map a background-only predicted average to a risk band."""
    if predicted_average < AVERAGE_RISK_HIGH:
        band = "high"
        label = "High overall academic risk"
    elif predicted_average < AVERAGE_RISK_WATCH:
        band = "watch"
        label = "Watch — below typical average"
    else:
        band = "low"
        label = "Low overall academic risk"
    return {
        "band": band,
        "label": label,
        "threshold_high": AVERAGE_RISK_HIGH,
        "threshold_watch": AVERAGE_RISK_WATCH,
        "basis": "predicted_average_score",
        "early_warning": False,
        "note": (
            "Uses only demographic and support fields from this dataset. "
            "It is a weak pre-exam profile, not longitudinal early warning."
        ),
    }


def instance_contributions(
    model,
    transformed_row: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
) -> List[Dict]:
    """
    Local contributions when the model is linear: x_i * coef_i.
    Otherwise fall back to global feature_importances_ if present.
    """
    row = np.asarray(transformed_row).reshape(-1)
    explanations: List[Dict] = []

    coef = getattr(model, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef).reshape(-1)
        n = min(len(coef), len(row), len(feature_names))
        contrib = row[:n] * coef[:n]
        order = np.argsort(np.abs(contrib))[::-1][:top_k]
        for i in order:
            explanations.append(
                {
                    "feature": feature_names[i],
                    "value": float(row[i]),
                    "coefficient": float(coef[i]),
                    "contribution": float(contrib[i]),
                    "method": "linear_contribution",
                }
            )
        intercept = getattr(model, "intercept_", None)
        if intercept is not None:
            explanations.append(
                {
                    "feature": "intercept",
                    "value": 1.0,
                    "coefficient": float(np.asarray(intercept).reshape(-1)[0]),
                    "contribution": float(np.asarray(intercept).reshape(-1)[0]),
                    "method": "linear_contribution",
                }
            )
        return explanations

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        importances = np.asarray(importances).reshape(-1)
        n = min(len(importances), len(feature_names))
        order = np.argsort(importances[:n])[::-1][:top_k]
        for i in order:
            explanations.append(
                {
                    "feature": feature_names[i],
                    "importance": float(importances[i]),
                    "method": "global_feature_importance",
                }
            )
    return explanations


def add_average_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["average_score"] = (
        out["math_score"] + out["reading_score"] + out["writing_score"]
    ) / 3.0
    return out


def subgroup_regression_report(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_columns: Optional[List[str]] = None,
) -> Dict:
    group_columns = group_columns or ["gender", "lunch", "race_ethnicity"]
    report = {}
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    for col in group_columns:
        if col not in frame.columns:
            continue
        by_group = {}
        for group_value, idx in frame.groupby(col).groups.items():
            index = list(idx)
            by_group[str(group_value)] = {
                "n": int(len(index)),
                "mae": float(np.mean(abs_err[index])),
                "mean_true": float(np.mean(np.asarray(y_true)[index])),
                "mean_pred": float(np.mean(np.asarray(y_pred)[index])),
            }
        report[col] = by_group
    return report
