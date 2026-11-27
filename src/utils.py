import json
import os
import sys
from datetime import datetime, timezone

import dill
import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV

from src.config import PREDICTION_LOG_PATH
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path, payload):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def regression_metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def thresholded_risk_metrics(y_true, y_pred, threshold: float) -> dict:
    y_true_flag = np.asarray(y_true) < threshold
    y_pred_flag = np.asarray(y_pred) < threshold
    if y_true_flag.sum() == 0:
        return {
            "threshold": threshold,
            "support": 0,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    return {
        "threshold": threshold,
        "support": int(y_true_flag.sum()),
        "precision": float(precision_score(y_true_flag, y_pred_flag, zero_division=0)),
        "recall": float(recall_score(y_true_flag, y_pred_flag, zero_division=0)),
        "f1": float(f1_score(y_true_flag, y_pred_flag, zero_division=0)),
    }


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    """
    Tune each model with CV on the training set only.
    Select later using cv_r2 — do not use the test set to pick a winner.
    """
    try:
        report = {}
        fitted_models = {}

        for name, model in models.items():
            para = param.get(name, {})
            gs = GridSearchCV(model, para, cv=3, scoring="r2", n_jobs=1)
            gs.fit(X_train, y_train)

            fitted = gs.best_estimator_
            y_train_pred = fitted.predict(X_train)
            y_test_pred = fitted.predict(X_test)

            train_scores = regression_metrics(y_train, y_train_pred)
            test_scores = regression_metrics(y_test, y_test_pred)

            report[name] = {
                "cv_r2": float(gs.best_score_),
                "best_params": gs.best_params_,
                "train": train_scores,
                "test": test_scores,
            }
            fitted_models[name] = fitted

        return report, fitted_models

    except Exception as e:
        raise CustomException(e, sys)


def log_prediction(record: dict):
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
    payload = dict(record)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(PREDICTION_LOG_PATH, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload) + "\n")
