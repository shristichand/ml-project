import os
import sys
from dataclasses import dataclass
from typing import Optional

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import MATH_MODEL_PATH
from src.exception import CustomException
from src.logger import logging
from src.pipeline.risk import subgroup_regression_report
from src.utils import (
    evaluate_model,
    regression_metrics,
    save_json,
    save_object,
    thresholded_risk_metrics,
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = MATH_MODEL_PATH
    metrics_file_path: Optional[str] = None
    task_name: str = "math_score_regression"


class ModelTrainer:
    def __init__(self, config: Optional[ModelTrainerConfig] = None):
        self.model_trainer_config = config or ModelTrainerConfig()

    def _candidate_models(self):
        if os.environ.get("FAST_TRAIN") == "1":
            return {"Linear Regression": LinearRegression()}
        return {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }

    def _param_grid(self):
        return {
            "Decision Tree": {
                "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            },
            "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Linear Regression": {},
            "K-Neighbors Regressor": {},
            "XGBRegressor": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "CatBoosting Regressor": {
                "depth": [6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [30, 50, 100],
            },
            "AdaBoost Regressor": {
                "learning_rate": [0.1, 0.01, 0.5, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        feature_names=None,
        test_frame=None,
        risk_threshold: Optional[float] = None,
    ):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = self._candidate_models()
            params = self._param_grid()

            model_report, fitted_models = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            best_model_name = max(
                model_report.keys(), key=lambda name: model_report[name]["cv_r2"]
            )
            best_model_cv = model_report[best_model_name]["cv_r2"]
            best_model = fitted_models[best_model_name]

            logging.info(
                f"Best model found, Model Name: {best_model_name}, CV R2 Score: {best_model_cv}"
            )

            if (
                self.model_trainer_config.task_name == "math_score_regression"
                and best_model_cv < 0.6
            ):
                raise CustomException("No best model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            test_metrics = regression_metrics(y_test, predicted)

            payload = {
                "task": self.model_trainer_config.task_name,
                "best_model": best_model_name,
                "selection_criterion": "cv_r2",
                "cv_r2": best_model_cv,
                "best_params": model_report[best_model_name]["best_params"],
                "feature_names": feature_names or [],
                "train": model_report[best_model_name]["train"],
                "test": test_metrics,
                "all_models": model_report,
            }

            if risk_threshold is not None:
                payload["thresholded_risk_on_test"] = thresholded_risk_metrics(
                    y_test, predicted, risk_threshold
                )

            if test_frame is not None:
                payload["fairness_mae_by_group"] = subgroup_regression_report(
                    test_frame, y_test, predicted
                )

            metrics_path = self.model_trainer_config.metrics_file_path
            if metrics_path:
                save_json(metrics_path, payload)

            logging.info(f"Test R2 for selected model: {test_metrics['r2']}")
            return test_metrics["r2"], payload

        except Exception as e:
            raise CustomException(e, sys)
