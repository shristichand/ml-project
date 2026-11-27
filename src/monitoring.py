"""Score the locked test split and write a monitoring snapshot."""

import os

import pandas as pd

from src.components.data_transformation import (
    background_transformation_config,
    math_transformation_config,
)
from src.config import ARTIFACTS_DIR, AVERAGE_RISK_HIGH, TEST_DATA_PATH
from src.pipeline.predict_pipeline import BackgroundPredictPipeline, PredictPipeline
from src.pipeline.risk import add_average_score, subgroup_regression_report
from src.utils import regression_metrics, save_json, thresholded_risk_metrics


def run_monitoring(output_path: str = None):
    output_path = output_path or os.path.join(ARTIFACTS_DIR, "monitoring_snapshot.json")
    test_df = pd.read_csv(TEST_DATA_PATH)

    math_cfg = math_transformation_config()
    math_inputs = test_df[math_cfg.numerical_columns + math_cfg.categorical_columns]
    math_model = PredictPipeline()
    math_pred = math_model.predict(math_inputs)
    math_true = test_df["math_score"].to_numpy()

    bg_df = add_average_score(test_df)
    bg_inputs = bg_df[background_transformation_config().categorical_columns]
    bg_pipeline = BackgroundPredictPipeline()
    bg_transformed = bg_pipeline.preprocessor.transform(bg_inputs)
    bg_scores = bg_pipeline.model.predict(bg_transformed)
    bg_true = bg_df["average_score"].to_numpy()

    snapshot = {
        "n_test": int(len(test_df)),
        "math": {
            **regression_metrics(math_true, math_pred),
            "feature_means": {
                "reading_score": float(test_df["reading_score"].mean()),
                "writing_score": float(test_df["writing_score"].mean()),
            },
            "fairness_mae_by_group": subgroup_regression_report(
                test_df, math_true, math_pred
            ),
        },
        "background": {
            **regression_metrics(bg_true, bg_scores),
            "thresholded_risk": thresholded_risk_metrics(
                bg_true, bg_scores, AVERAGE_RISK_HIGH
            ),
            "fairness_mae_by_group": subgroup_regression_report(
                bg_df, bg_true, bg_scores
            ),
        },
    }
    save_json(output_path, snapshot)
    return snapshot


if __name__ == "__main__":
    print(run_monitoring())
