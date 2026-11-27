import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import (
    DataTransformation,
    background_transformation_config,
    math_transformation_config,
)
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.config import (
    AVERAGE_RISK_HIGH,
    BACKGROUND_METRICS_PATH,
    BACKGROUND_MODEL_PATH,
    MATH_METRICS_PATH,
    MATH_MODEL_PATH,
    MODEL_VERSION,
    TRAINING_SUMMARY_PATH,
)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_json


class TrainPipeline:
    def run(self):
        try:
            logging.info("Starting training pipeline")
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            math_transform = DataTransformation(math_transformation_config())
            math_train, math_test, _, math_features, math_test_df = (
                math_transform.initiate_data_transformation(train_path, test_path)
            )
            math_trainer = ModelTrainer(
                ModelTrainerConfig(
                    trained_model_file_path=MATH_MODEL_PATH,
                    metrics_file_path=MATH_METRICS_PATH,
                    task_name="math_score_regression",
                )
            )
            math_r2, math_metrics = math_trainer.initiate_model_trainer(
                math_train,
                math_test,
                feature_names=math_features,
                test_frame=math_test_df,
            )

            background_transform = DataTransformation(background_transformation_config())
            bg_train, bg_test, _, bg_features, bg_test_df = (
                background_transform.initiate_data_transformation(train_path, test_path)
            )
            background_trainer = ModelTrainer(
                ModelTrainerConfig(
                    trained_model_file_path=BACKGROUND_MODEL_PATH,
                    metrics_file_path=BACKGROUND_METRICS_PATH,
                    task_name="background_average_regression",
                )
            )
            bg_r2, bg_metrics = background_trainer.initiate_model_trainer(
                bg_train,
                bg_test,
                feature_names=bg_features,
                test_frame=bg_test_df,
                risk_threshold=AVERAGE_RISK_HIGH,
            )

            summary = {
                "model_version": MODEL_VERSION,
                "math": {
                    "r2_test": math_r2,
                    "best_model": math_metrics["best_model"],
                    "note": (
                        "Predicts math_score from demographics plus reading and writing. "
                        "Reading/writing are same-exam scores, not early-warning features."
                    ),
                },
                "background": {
                    "r2_test": bg_r2,
                    "best_model": bg_metrics["best_model"],
                    "note": (
                        "Predicts average of the three exam scores from categorical fields only. "
                        "No exam scores are used as inputs (avoids target leakage)."
                    ),
                },
            }
            save_json(TRAINING_SUMMARY_PATH, summary)
            logging.info(f"Training complete: {summary}")
            return summary
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print(TrainPipeline().run())
