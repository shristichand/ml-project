import sys
from typing import Optional

import pandas as pd

from src.config import (
    ALLOWED_CATEGORIES,
    BACKGROUND_MODEL_PATH,
    BACKGROUND_PREPROCESSOR_PATH,
    CATEGORICAL_COLUMNS,
    MATH_MODEL_PATH,
    MATH_PREPROCESSOR_PATH,
    MODEL_VERSION,
)
from src.exception import CustomException
from src.pipeline.risk import average_risk_band, instance_contributions, math_risk_band
from src.utils import load_object, log_prediction


class InputValidationError(ValueError):
    pass


def validate_categories(payload: dict):
    for column, allowed in ALLOWED_CATEGORIES.items():
        value = payload.get(column)
        if value not in allowed:
            raise InputValidationError(
                f"Invalid {column}={value!r}. Allowed: {allowed}"
            )


def validate_score(name: str, value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} must be a number") from exc
    if score < 0 or score > 100:
        raise InputValidationError(f"{name} must be between 0 and 100")
    return score


class PredictPipeline:
    def __init__(self):
        self.model = load_object(file_path=MATH_MODEL_PATH)
        self.preprocessor = load_object(file_path=MATH_PREPROCESSOR_PATH)

    def predict(self, features):
        """Preserve the original array-style math prediction API."""
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

    def predict_math_details(self, features: pd.DataFrame, log: bool = True) -> dict:
        try:
            transformed = self.preprocessor.transform(features)
            preds = self.model.predict(transformed)
            predicted = float(preds[0])
            feature_names = list(self.preprocessor.get_feature_names_out())
            result = {
                "task": "math_score_regression",
                "model_version": MODEL_VERSION,
                "predicted_math_score": predicted,
                "risk": math_risk_band(predicted),
                "explanations": instance_contributions(
                    self.model, transformed[0], feature_names
                ),
                "caveat": (
                    "This uses reading and writing scores from the same exam sitting. "
                    "It estimates a missing math mark, it does not identify students early."
                ),
            }
            if log:
                log_prediction(
                    {
                        "endpoint": "math",
                        "model_version": MODEL_VERSION,
                        "prediction": predicted,
                        "risk_band": result["risk"]["band"],
                    }
                )
            return result
        except Exception as e:
            raise CustomException(e, sys)


class BackgroundPredictPipeline:
    def __init__(self):
        self.model = load_object(file_path=BACKGROUND_MODEL_PATH)
        self.preprocessor = load_object(file_path=BACKGROUND_PREPROCESSOR_PATH)

    def predict_details(self, features: pd.DataFrame, log: bool = True) -> dict:
        try:
            transformed = self.preprocessor.transform(features)
            preds = self.model.predict(transformed)
            predicted = float(preds[0])
            feature_names = list(self.preprocessor.get_feature_names_out())
            result = {
                "task": "background_average_regression",
                "model_version": MODEL_VERSION,
                "predicted_average_score": predicted,
                "risk": average_risk_band(predicted),
                "explanations": instance_contributions(
                    self.model, transformed[0], feature_names
                ),
                "caveat": (
                    "This profile uses only gender, ethnicity, parental education, lunch, "
                    "and test preparation. Signal is limited and associations can reflect "
                    "socioeconomic patterns. It is not causal and is not longitudinal early warning."
                ),
            }
            if log:
                log_prediction(
                    {
                        "endpoint": "background_risk",
                        "model_version": MODEL_VERSION,
                        "prediction": predicted,
                        "risk_band": result["risk"]["band"],
                    }
                )
            return result
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: Optional[float] = None,
        writing_score: Optional[float] = None,
        require_scores: bool = True,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.require_scores = require_scores
        self.validate()

    def validate(self):
        payload = {
            "gender": self.gender,
            "race_ethnicity": self.race_ethnicity,
            "parental_level_of_education": self.parental_level_of_education,
            "lunch": self.lunch,
            "test_preparation_course": self.test_preparation_course,
        }
        validate_categories(payload)
        if self.require_scores:
            self.reading_score = validate_score("reading_score", self.reading_score)
            self.writing_score = validate_score("writing_score", self.writing_score)

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
            }
            if self.reading_score is not None:
                custom_data_input_dict["reading_score"] = [self.reading_score]
            if self.writing_score is not None:
                custom_data_input_dict["writing_score"] = [self.writing_score]
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

    def get_background_frame(self):
        return self.get_data_as_data_frame()[CATEGORICAL_COLUMNS]


def custom_data_from_mapping(payload: dict, require_scores: bool = True) -> CustomData:
    return CustomData(
        gender=payload.get("gender"),
        race_ethnicity=payload.get("race_ethnicity") or payload.get("ethnicity"),
        parental_level_of_education=payload.get("parental_level_of_education"),
        lunch=payload.get("lunch"),
        test_preparation_course=payload.get("test_preparation_course"),
        reading_score=payload.get("reading_score"),
        writing_score=payload.get("writing_score"),
        require_scores=require_scores,
    )
