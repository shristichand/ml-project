import numpy as np
import pandas as pd
import pytest

from src.components.data_transformation import (
    DataTransformation,
    background_transformation_config,
)
from src.config import CATEGORICAL_COLUMNS, SCORE_COLUMNS
from src.pipeline.predict_pipeline import CustomData, InputValidationError
from src.pipeline.risk import add_average_score, average_risk_band, math_risk_band


def test_math_risk_bands():
    assert math_risk_band(20)["band"] == "high"
    assert math_risk_band(45)["band"] == "watch"
    assert math_risk_band(70)["band"] == "low"
    assert math_risk_band(70)["early_warning"] is False


def test_average_risk_bands():
    assert average_risk_band(40)["band"] == "high"
    assert average_risk_band(55)["band"] == "watch"
    assert average_risk_band(80)["band"] == "low"


def test_custom_data_rejects_bad_category():
    with pytest.raises(InputValidationError):
        CustomData(
            gender="unknown",
            race_ethnicity="group B",
            parental_level_of_education="high school",
            lunch="standard",
            test_preparation_course="none",
            reading_score=70,
            writing_score=70,
        )


def test_custom_data_rejects_out_of_range_score():
    with pytest.raises(InputValidationError):
        CustomData(
            gender="female",
            race_ethnicity="group B",
            parental_level_of_education="high school",
            lunch="standard",
            test_preparation_course="none",
            reading_score=120,
            writing_score=70,
        )


def test_background_frame_excludes_exam_scores():
    data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="high school",
        lunch="standard",
        test_preparation_course="none",
        reading_score=70,
        writing_score=71,
    )
    frame = data.get_background_frame()
    assert list(frame.columns) == CATEGORICAL_COLUMNS
    for col in SCORE_COLUMNS:
        assert col not in frame.columns


def test_background_transformer_does_not_use_scores():
    train = pd.DataFrame(
        {
            "gender": ["female", "male"],
            "race_ethnicity": ["group B", "group C"],
            "parental_level_of_education": ["high school", "some college"],
            "lunch": ["standard", "free/reduced"],
            "test_preparation_course": ["none", "completed"],
            "math_score": [70, 40],
            "reading_score": [80, 50],
            "writing_score": [75, 45],
        }
    )
    test = train.copy()
    transform = DataTransformation(background_transformation_config())
    prepared = transform._prepare_frame(train)
    for col in SCORE_COLUMNS:
        assert col not in prepared.columns
    assert "average_score" in prepared.columns
    np.testing.assert_allclose(prepared["average_score"].iloc[0], (70 + 80 + 75) / 3)


def test_add_average_is_derived_only():
    df = pd.DataFrame(
        {"math_score": [10], "reading_score": [20], "writing_score": [30]}
    )
    out = add_average_score(df)
    assert out.loc[0, "average_score"] == 20
