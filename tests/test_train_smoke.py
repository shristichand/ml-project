import os

import pytest

from src.config import BACKGROUND_MODEL_PATH, MATH_MODEL_PATH
from src.pipeline.predict_pipeline import CustomData, PredictPipeline, BackgroundPredictPipeline
from src.pipeline.train_pipeline import TrainPipeline


@pytest.mark.slow
def test_fast_train_and_predict(tmp_path, monkeypatch):
    monkeypatch.setenv("FAST_TRAIN", "1")
    summary = TrainPipeline().run()
    assert "math" in summary and "background" in summary
    assert os.path.exists(MATH_MODEL_PATH)
    assert os.path.exists(BACKGROUND_MODEL_PATH)

    data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        reading_score=72,
        writing_score=74,
    )
    math_details = PredictPipeline().predict_math_details(
        data.get_data_as_data_frame(), log=False
    )
    assert 0 <= math_details["predicted_math_score"] <= 120
    bg = BackgroundPredictPipeline().predict_details(
        data.get_background_frame(), log=False
    )
    assert bg["predicted_average_score"] > 0
