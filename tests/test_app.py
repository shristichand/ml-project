import os

import pytest

from app import app
from src.config import BACKGROUND_MODEL_PATH, MATH_MODEL_PATH


pytestmark = pytest.mark.skipif(
    not (os.path.exists(MATH_MODEL_PATH) and os.path.exists(BACKGROUND_MODEL_PATH)),
    reason="Trained artifacts are required",
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Student Academic Risk" in response.data


def test_math_form_get(client):
    assert client.get("/predictdata").status_code == 200


def test_api_math_validation(client):
    response = client.post("/api/predict/math", json={"gender": "nope"})
    assert response.status_code == 400


def test_api_math_and_risk(client):
    payload = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
        "reading_score": 72,
        "writing_score": 74,
    }
    math_response = client.post("/api/predict/math", json=payload)
    assert math_response.status_code == 200
    body = math_response.get_json()
    assert "predicted_math_score" in body
    assert body["risk"]["early_warning"] is False

    risk_response = client.post("/api/predict/risk", json=payload)
    assert risk_response.status_code == 200
    risk_body = risk_response.get_json()
    assert "predicted_average_score" in risk_body
