import os

RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_DATA_PATH = os.path.join("notebook", "data", "stud.csv")
ARTIFACTS_DIR = "artifacts"
LOGS_DIR = "logs"
PREDICTION_LOG_PATH = os.path.join(LOGS_DIR, "predictions.jsonl")

TRAIN_DATA_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")
RAW_COPY_PATH = os.path.join(ARTIFACTS_DIR, "raw.csv")

CATEGORICAL_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]

SCORE_COLUMNS = ["math_score", "reading_score", "writing_score"]

MATH_TARGET = "math_score"
MATH_NUMERIC_COLUMNS = ["reading_score", "writing_score"]
MATH_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
MATH_PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
MATH_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics_math.json")

BACKGROUND_TARGET = "average_score"
BACKGROUND_NUMERIC_COLUMNS = []
BACKGROUND_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_background.pkl")
BACKGROUND_PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor_background.pkl")
BACKGROUND_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics_background.json")

TRAINING_SUMMARY_PATH = os.path.join(ARTIFACTS_DIR, "training_summary.json")

ALLOWED_CATEGORIES = {
    "gender": ["female", "male"],
    "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
    "parental_level_of_education": [
        "associate's degree",
        "bachelor's degree",
        "high school",
        "master's degree",
        "some college",
        "some high school",
    ],
    "lunch": ["free/reduced", "standard"],
    "test_preparation_course": ["none", "completed"],
}

# Documented operating points from the existing score distribution, not new labels in the CSV.
MATH_RISK_HIGH = 40
MATH_RISK_WATCH = 50
AVERAGE_RISK_HIGH = 50
AVERAGE_RISK_WATCH = 60

MODEL_VERSION = "0.2.0"
