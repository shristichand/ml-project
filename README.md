# Student Academic Risk & Performance Monitoring System

End-to-end ML project around the Kaggle Students Performance in Exams dataset (1,000 rows, 8 columns). The original math-score predictor is preserved. Risk monitoring is added **on top of that model**, without inventing dataset features.

## What this system does

**Exam-score prediction (existing):** estimate `math_score` from demographics, lunch, test preparation, `reading_score`, and `writing_score`. This is **same-exam completion**. Reading and writing are not early-warning inputs; they come from the same sitting as math.

**Academic risk monitoring (added):**

1. Map the predicted math score to a band (`high` &lt; 40, `watch` 40–50, `low` otherwise). Still not early warning.
2. A **background-only** model predicts the average of the three exam scores using **only the five categorical columns**. No exam scores are inputs, so the average target is not leaked. Signal is weak by design. Associations can track gender, ethnicity, and socioeconomic proxies (`lunch`, parental education). Treat flags as exploratory, not decisions.

There are no student IDs, timestamps, or prior-term variables in this CSV, so the project **cannot** honestly do longitudinal early warning until a future dataset provides them.

## Dataset

Source: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

| Column | Role |
|---|---|
| `gender` | categorical feature (sensitive) |
| `race_ethnicity` | categorical feature (sensitive) |
| `parental_level_of_education` | categorical feature |
| `lunch` | categorical feature (SES proxy) |
| `test_preparation_course` | categorical feature |
| `math_score` | Model A target |
| `reading_score`, `writing_score` | Model A features only |
| `average_score` | **derived label** for Model B = mean of the three scores (not a new survey field) |

## Project layout

```
notebook/                 EDA and original experiments
notebook/data/stud.csv    source data
src/config.py             paths, allowed categories, risk thresholds
src/components/           ingestion, transformation, training
src/pipeline/train_pipeline.py
src/pipeline/predict_pipeline.py
src/pipeline/risk.py      bands and explanations
src/monitoring.py         test-set snapshot
app.py                    Flask UI + JSON API
tests/
```

Existing notebooks and the math form/CLI are kept.

## Train

```bash
python -m src.pipeline.train_pipeline
```

Writes `artifacts/model.pkl` and `preprocessor.pkl` (math), `model_background.pkl` and `preprocessor_background.pkl`, plus metrics JSON. The winner is chosen by **cross-validated R² on the training set**; the test split is used once for reporting.

`FAST_TRAIN=1` trains only Linear Regression (for tests).

## Serve

```bash
python app.py
```

- `/predictdata` — original math predictor (now with band + explanations)
- `/predictrisk` — background-only profile
- `POST /api/predict/math` and `POST /api/predict/risk`

CLI: `python src/make_prediction.py`

Predictions append to `logs/predictions.jsonl`.

## Ethics and leakage

- Do not use exam scores as features when the target is `average_score` or a label derived from those scores.
- Do not describe Model A as early identification.
- Gender, ethnicity, and lunch can drive Model B; report group MAE in `artifacts/metrics_*.json`.
- Thresholds are operating points from the existing score distribution, not extra data.

## Tests

```bash
FAST_TRAIN=1 python -m pytest tests/test_risk_and_validation.py -q
FAST_TRAIN=1 python -m pytest tests/test_train_smoke.py -q
```
