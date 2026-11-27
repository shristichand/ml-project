from flask import Flask, jsonify, render_template, request

from src.config import MODEL_VERSION
from src.pipeline.predict_pipeline import (
    BackgroundPredictPipeline,
    CustomData,
    InputValidationError,
    PredictPipeline,
    custom_data_from_mapping,
)

app = Flask(__name__)


def _form_custom_data(require_scores: bool) -> CustomData:
    return CustomData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=request.form.get("reading_score"),
        writing_score=request.form.get("writing_score"),
        require_scores=require_scores,
    )


@app.route("/")
def index():
    return render_template("index.html", model_version=MODEL_VERSION)


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    context = {"model_version": MODEL_VERSION, "mode": "math"}
    if request.method == "GET":
        return render_template("home.html", **context)
    try:
        data = _form_custom_data(require_scores=True)
        pred_df = data.get_data_as_data_frame()
        details = PredictPipeline().predict_math_details(pred_df)
        context.update(
            {
                "results": details["predicted_math_score"],
                "math_details": details,
            }
        )
        return render_template("home.html", **context)
    except InputValidationError as exc:
        context["error"] = str(exc)
        return render_template("home.html", **context), 400


@app.route("/predictrisk", methods=["GET", "POST"])
def predict_risk():
    context = {"model_version": MODEL_VERSION, "mode": "background"}
    if request.method == "GET":
        return render_template("home.html", **context)
    try:
        data = _form_custom_data(require_scores=False)
        details = BackgroundPredictPipeline().predict_details(data.get_background_frame())
        context["background_details"] = details
        return render_template("home.html", **context)
    except InputValidationError as exc:
        context["error"] = str(exc)
        return render_template("home.html", **context), 400


@app.route("/api/predict/math", methods=["POST"])
def api_predict_math():
    payload = request.get_json(silent=True) or {}
    try:
        data = custom_data_from_mapping(payload, require_scores=True)
        details = PredictPipeline().predict_math_details(data.get_data_as_data_frame())
        return jsonify(details)
    except InputValidationError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict/risk", methods=["POST"])
def api_predict_risk():
    payload = request.get_json(silent=True) or {}
    try:
        data = custom_data_from_mapping(payload, require_scores=False)
        details = BackgroundPredictPipeline().predict_details(data.get_background_frame())
        return jsonify(details)
    except InputValidationError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5001)
