from __future__ import annotations

import os
import secrets
from typing import Any, Dict, Optional

from flask import Flask, flash, redirect, render_template, request, session, url_for

from model_pipeline import evaluate_saved_model, predict_sms, train_and_save_model
from mongodb_service import MongoDBService


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
database = MongoDBService()
cached_metrics: Optional[Dict[str, Any]] = None


def get_model_metrics() -> Optional[Dict[str, Any]]:
    global cached_metrics
    if cached_metrics is None:
        try:
            cached_metrics = evaluate_saved_model()
        except (FileNotFoundError, OSError, ValueError) as exc:
            app.logger.info("Model evaluation is unavailable: %s", exc)
    return cached_metrics


def serialize_records(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [
        {key: str(value) if key in {"_id", "created_at"} else value for key, value in record.items()}
        for record in records
    ]


@app.get("/")
def dashboard():
    metrics = get_model_metrics()
    analytics: Optional[Dict[str, Any]] = None
    if database.connected:
        analytics = {
            "message_distribution": database.label_distribution(),
            "message_percentages": database.label_percentages(),
            "prediction_distribution": database.prediction_distribution(),
            "total_predictions": database.total_predictions(),
            "average_spam_probability": database.average_spam_probability(),
            "daily_stats": database.daily_prediction_stats(),
            "feedback_stats": database.feedback_statistics(),
            "recent_messages": serialize_records(database.get_all_messages()[:8]),
            "recent_predictions": serialize_records(database.get_prediction_history()[:8]),
        }

    return render_template(
        "index.html",
        metrics=metrics,
        analytics=analytics,
        mongo_connected=database.connected,
        mongo_error=database.connection_error,
        prediction=session.get("last_prediction"),
        prediction_id=session.get("last_prediction_id"),
        message=session.get("message", ""),
    )


@app.post("/predict")
def predict():
    message = request.form.get("message", "").strip()
    session["message"] = message
    try:
        prediction = predict_sms(message)
        session["last_prediction"] = prediction
        session.pop("last_prediction_id", None)
        if database.connected and prediction["has_probabilities"]:
            session["last_prediction_id"] = database.insert_prediction(
                message=message,
                predicted_label=prediction["predicted_label"],
                spam_probability=prediction["spam_probability"],
                ham_probability=prediction["ham_probability"],
                model_name=prediction["model_name"],
            )
            flash("Prediction complete and saved to MongoDB.", "success")
        elif database.connected:
            flash("Prediction complete. This model does not provide probabilities, so it was not stored.", "error")
        else:
            flash("Prediction complete, but MongoDB is unavailable so it was not saved.", "error")
    except Exception as exc:
        flash(f"Prediction failed: {exc}", "error")
    return redirect(url_for("dashboard") + "#classifier")


@app.post("/feedback")
def feedback():
    prediction = session.get("last_prediction")
    prediction_id = session.get("last_prediction_id")
    feedback_text = request.form.get("feedback", "").strip()
    correct_label = request.form.get("correct_label", "").lower()
    if not prediction or not prediction_id:
        flash("Run a prediction while MongoDB is connected before submitting feedback.", "error")
    else:
        try:
            database.insert_feedback(
                prediction_id=prediction_id,
                message=session.get("message", ""),
                predicted_label=prediction["predicted_label"],
                correct_label=correct_label,
                feedback=feedback_text,
            )
            flash("Thanks — your feedback has been saved.", "success")
        except Exception as exc:
            flash(f"Feedback could not be saved: {exc}", "error")
    return redirect(url_for("dashboard") + "#classifier")


@app.post("/initialize")
def initialize_database():
    if not database.connected:
        flash("MongoDB is unavailable. Start the MongoDB service and try again.", "error")
    else:
        try:
            database.ensure_database()
            flash("MongoDB collections and validators are ready.", "success")
        except Exception as exc:
            flash(f"Database initialization failed: {exc}", "error")
    return redirect(url_for("dashboard") + "#database")


@app.post("/train")
def train_model():
    global cached_metrics
    try:
        cached_metrics, _, _ = train_and_save_model()
        flash("Model trained and saved successfully.", "success")
    except Exception as exc:
        flash(f"Training failed: {exc}", "error")
    return redirect(url_for("dashboard") + "#model")


@app.template_filter("percent")
def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
