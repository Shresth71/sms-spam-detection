from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

DATASET_PATH = Path(__file__).resolve().parent / "spam.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "sms_spam_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
POSITIVE_LABEL = "spam"
CLASS_LABELS = ["ham", "spam"]
MODEL_DEFINITIONS = {
    "Multinomial Naive Bayes": (
        MultinomialNB(),
        [{"classifier__alpha": [0.1, 0.5, 1.0]}],
    ),
    "Complement Naive Bayes": (
        ComplementNB(),
        [{"classifier__alpha": [0.1, 0.5, 1.0]}],
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1500, solver="liblinear", random_state=RANDOM_STATE),
        [{"classifier__C": [0.5, 1.0, 2.0], "classifier__class_weight": [None, "balanced"]}],
    ),
    "Linear SVM": (
        LinearSVC(random_state=RANDOM_STATE, dual="auto"),
        [{"classifier__C": [0.5, 1.0, 2.0], "classifier__class_weight": [None, "balanced"]}],
    ),
}
VECTORIZER_CONFIGS = [
    {
        "tfidf__ngram_range": [(1, 1)],
        "tfidf__max_features": [5000],
        "tfidf__min_df": [1],
        "tfidf__max_df": [1.0],
    },
    {
        "tfidf__ngram_range": [(1, 2)],
        "tfidf__max_features": [10000],
        "tfidf__min_df": [1],
        "tfidf__max_df": [0.95],
    },
    {
        "tfidf__ngram_range": [(1, 2)],
        "tfidf__max_features": [20000],
        "tfidf__min_df": [2],
        "tfidf__max_df": [0.95],
    },
]


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_dataset(dataset_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, encoding="latin-1", usecols=["v1", "v2"])
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df.dropna(subset=["label", "message"]).drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].apply(clean_text)
    df = df[df["label"].isin(CLASS_LABELS)].copy()
    df = df[df["message"].str.len() > 0].reset_index(drop=True)
    return df


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    return value


def _combined_parameter_grid(classifier_grid: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [
        {**vectorizer_config, **classifier_parameters}
        for vectorizer_config in VECTORIZER_CONFIGS
        for classifier_parameters in classifier_grid
    ]


def _positive_scores(estimator: Pipeline, messages: pd.Series) -> tuple[np.ndarray, bool]:
    classifier = estimator.named_steps["classifier"]
    transformed = estimator.named_steps["tfidf"].transform(messages)
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(transformed)
        class_index = list(classifier.classes_).index(POSITIVE_LABEL)
        return probabilities[:, class_index], True

    scores = classifier.decision_function(transformed)
    if len(classifier.classes_) == 2 and classifier.classes_[1] != POSITIVE_LABEL:
        scores = -scores
    return np.asarray(scores), False


def _best_f1_threshold(y_true: pd.Series, positive_scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(
        y_true == POSITIVE_LABEL, positive_scores
    )
    f1_values = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + np.finfo(float).eps
    )
    best_f1 = np.max(f1_values)
    candidates = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_index = max(
        candidates,
        key=lambda index: (recall[index], precision[index], thresholds[index]),
    )
    return float(thresholds[best_index])


def _metric_summary(
    y_true: pd.Series,
    predictions: np.ndarray,
    positive_scores: np.ndarray,
) -> Dict[str, Any]:
    matrix = confusion_matrix(y_true, predictions, labels=CLASS_LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "roc_auc": float(roc_auc_score(y_true == POSITIVE_LABEL, positive_scores)),
        "pr_auc": float(average_precision_score(y_true == POSITIVE_LABEL, positive_scores)),
    }


def _make_curve_data(y_true: pd.Series, positive_scores: np.ndarray) -> Dict[str, Any]:
    binary_truth = y_true == POSITIVE_LABEL
    fpr, tpr, _ = roc_curve(binary_truth, positive_scores)
    precision, recall, thresholds = precision_recall_curve(binary_truth, positive_scores)
    return {
        "roc_curve": {
            "fpr": np.round(fpr, 6).tolist(),
            "tpr": np.round(tpr, 6).tolist(),
            "svg_points": " ".join(
                f"{x * 300:.1f},{(1 - y) * 170:.1f}" for x, y in zip(fpr, tpr)
            ),
        },
        "precision_recall_curve": {
            "precision": np.round(precision, 6).tolist(),
            "recall": np.round(recall, 6).tolist(),
            "thresholds": np.round(thresholds, 6).tolist(),
            "svg_points": " ".join(
                f"{x * 300:.1f},{(1 - y) * 170:.1f}"
                for x, y in zip(recall, precision)
            ),
        },
    }


def train_and_save_model(dataset_path: str | Path = DATASET_PATH):
    df = load_and_prepare_dataset(dataset_path)
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE + 1,
        stratify=y_train,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    fitted_searches = {}

    for model_name, (classifier, classifier_grid) in MODEL_DEFINITIONS.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english")),
                ("classifier", classifier),
            ]
        )
        search = GridSearchCV(
            pipeline,
            _combined_parameter_grid(classifier_grid),
            scoring=make_scorer(f1_score, pos_label=POSITIVE_LABEL, zero_division=0),
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_fit, y_fit)
        fitted_searches[model_name] = search
        positive_scores, has_probabilities = _positive_scores(search.best_estimator_, X_validation)

        if has_probabilities:
            threshold = _best_f1_threshold(y_validation, positive_scores)
            validation_predictions = np.where(positive_scores >= threshold, POSITIVE_LABEL, "ham")
        else:
            threshold = None
            validation_predictions = search.best_estimator_.predict(X_validation)

        comparison_metrics = _metric_summary(
            y_validation, validation_predictions, positive_scores
        )
        comparison.append(
            {
                "model_name": model_name,
                "has_probabilities": has_probabilities,
                "selection_eligible": has_probabilities,
                "decision_threshold": threshold,
                "cross_validation_f1": float(search.best_score_),
                "best_parameters": _json_safe(search.best_params_),
                **comparison_metrics,
            }
        )

    probability_candidates = [row for row in comparison if row["selection_eligible"]]
    selected_summary = max(
        probability_candidates,
        key=lambda row: (row["f1_score"], row["recall"], row["precision"]),
    )
    selected_name = selected_summary["model_name"]
    selected_search = fitted_searches[selected_name]
    decision_threshold = float(selected_summary["decision_threshold"])

    final_pipeline = selected_search.best_estimator_
    final_pipeline.fit(X_train, y_train)
    vectorizer = final_pipeline.named_steps["tfidf"]
    model = final_pipeline.named_steps["classifier"]
    test_scores, _ = _positive_scores(final_pipeline, X_test)
    test_predictions = np.where(test_scores >= decision_threshold, POSITIVE_LABEL, "ham")
    final_metrics = _metric_summary(y_test, test_predictions, test_scores)

    classification = classification_report(
        y_test,
        test_predictions,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    confusion = final_metrics["confusion_matrix"]
    tn, fp, fn, tp = confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]
    trained_at = datetime.now(timezone.utc).isoformat()
    metrics: Dict[str, Any] = {
        **final_metrics,
        "classification_report": classification,
        "selected_model": selected_name,
        "selected_model_parameters": _json_safe(selected_search.best_params_),
        "decision_threshold": decision_threshold,
        "threshold_selection": "F1 maximized on the separate stratified validation set; ties favor recall.",
        "validation_rows": int(len(X_validation)),
        "hyperparameter_training_rows": int(len(X_fit)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows": int(len(df)),
        "positive_label": POSITIVE_LABEL,
        "confusion_matrix_labels": CLASS_LABELS,
        "confusion_matrix_explanation": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "comparison_set": "Validation set; used for model selection and threshold tuning.",
        "comparison": comparison,
        "trained_at": trained_at,
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
        **_make_curve_data(y_test, test_scores),
    }
    metadata = {
        "model_name": selected_name,
        "model_class": type(model).__name__,
        "decision_threshold": decision_threshold,
        "probability_class_order": [str(label) for label in model.classes_],
        "positive_label": POSITIVE_LABEL,
        "random_state": RANDOM_STATE,
        "trained_at": trained_at,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "vectorizer_parameters": _json_safe(vectorizer.get_params()),
        "model_parameters": _json_safe(model.get_params()),
        "threshold_selection": metrics["threshold_selection"],
    }

    metadata_json = json.dumps(_json_safe(metadata), indent=2, allow_nan=False)
    metrics_json = json.dumps(_json_safe(metrics), indent=2, allow_nan=False)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METADATA_PATH.write_text(metadata_json, encoding="utf-8")
    METRICS_PATH.write_text(metrics_json, encoding="utf-8")
    return metrics, model, vectorizer


def load_model_bundle():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def load_model_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    with METADATA_PATH.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def predict_sms(message: str, model=None, vectorizer=None):
    if not isinstance(message, str):
        raise ValueError("Message must be a string.")

    if model is None or vectorizer is None:
        model, vectorizer = load_model_bundle()

    if model is None or vectorizer is None:
        raise FileNotFoundError("Model bundle not found. Train the model first.")

    cleaned_message = clean_text(message)
    if not cleaned_message:
        raise ValueError("Message cannot be empty after preprocessing.")

    metadata = load_model_metadata() if model is not None else {}
    model_name = (
        metadata.get("model_name", type(model).__name__)
        if metadata.get("model_class") == type(model).__name__
        else type(model).__name__
    )
    vectorized = vectorizer.transform([cleaned_message])
    class_labels = [str(label) for label in model.classes_]
    if POSITIVE_LABEL not in class_labels or "ham" not in class_labels:
        raise ValueError("The saved model must contain both ham and spam classes.")

    spam_probability: Optional[float] = None
    ham_probability: Optional[float] = None
    if hasattr(model, "predict_proba"):
        threshold = float(metadata.get("decision_threshold", 0.5))
        probabilities = model.predict_proba(vectorized)[0]
        probability_map = {
            label: float(probability)
            for label, probability in zip(class_labels, probabilities)
        }
        spam_probability = probability_map[POSITIVE_LABEL]
        ham_probability = probability_map["ham"]
        predicted_label = POSITIVE_LABEL if spam_probability >= threshold else "ham"
    else:
        threshold = float(
            metadata.get("decision_threshold", 0.0)
            if metadata.get("model_class") == type(model).__name__
            else 0.0
        )
        decision_scores = model.decision_function(vectorized)
        if len(class_labels) == 2 and class_labels[1] != POSITIVE_LABEL:
            decision_scores = -decision_scores
        predicted_label = POSITIVE_LABEL if decision_scores[0] >= threshold else "ham"

    return {
        "predicted_label": predicted_label,
        "spam_probability": round(spam_probability, 4) if spam_probability is not None else None,
        "ham_probability": round(ham_probability, 4) if ham_probability is not None else None,
        "model_name": model_name,
        "decision_threshold": threshold,
        "has_probabilities": spam_probability is not None,
    }


def evaluate_saved_model() -> Optional[Dict[str, Any]]:
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


if __name__ == "__main__":
    trained_metrics, _, _ = train_and_save_model()
    print(f"Selected model: {trained_metrics['selected_model']}")
    print(f"Test accuracy: {trained_metrics['accuracy']:.4f}")
    print(f"Test precision (spam): {trained_metrics['precision']:.4f}")
    print(f"Test recall (spam): {trained_metrics['recall']:.4f}")
    print(f"Test F1 (spam): {trained_metrics['f1_score']:.4f}")
    print(f"Test confusion matrix: {trained_metrics['confusion_matrix']}")
    print(f"Saved evaluation to {METRICS_PATH}")
