"""Shared production model loading and prediction logic."""
from pathlib import Path
import joblib
from config import MODEL_FILE, VECTORIZER_FILE
from preprocess import clean_text

class ModelLoadError(RuntimeError):
    pass

def load_production_artifacts():
    if not Path(MODEL_FILE).exists():
        raise ModelLoadError(
            f"Production model not found: {MODEL_FILE}. "
            "Copy logistic_regression_C2_FINAL.pkl from Colab into models/."
        )
    if not Path(VECTORIZER_FILE).exists():
        raise ModelLoadError(
            f"TF-IDF vectorizer not found: {VECTORIZER_FILE}. "
            "Copy tfidf_vectorizer.pkl from Colab into models/."
        )
    try:
        return joblib.load(MODEL_FILE), joblib.load(VECTORIZER_FILE)
    except Exception as exc:
        raise ModelLoadError(
            "Could not load the model/vectorizer. Check file integrity and "
            f"scikit-learn compatibility. Details: {exc}"
        ) from exc

def predict_review(review: str):
    review = "" if review is None else str(review).strip()
    if not review:
        raise ValueError("Please enter a review before predicting.")
    model, vectorizer = load_production_artifacts()
    cleaned = clean_text(review)
    if not cleaned:
        raise ValueError("The review became empty after preprocessing. Please enter more meaningful text.")
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = {}
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(features)[0]
        probabilities = {str(c): float(p) for c, p in zip(model.classes_, values)}
    return {
        "sentiment": str(prediction),
        "confidence": max(probabilities.values()) if probabilities else None,
        "probabilities": probabilities,
        "cleaned_text": cleaned,
    }
