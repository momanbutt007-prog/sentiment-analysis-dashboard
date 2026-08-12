from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Change DATA_FILE only if your final CSV has a different filename.
DATA_FILE = BASE_DIR / "data" / "final_sentiment_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "logistic_regression_C2_FINAL.pkl"
VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

APP_HOST = "127.0.0.1"
APP_PORT = 5000
DEBUG = False

REQUIRED_COLUMNS = [
    "Company", "Industry", "Review Title",
    "Review Text", "Rating", "Sentiment"
]
