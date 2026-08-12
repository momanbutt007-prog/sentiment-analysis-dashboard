"""Shared text cleaning before TF-IDF transformation."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NEGATIONS = {
    "no", "not", "never", "neither", "nor", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't",
    "couldn't", "shouldn't", "can't", "cannot"
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS

def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word for word in text.split() if word not in STOP_WORDS)
