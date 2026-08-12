"""Save real trained objects from the supplied Colab notebook.

In Colab, after the model/vectorizer objects exist, run:
    %run save_models.py

No model is invented. Missing objects are skipped and reported.
"""
from pathlib import Path
import joblib

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "logistic_regression_C2_FINAL.pkl": ["final_model", "model"],
    "tfidf_vectorizer.pkl": ["tfidf", "tfidf_vectorizer"],
    "logistic_regression.pkl": ["lr_model"],
    "naive_bayes.pkl": ["nb_model"],
    "linear_svm.pkl": ["svm_model"],
    "random_forest.pkl": ["rf_model"],
    "logistic_regression_balanced.pkl": ["lr_balanced"],
}

def save_if_available(filename, names):
    namespace = globals()
    for name in names:
        if name in namespace:
            joblib.dump(namespace[name], OUTPUT_DIR / filename)
            print(f"[OK] {filename} <- {name}")
            return True
    print(f"[SKIP] {filename}: none of {names} exists.")
    return False

def main():
    saved = sum(save_if_available(f, n) for f,n in CANDIDATES.items())
    print(f"Saved {saved} artifact(s) to {OUTPUT_DIR.resolve()}")
    print("Required: logistic_regression_C2_FINAL.pkl and tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()
