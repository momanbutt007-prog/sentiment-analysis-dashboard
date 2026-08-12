"""Robust dataset loading and filtering."""
from pathlib import Path
import pandas as pd
from config import DATA_FILE, REQUIRED_COLUMNS

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    aliases = {
        "review_text":"Review Text", "reviewtext":"Review Text",
        "review title":"Review Title", "review_title":"Review Title",
        "company":"Company", "industry":"Industry", "rating":"Rating",
        "sentiment":"Sentiment"
    }
    rename = {}
    for c in df.columns:
        key = c.lower().replace("_"," ").strip()
        if key in aliases:
            rename[c] = aliases[key]
    return df.rename(columns=rename)

def load_dataset(path=None):
    path = Path(path or DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Copy final_sentiment_dataset.csv "
            "into data/ or change DATA_FILE in config.py."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read CSV: {exc}") from exc
    df = normalize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    df["Review Text"] = df["Review Text"].fillna("").astype(str)
    df["Company"] = df["Company"].fillna("Unknown").astype(str)
    df["Industry"] = df["Industry"].fillna("Unknown").astype(str)
    df["Sentiment"] = df["Sentiment"].fillna("Unknown").astype(str)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    if "Word_Count" not in df.columns:
        df["Word_Count"] = df["Review Text"].str.split().str.len()
    if "Character_Count" not in df.columns:
        df["Character_Count"] = df["Review Text"].str.len()
    if "Sentiment_Score" not in df.columns:
        df["Sentiment_Score"] = df["Sentiment"].map({"Negative":-1,"Neutral":0,"Positive":1})
    return df

def filtered_data(df, companies=None, industries=None, sentiments=None, ratings=None):
    out = df.copy()
    if companies: out = out[out["Company"].isin(companies)]
    if industries: out = out[out["Industry"].isin(industries)]
    if sentiments: out = out[out["Sentiment"].isin(sentiments)]
    if ratings: out = out[out["Rating"].isin(ratings)]
    return out
