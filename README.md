# Trustpilot Review Sentiment Analysis

A deployment-ready NLP/ML portfolio project built from the supplied Trustpilot sentiment-analysis notebook.

## What is included

- Flask prediction application
- Flask analytics dashboard
- Streamlit interactive dashboard
- Shared TF-IDF + Logistic Regression C=2 prediction pipeline
- 12 project charts extracted from the supplied notebook/results
- 25-slide PowerPoint final report
- Professional PDF final report
- Model-saving utility
- Robust configuration and error handling
- Python 3.12 setup and deployment instructions

> **Artifact note:** the uploaded notebook contains model-saving code and recorded outputs, but it does not contain binary `.pkl` files or the two CSV datasets. This project never fabricates those artifacts. Copy your real model/vectorizer and CSV files into the indicated folders.

## Final Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 90.36% | 0.5732 |
| Multinomial Naive Bayes | 82.23% | 0.4056 |
| Linear SVM | 89.34% | 0.5534 |
| Random Forest | 89.34% | 0.5536 |
| Logistic Regression + Oversampling | 90.36% | 0.5699 |
| **Logistic Regression C=2** | **91.37%** | **0.5845** |

## Dataset

983 cleaned reviews from:

- Uber Eats
- Temu
- Spotify
- Canva
- Booking.com
- PayPal

Final sentiment distribution:

- Negative: 773 (78.64%)
- Neutral: 24 (2.44%)
- Positive: 186 (18.92%)

The Neutral class is very small. The reported final test set contains only 5 Neutral examples, and Neutral precision/recall/F1 are 0.00. Accuracy must therefore be interpreted together with Macro F1 and class-level metrics.

## TF-IDF

The supplied notebook uses 5,000 features with unigrams and bigrams, `min_df=2`, `max_df=0.95`, and `sublinear_tf=True`.

The production system reuses the fitted vectorizer:

`models/tfidf_vectorizer.pkl`

Do not fit a new vectorizer during prediction.

## Final Model

`models/logistic_regression_C2_FINAL.pkl`

Logistic Regression, C=2.

Accuracy: 91.37%.

Macro F1: 0.5845.

Confusion matrix:

```text
[[154, 0, 1],
 [4,   0, 1],
 [10,  1, 26]]
```

## Company Insights

| Company | Reviews | Avg Rating | Sentiment Score |
|---|---:|---:|---:|
| Canva | 172 | 4.093 | +0.581 |
| Spotify | 194 | 1.541 | -0.758 |
| Temu | 198 | 1.414 | -0.833 |
| PayPal | 199 | 1.276 | -0.869 |
| Booking.com | 200 | 1.155 | -0.920 |
| Uber Eats | 20 | 1.150 | -0.900 |

Canva is dramatically more positive in this collected sample: 76.16% Positive and an average rating of about 4.09. Booking.com has 96% Negative in the collected sample. Uber Eats has 95% Negative but only 20 collected reviews, so that estimate is much less reliable.

These are **sample-level findings**, not claims about the complete Trustpilot reputation of any company.

## Explain the 12 Charts

1. **Overall Sentiment Distribution** — 773 Negative, 24 Neutral and 186 Positive reviews; Negative dominates and creates class imbalance.
2. **Sentiment Distribution by Company** — absolute sentiment counts for each company.
3. **Sentiment Percentage by Company** — normalizes each company to 100%, making company comparison easier.
4. **Average Rating by Company** — compares average ratings on the 1–5 scale.
5. **Sentiment Score vs Average Rating** — compares the -1/0/+1 sentiment score with average rating.
6. **Rating Distribution** — notebook output: 719 one-star, 54 two-star, 24 three-star, 38 four-star, 148 five-star.
7. **Review Count by Company** — shows sample size differences, especially Uber Eats' 20 reviews.
8. **Review Length Distribution** — notebook reports median word count 32 and maximum 451.
9. **Top 20 Words** — most frequent cleaned words across all reviews.
10. **Top Positive Words** — most frequent words inside Positive reviews.
11. **Top Negative Words** — most frequent words inside Negative reviews.
12. **Model Comparison** — compares Accuracy and Macro F1; Logistic Regression C=2 is best on both supplied metrics.

## Run on Windows

```powershell
python --version
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Add the real artifacts

Place:

```text
data/final_sentiment_dataset.csv
models/logistic_regression_C2_FINAL.pkl
models/tfidf_vectorizer.pkl
```

Optional models may also be copied into `models/`.

### Run Flask

```powershell
python app.py
```

Open:

`http://127.0.0.1:5000`

Dashboard:

`http://127.0.0.1:5000/dashboard`

### Run Streamlit

```powershell
streamlit run streamlit_app.py
```

## Save Models from Colab

In the completed notebook:

```python
from pathlib import Path
import joblib

Path("models").mkdir(exist_ok=True)
joblib.dump(final_model, "models/logistic_regression_C2_FINAL.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
```

Or run the included script from a notebook after the objects exist:

```python
%run save_models.py
```

The script checks for real objects and skips missing ones. It never creates fake models.

## Deployment

### Flask

Linux-compatible production command:

```bash
gunicorn app:app
```

A `Procfile` and `runtime.txt` are included.

### Streamlit

Deploy `streamlit_app.py` on Streamlit Community Cloud.

Neither application depends on `/content/`, Colab, Google Drive or notebook variables.

## GitHub

```bash
git init
git add .
git commit -m "Complete Trustpilot sentiment analysis project"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Do not commit `venv/`. Small model files may be committed; large binaries should use Git LFS or artifact storage.

## Project Structure

```text
sentiment_analysis_project/
├── app.py
├── streamlit_app.py
├── save_models.py
├── config.py
├── preprocess.py
├── model_utils.py
├── analytics.py
├── requirements.txt
├── README.md
├── LICENSE
├── Procfile
├── runtime.txt
├── presentation/final_report.pptx
├── reports/final_report.pdf
├── data/README.md
├── models/README.md
├── templates/index.html
├── templates/dashboard.html
├── static/css/style.css
├── static/js/dashboard.js
├── charts/
├── notebooks/sentiment_analysis.ipynb
└── sample/sample_predictions.md
```

## Limitations

- The dataset is a collected sample.
- Negative is 78.64% of the final data.
- Neutral is only 2.44%.
- Neutral test recall/F1 are 0.00.
- Uber Eats has only 20 reviews.
- TF-IDF has less contextual understanding than transformer models.

## Future Improvements

Larger balanced datasets, better Neutral coverage, class-aware training, transformer models, multilingual sentiment, explanation highlights, monitoring, drift detection and automated retraining.

## License

MIT.
