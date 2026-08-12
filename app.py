from flask import Flask, render_template, request, jsonify
from analytics import load_dataset
from model_utils import predict_review, ModelLoadError
from config import APP_HOST, APP_PORT, DEBUG

app = Flask(__name__)

def dashboard_payload(df):
    sentiment = df["Sentiment"].value_counts().reindex(
        ["Negative","Neutral","Positive"], fill_value=0
    )
    company = df.groupby("Company").agg(
        Reviews=("Sentiment","count"),
        Average_Rating=("Rating","mean"),
        Sentiment_Score=("Sentiment_Score","mean"),
        Negative=("Sentiment", lambda s:(s=="Negative").mean()*100),
        Neutral=("Sentiment", lambda s:(s=="Neutral").mean()*100),
        Positive=("Sentiment", lambda s:(s=="Positive").mean()*100),
    ).reset_index()
    return {
        "total": int(len(df)), "negative": int(sentiment["Negative"]),
        "neutral": int(sentiment["Neutral"]), "positive": int(sentiment["Positive"]),
        "avg_rating": float(df["Rating"].mean()) if df["Rating"].notna().any() else 0,
        "company": company.to_dict(orient="records"),
        "sentiment_labels": sentiment.index.tolist(),
        "sentiment_values": sentiment.tolist()
    }

@app.route("/")
def home():
    return render_template("index.html", result=None, error=None, review="")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review","").strip()
    if not review:
        return render_template("index.html", result=None, error="Please enter a review before predicting.", review="")
    try:
        return render_template("index.html", result=predict_review(review), error=None, review=review)
    except (ValueError, ModelLoadError) as exc:
        return render_template("index.html", result=None, error=str(exc), review=review)
    except Exception:
        return render_template("index.html", result=None, error="Prediction failed. Check the model and vectorizer files.", review=review)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(predict_review(str(data.get("review",""))))
    except ValueError as exc:
        return jsonify({"error":str(exc)}),400
    except ModelLoadError as exc:
        return jsonify({"error":str(exc)}),503
    except Exception:
        return jsonify({"error":"Prediction failed."}),500

@app.route("/dashboard")
def dashboard():
    try:
        return render_template("dashboard.html", data=dashboard_payload(load_dataset()), error=None)
    except Exception as exc:
        return render_template("dashboard.html", data=None, error=str(exc))

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG)
