import streamlit as st
import pandas as pd
import plotly.express as px
from analytics import load_dataset, filtered_data
from model_utils import predict_review, ModelLoadError


TOP_WORDS = [
    ("not", 801), ("booking", 631), ("no", 476), ("paypal", 438), ("com", 437),
    ("customer", 382), ("service", 356), ("money", 317), ("refund", 315), ("account", 298),
    ("get", 297), ("temu", 263), ("never", 246), ("use", 237), ("one", 228),
    ("even", 228), ("would", 221), ("time", 217), ("company", 188), ("back", 183),
]

NEGATIVE_FEATURES = [
    ("no", 1.813760), ("not", 1.475190), ("money", 1.400457), ("booking", 1.174516),
    ("service", 1.114500), ("customer", 1.090487), ("paypal", 1.029237),
    ("booking com", 1.020469), ("company", 0.994677), ("com", 0.993718),
]
POSITIVE_FEATURES = [
    ("canva", 2.676348), ("great", 2.338716), ("love", 1.644564), ("excellent", 1.610590),
    ("easy", 1.603039), ("helpful", 1.526759), ("professional", 1.268523),
    ("thank", 1.173690), ("super", 1.035766), ("quickly", 1.021511),
]
NEUTRAL_FEATURES = [
    ("music", 2.250976), ("advertise", 2.023801), ("shirt", 1.843876), ("fits", 1.810629),
    ("everyone", 1.754310), ("better", 1.682151), ("fast", 1.660146),
    ("free", 1.573568), ("good", 1.532006), ("bit high", 1.455355),
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Trustpilot Sentiment Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# COLOR PALETTE (shared across every chart) — Amber / Rose / Emerald
# ============================================================

ACCENT_AMBER = "#f59e0b"
ACCENT_ROSE = "#fb7185"
ACCENT_EMERALD = "#34d399"
ACCENT_SKY = "#38bdf8"
ACCENT_VIOLET = "#a78bfa"
ACCENT_RED = "#f87171"

SENTIMENT_COLORS = {
    "Negative": ACCENT_ROSE,
    "Neutral": ACCENT_AMBER,
    "Positive": ACCENT_EMERALD,
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#f1ede6", size=13),
    title_font=dict(family="Poppins, sans-serif", size=16, color="#ffffff"),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    colorway=[ACCENT_AMBER, ACCENT_ROSE, ACCENT_EMERALD, ACCENT_SKY, ACCENT_VIOLET],
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
)


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ============================================================
# THEME / CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; }

    /* ---------- Warm charcoal-indigo professional background ---------- */
    .stApp {
        background:
            radial-gradient(circle at 8% -8%, rgba(245, 158, 11, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 92% 6%, rgba(251, 113, 133, 0.10) 0%, transparent 48%),
            radial-gradient(circle at 50% 105%, rgba(52, 211, 153, 0.06) 0%, transparent 55%),
            linear-gradient(165deg, #12100f 0%, #1b1620 45%, #100d16 100%);
        color: #f1ede6;
    }

    /* ---------- Hero banner ---------- */
    .hero-banner {
        padding: 1.9rem 2.4rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #7c2d12 0%, #9d174d 55%, #4c1d95 100%);
        box-shadow: 0 20px 45px rgba(120, 40, 60, 0.35);
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero-banner::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }

    .hero-desc {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.85);
        max-width: 700px;
    }

    /* ---------- KPI cards ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(165deg, rgba(255,255,255,0.055), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.30);
    }

    div[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.6) !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Poppins', sans-serif; }

    /* ---------- Chart / content card wrapper ---------- */
    .chart-card {
        background: linear-gradient(165deg, rgba(255,255,255,0.045), rgba(255,255,255,0.012));
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 0.9rem 1rem 0.3rem 1rem;
        box-shadow: 0 10px 26px rgba(0,0,0,0.28);
        margin-bottom: 1.1rem;
    }

    .section-card {
        background: linear-gradient(165deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.30);
        margin-bottom: 1.1rem;
    }

    .section-card h3, .section-card h4 { margin-top: 0; color: #ffffff; }
    .section-card p, .section-card li { color: rgba(255,255,255,0.78); }

    .pill-tag {
        display: inline-block;
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        color: rgba(255,255,255,0.75);
        font-size: 0.8rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
    }

    .badge-neg { color: #fda4af !important; border-color: rgba(251,113,133,0.35) !important; background: rgba(251,113,133,0.10) !important; }
    .badge-pos { color: #6ee7b7 !important; border-color: rgba(52,211,153,0.35) !important; background: rgba(52,211,153,0.10) !important; }
    .badge-neu { color: #fcd34d !important; border-color: rgba(245,158,11,0.35) !important; background: rgba(245,158,11,0.10) !important; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 30% 0%, rgba(245,158,11,0.07), transparent 45%),
            radial-gradient(circle at 90% 40%, rgba(251,113,133,0.08), transparent 50%),
            linear-gradient(180deg, #16121a 0%, #0c0a10 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] * { color: #f1ede6 !important; }

    .sb-logo-wrap { text-align: center; padding: 0.6rem 0 0.4rem 0; }

    .sb-logo-badge {
        width: 58px;
        height: 58px;
        margin: 0 auto 0.5rem auto;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.7rem;
        background: linear-gradient(135deg, #7c2d12, #9d174d 55%, #4c1d95);
        box-shadow: 0 10px 26px rgba(157,23,77,0.35);
    }

    .sb-title {
        font-size: 1.15rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        margin-bottom: 0.1rem;
    }

    .sb-subtitle {
        font-size: 0.68rem;
        color: rgba(255,255,255,0.5) !important;
        letter-spacing: 1.4px;
        text-transform: uppercase;
    }

    .sb-section-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 0.88rem;
        margin: 0.5rem 0 0.4rem 0;
        color: #ffffff !important;
    }

    .sb-section-label .icon-chip-sm {
        width: 24px;
        height: 24px;
        min-width: 24px;
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        background: linear-gradient(135deg, rgba(245,158,11,0.28), rgba(251,113,133,0.28));
        border: 1px solid rgba(255,255,255,0.12);
    }

    /* Sidebar radio nav styled as a menu list */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.3rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 0.5rem 0.7rem !important;
        transition: all 0.15s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(245,158,11,0.10);
        border-color: rgba(245,158,11,0.35);
    }

    /* Sidebar multiselect boxes */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"] span[data-baseweb="tag"] {
        background: linear-gradient(120deg, #b45309, #9d174d) !important;
        border-radius: 8px !important;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        background: linear-gradient(120deg, #b45309, #9d174d);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(157,23,77,0.35);
    }

    /* ---------- Text area ---------- */
    div[data-testid="stTextArea"] textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        color: #eef1fa !important;
    }

    /* ---------- Dataframe ---------- */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.09);
    }

    hr { border-color: rgba(255,255,255,0.08) !important; }

    .footer-caption {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.8rem;
        padding-top: 0.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    return load_dataset()


# ============================================================
# SIDEBAR — LOGO + NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sb-logo-wrap">
            <div class="sb-logo-badge">🧠</div>
            <div class="sb-title">Sentiment Intelligence</div>
            <div class="sb-subtitle">NLP • Trustpilot Reviews</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">🧭</span> Navigation</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Dashboard", "Prediction", "Company Analysis", "Model Performance", "Word Analysis", "About Project"],
        label_visibility="collapsed",
    )

try:
    df = get_data()
    data_loaded = True
except Exception as exc:
    df = None
    data_loaded = False
    st.sidebar.divider()
    st.sidebar.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">⚠️</span> Data Status</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.error("Final CSV not found")
    st.sidebar.caption(str(exc))


def hero(title, desc):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">{title}</div>
            <div class="hero-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    hero(
        "🧠 Trustpilot Sentiment Intelligence",
        "983-review NLP project • TF-IDF vectorization • Logistic Regression (C = 2)",
    )

    if data_loaded:

        sent = df["Sentiment"].value_counts().reindex(["Negative", "Neutral", "Positive"], fill_value=0)

        cols = st.columns(5)
        vals = [len(df), sent["Negative"], sent["Positive"], sent["Neutral"], df["Rating"].mean()]
        labs = ["Total Reviews", "Negative", "Positive", "Neutral", "Average Rating"]
        icons = ["📊", "😠", "😊", "😐", "⭐"]

        for c, l, v, ic in zip(cols, labs, vals, icons):
            c.metric(f"{ic} {l}", f"{v:.2f}" if isinstance(v, float) else int(v))

        st.divider()

        chart_col, table_col = st.columns([1, 1.4])

        with chart_col:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig = px.pie(
                names=sent.index,
                values=sent.values,
                hole=0.55,
                color=sent.index,
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment Distribution",
            )
            fig.update_traces(marker=dict(line=dict(color="#0a1120", width=2)))
            st.plotly_chart(style_fig(fig), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with table_col:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown("**Company Summary**")
            company = df.groupby("Company").agg(
                Reviews=("Sentiment", "count"),
                Average_Rating=("Rating", "mean"),
                Sentiment_Score=("Sentiment_Score", "mean"),
            ).reset_index()
            st.dataframe(
                company.style.format({"Average_Rating": "{:.3f}", "Sentiment_Score": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Copy final_sentiment_dataset.csv into data/ to enable live filtering and raw-data analytics.")
        c = st.columns(4)
        icons = ["📊", "😠", "😊", "😐"]
        for col, ic, label, val in zip(
            c, icons, ["Total Reviews", "Negative", "Positive", "Neutral"], [983, "78.64%", "18.92%", "2.44%"]
        ):
            col.metric(f"{ic} {label}", val)


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    hero(
        "🔎 Live Sentiment Prediction",
        "Type or paste a customer review below to get an instant sentiment prediction from the trained model.",
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    review = st.text_area(
        "Enter a customer review",
        height=180,
        placeholder="The service was excellent and very easy to use...",
    )

    predict_clicked = st.button("✨ Predict Sentiment", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        try:
            r = predict_review(review)

            sentiment = r["sentiment"]
            badge_class = {
                "Negative": "badge-neg",
                "Positive": "badge-pos",
                "Neutral": "badge-neu",
            }.get(sentiment, "pill-tag")

            st.markdown(
                f"""
                <div class="section-card">
                    <div class="sb-section-label" style="font-size:1rem;">🎯 Predicted Sentiment</div>
                    <span class="pill-tag {badge_class}" style="font-size:1rem; font-weight:700;">{sentiment}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if r["confidence"] is not None:
                st.metric("Confidence", f"{r['confidence'] * 100:.1f}%")

            if r["probabilities"]:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                d = pd.DataFrame(
                    {"Sentiment": list(r["probabilities"]), "Probability": list(r["probabilities"].values())}
                )
                fig = px.bar(
                    d,
                    x="Sentiment",
                    y="Probability",
                    range_y=[0, 1],
                    text_auto=".1%",
                    color="Sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                    title="Class Probabilities",
                )
                fig.update_traces(marker_line_width=0, showlegend=False)
                st.plotly_chart(style_fig(fig), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except (ValueError, ModelLoadError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Prediction failed. Check the saved model and vectorizer.")


# ============================================================
# COMPANY ANALYSIS
# ============================================================

elif page == "Company Analysis":

    hero(
        "🏢 Company Analysis",
        "Filter and compare sentiment breakdowns across companies in the dataset.",
    )

    if not data_loaded:
        st.warning("Copy final_sentiment_dataset.csv into data/ first.")
    else:
        with st.sidebar:
            st.divider()
            st.markdown(
                '<div class="sb-section-label"><span class="icon-chip-sm">🏷️</span> Filters</div>',
                unsafe_allow_html=True,
            )
            cf = st.multiselect("Company", sorted(df["Company"].unique()))
            sf = st.multiselect("Sentiment", ["Negative", "Neutral", "Positive"])
            rf = st.multiselect("Rating", sorted(df["Rating"].dropna().unique()))

        view = filtered_data(df, cf, None, sf, rf)

        st.metric("🔍 Filtered Reviews", len(view))

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        company = view.groupby("Company").agg(
            Reviews=("Sentiment", "count"),
            Average_Rating=("Rating", "mean"),
            Sentiment_Score=("Sentiment_Score", "mean"),
            Positive_Pct=("Sentiment", lambda s: (s == "Positive").mean() * 100),
            Negative_Pct=("Sentiment", lambda s: (s == "Negative").mean() * 100),
            Neutral_Pct=("Sentiment", lambda s: (s == "Neutral").mean() * 100),
        ).reset_index()
        st.dataframe(
            company.style.format(
                {
                    "Average_Rating": "{:.3f}",
                    "Sentiment_Score": "{:.3f}",
                    "Positive_Pct": "{:.1f}%",
                    "Negative_Pct": "{:.1f}%",
                    "Neutral_Pct": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = px.bar(
            company,
            x="Company",
            y=["Negative_Pct", "Neutral_Pct", "Positive_Pct"],
            barmode="stack",
            title="Sentiment Composition by Company",
            color_discrete_map={
                "Negative_Pct": ACCENT_RED,
                "Neutral_Pct": ACCENT_AMBER,
                "Positive_Pct": ACCENT_EMERALD,
            },
        )
        fig.update_layout(yaxis_title="Percentage")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    hero(
        "🤖 Model Performance",
        "Comparison of candidate models and hyperparameter tuning results for the final classifier.",
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Multinomial Naive Bayes",
                "Linear SVM",
                "Random Forest",
                "Logistic Regression + Oversampling",
                "Logistic Regression C=2",
            ],
            "Accuracy": [0.9036, 0.8223, 0.8934, 0.8934, 0.9036, 0.9137],
            "Macro F1": [0.5732, 0.4056, 0.5534, 0.5536, 0.5699, 0.5845],
        }
    )
    st.dataframe(
        results.style.format({"Accuracy": "{:.2%}", "Macro F1": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.warning("Neutral has only 24 total reviews and 5 test reviews; Neutral test recall/F1 are 0.00.")

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    tuning = pd.DataFrame(
        {
            "C": [0.01, 0.10, 0.50, 1, 2, 5, 10],
            "Accuracy": [0.873096, 0.893401, 0.903553, 0.903553, 0.913706, 0.908629, 0.908629],
            "Macro F1": [0.548522, 0.560587, 0.572561, 0.573180, 0.584520, 0.577289, 0.577289],
        }
    )
    fig = px.line(
        tuning,
        x="C",
        y=["Accuracy", "Macro F1"],
        markers=True,
        log_x=True,
        title="Hyperparameter Tuning (Regularization C)",
        color_discrete_sequence=[ACCENT_AMBER, ACCENT_VIOLET],
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# WORD ANALYSIS
# ============================================================

elif page == "Word Analysis":

    hero(
        "📝 Word & Feature Analysis",
        "Most frequent words and the strongest coefficient features driving each sentiment class.",
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("**Top Words (from the supplied notebook)**")
    st.dataframe(
        pd.DataFrame(TOP_WORDS, columns=["Word", "Frequency"]),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("**Top Logistic Regression Coefficient Features**")
    st.dataframe(
        pd.DataFrame(
            {
                "Negative": [x[0] for x in NEGATIVE_FEATURES],
                "Positive": [x[0] for x in POSITIVE_FEATURES],
                "Neutral": [x[0] for x in NEUTRAL_FEATURES],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Coefficients describe contribution in this trained model; a word is not universally positive or negative without context.")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ABOUT PROJECT
# ============================================================

else:

    hero(
        "ℹ️ About This Project",
        "Trustpilot Review Sentiment Analysis Using Machine Learning",
    )

    st.markdown(
        """
        <div class="section-card">
            <h4>🔧 Pipeline</h4>
            <p>Raw review → preprocessing → saved TF-IDF vectorizer → Logistic Regression (C = 2) → sentiment + probability.</p>
        </div>

        <div class="section-card">
            <h4>🏆 Final Result</h4>
            <span class="pill-tag badge-pos">91.37% Accuracy</span>
            <span class="pill-tag" style="color:#c4b5fd; border-color:rgba(167,139,250,0.35); background:rgba(167,139,250,0.10);">0.5845 Macro F1</span>
        </div>

        <div class="section-card">
            <h4>⚠️ Critical Limitation</h4>
            <p>Only 24 Neutral reviews exist in the final dataset, and only 5 are in the test set — producing 0.00 Neutral recall/F1 in the reported evaluation.</p>
        </div>

        <div class="section-card">
            <h4>📌 Note</h4>
            <p>Company statistics describe the collected project sample, not each company's complete Trustpilot reputation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="footer-caption">🧠 Trustpilot Sentiment Intelligence &nbsp;•&nbsp; '
    'TF-IDF + Logistic Regression &nbsp;•&nbsp; Streamlit &amp; Plotly</div>',
    unsafe_allow_html=True,
)
