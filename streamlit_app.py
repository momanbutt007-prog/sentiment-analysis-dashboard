import streamlit as st
import pandas as pd
import plotly.express as px
from analytics import load_dataset, filtered_data
from model_utils import predict_review, ModelLoadError


TOP_WORDS = [
    ("not",801),("booking",631),("no",476),("paypal",438),("com",437),
    ("customer",382),("service",356),("money",317),("refund",315),("account",298),
    ("get",297),("temu",263),("never",246),("use",237),("one",228),
    ("even",228),("would",221),("time",217),("company",188),("back",183)
]

NEGATIVE_FEATURES = [
    ("no",1.813760),("not",1.475190),("money",1.400457),("booking",1.174516),
    ("service",1.114500),("customer",1.090487),("paypal",1.029237),
    ("booking com",1.020469),("company",0.994677),("com",0.993718)
]
POSITIVE_FEATURES = [
    ("canva",2.676348),("great",2.338716),("love",1.644564),("excellent",1.610590),
    ("easy",1.603039),("helpful",1.526759),("professional",1.268523),
    ("thank",1.173690),("super",1.035766),("quickly",1.021511)
]
NEUTRAL_FEATURES = [
    ("music",2.250976),("advertise",2.023801),("shirt",1.843876),("fits",1.810629),
    ("everyone",1.754310),("better",1.682151),("fast",1.660146),
    ("free",1.573568),("good",1.532006),("bit high",1.455355)
]

st.set_page_config(page_title="Trustpilot Sentiment Intelligence", page_icon="🧠", layout="wide")
st.markdown("""<style>
.stApp{background:#0a0d12;color:#eef2f7}[data-testid="stSidebar"]{background:#0f141c}
.kpi{background:#11161f;border:1px solid #222b37;border-radius:16px;padding:18px}
.kpi-label{color:#8995a7;font-size:12px}.kpi-value{font-size:30px;font-weight:800}
.accent{color:#ff6744}.small{color:#8995a7}
</style>""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_dataset()

st.sidebar.title("🧠 Sentiment Intelligence")
page = st.sidebar.radio("Navigation", ["Dashboard","Prediction","Company Analysis","Model Performance","Word Analysis","About Project"])

try:
    df=get_data(); data_loaded=True
except Exception as exc:
    df=None; data_loaded=False
    st.sidebar.error("Final CSV not found")
    st.sidebar.caption(str(exc))

if page=="Dashboard":
    st.title("Trustpilot Sentiment Intelligence")
    st.caption("983-review NLP project • TF-IDF • Logistic Regression C=2")
    if data_loaded:
        sent=df["Sentiment"].value_counts().reindex(["Negative","Neutral","Positive"],fill_value=0)
        cols=st.columns(5)
        vals=[len(df),sent["Negative"],sent["Positive"],sent["Neutral"],df["Rating"].mean()]
        labs=["Total Reviews","Negative","Positive","Neutral","Average Rating"]
        for c,l,v in zip(cols,labs,vals): c.metric(l,f"{v:.2f}" if isinstance(v,float) else int(v))
        fig=px.pie(names=sent.index,values=sent.values,hole=.55)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig,use_container_width=True)
        company=df.groupby("Company").agg(Reviews=("Sentiment","count"),Average_Rating=("Rating","mean"),Sentiment_Score=("Sentiment_Score","mean")).reset_index()
        st.dataframe(company.style.format({"Average_Rating":"{:.3f}","Sentiment_Score":"{:.3f}"}),use_container_width=True,hide_index=True)
    else:
        st.warning("Copy final_sentiment_dataset.csv into data/ to enable live filtering and raw-data analytics.")
        c=st.columns(4)
        for col,label,val in zip(c,["Total Reviews","Negative","Positive","Neutral"],[983,"78.64%","18.92%","2.44%"]): col.metric(label,val)

elif page=="Prediction":
    st.title("🔎 Live Sentiment Prediction")
    review=st.text_area("Enter a customer review",height=180,placeholder="The service was excellent and very easy to use...")
    if st.button("Predict Sentiment",type="primary",use_container_width=True):
        try:
            r=predict_review(review); st.success(f"Predicted sentiment: {r['sentiment']}")
            if r["confidence"] is not None: st.metric("Confidence",f"{r['confidence']*100:.1f}%")
            if r["probabilities"]:
                d=pd.DataFrame({"Sentiment":list(r["probabilities"]), "Probability":list(r["probabilities"].values())})
                fig=px.bar(d,x="Sentiment",y="Probability",range_y=[0,1],text_auto=".1%"); fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig,use_container_width=True)
        except (ValueError,ModelLoadError) as exc: st.error(str(exc))
        except Exception: st.error("Prediction failed. Check the saved model and vectorizer.")

elif page=="Company Analysis":
    st.title("🏢 Company Analysis")
    if not data_loaded: st.warning("Copy final_sentiment_dataset.csv into data/ first.")
    else:
        cf=st.sidebar.multiselect("Company",sorted(df["Company"].unique()))
        sf=st.sidebar.multiselect("Sentiment",["Negative","Neutral","Positive"])
        rf=st.sidebar.multiselect("Rating",sorted(df["Rating"].dropna().unique()))
        view=filtered_data(df,cf,None,sf,rf)
        st.metric("Filtered Reviews",len(view))
        company=view.groupby("Company").agg(Reviews=("Sentiment","count"),Average_Rating=("Rating","mean"),Sentiment_Score=("Sentiment_Score","mean"),Positive_Pct=("Sentiment",lambda s:(s=="Positive").mean()*100),Negative_Pct=("Sentiment",lambda s:(s=="Negative").mean()*100),Neutral_Pct=("Sentiment",lambda s:(s=="Neutral").mean()*100)).reset_index()
        st.dataframe(company.style.format({"Average_Rating":"{:.3f}","Sentiment_Score":"{:.3f}","Positive_Pct":"{:.1f}%","Negative_Pct":"{:.1f}%","Neutral_Pct":"{:.1f}%"}),use_container_width=True,hide_index=True)
        fig=px.bar(company,x="Company",y=["Negative_Pct","Neutral_Pct","Positive_Pct"],barmode="stack"); fig.update_layout(template="plotly_dark",yaxis_title="Percentage")
        st.plotly_chart(fig,use_container_width=True)

elif page=="Model Performance":
    st.title("🤖 Model Performance")
    results=pd.DataFrame({"Model":["Logistic Regression","Multinomial Naive Bayes","Linear SVM","Random Forest","Logistic Regression + Oversampling","Logistic Regression C=2"],"Accuracy":[.9036,.8223,.8934,.8934,.9036,.9137],"Macro F1":[.5732,.4056,.5534,.5536,.5699,.5845]})
    st.dataframe(results.style.format({"Accuracy":"{:.2%}","Macro F1":"{:.4f}"}),use_container_width=True,hide_index=True)
    st.warning("Neutral has only 24 total reviews and 5 test reviews; Neutral test recall/F1 are 0.00.")
    tuning=pd.DataFrame({"C":[.01,.10,.50,1,2,5,10],"Accuracy":[.873096,.893401,.903553,.903553,.913706,.908629,.908629],"Macro F1":[.548522,.560587,.572561,.573180,.584520,.577289,.577289]})
    fig=px.line(tuning,x="C",y=["Accuracy","Macro F1"],markers=True,log_x=True); fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)

elif page=="Word Analysis":
    st.title("📝 Word & Feature Analysis")
    st.write("Top words from the supplied notebook")
    st.dataframe(pd.DataFrame(TOP_WORDS,columns=["Word","Frequency"]),use_container_width=True,hide_index=True)
    st.subheader("Top Logistic Regression coefficient features")
    st.dataframe(pd.DataFrame({
        "Negative":[x[0] for x in NEGATIVE_FEATURES],
        "Positive":[x[0] for x in POSITIVE_FEATURES],
        "Neutral":[x[0] for x in NEUTRAL_FEATURES]
    }),use_container_width=True,hide_index=True)
    st.caption("Coefficients describe contribution in this trained model; a word is not universally positive or negative without context.")

else:
    st.title("ℹ️ About Project")
    st.markdown("""**Trustpilot Review Sentiment Analysis Using Machine Learning**

Pipeline: Raw review → preprocessing → saved TF-IDF → Logistic Regression C=2 → sentiment + probability.

**Final result:** 91.37% accuracy and 0.5845 Macro F1.

**Critical limitation:** only 24 Neutral reviews exist in the final dataset and only 5 are in the test set, producing 0.00 Neutral recall/F1 in the reported evaluation.

Company statistics describe the collected project sample, not each company's complete Trustpilot reputation.
""")
