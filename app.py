import streamlit as st
import joblib

# ------------------- Page Config ------------------- #
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ------------------- Custom CSS Styling ------------------- #
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: white;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        background-color: #1e293b;
        color: white;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        height: 45px;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Load Model ------------------- #
@st.cache_resource
def load_pipeline():
    data = joblib.load("sentiment_pipeline.pkl")
    return data["model"], data["vectorizer"]

model, vectorizer = load_pipeline()

# ------------------- UI ------------------- #
st.markdown('<div class="title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by TF-IDF & Multinomial Naive Bayes</div>', unsafe_allow_html=True)

st.divider()

review = st.text_area(
    "Write your movie review below:",
    height=180,
    placeholder="Example: The movie had amazing performances and a gripping storyline..."
)

if st.button("Analyze Sentiment 🚀"):

    if review.strip() == "":
        st.warning("⚠ Please enter a review to analyze.")
    else:
        transformed = vectorizer.transform([review])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)

        confidence = round(max(probability[0]) * 100, 2)

        st.divider()

        if prediction == 1:
            st.success("😊 Positive Review")
            st.progress(int(confidence))
        else:
            st.error("😠 Negative Review")
            st.progress(int(confidence))

        st.info(f"Model Confidence: {confidence}%")

st.divider()
st.caption("Dataset: IMDB 50K Movie Reviews | Model: Multinomial Naive Bayes")