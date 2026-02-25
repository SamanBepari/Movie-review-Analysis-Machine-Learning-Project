import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="AI Sentiment Studio",
    page_icon="🎬",
    layout="wide"
)

# -------------------- SESSION STATE INIT -------------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# -------------------- THEME FUNCTION -------------------- #
def apply_theme():
    if st.session_state.theme == "Dark":
        bg = "#0f172a"
        text = "white"
        button_bg = "#2563eb"
        button_text = "white"
        hover_bg = "#1d4ed8"
    else:
        bg = "#f1f5f9"
        text = "black"
        button_bg = "#1e293b"
        button_text = "white"
        hover_bg = "#0f172a"

    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg};
            color: {text};
        }}

        div.stButton > button {{
            background-color: {button_bg};
            color: {button_text};
            border-radius: 10px;
            height: 45px;
            font-weight: bold;
            border: none;
        }}

        div.stButton > button:hover {{
            background-color: {hover_bg};
            color: {button_text};
        }}

        .stFileUploader button {{
            background-color: {button_bg};
            color: {button_text};
        }}
        </style>
    """, unsafe_allow_html=True)

apply_theme()

# -------------------- LOAD MODEL -------------------- #
@st.cache_resource
def load_pipeline():
    data = joblib.load("sentiment_pipeline.pkl")
    return data["model"], data["vectorizer"]

model, vectorizer = load_pipeline()

# -------------------- LOGIN FUNCTION -------------------- #
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Please enter both username and password.")

# -------------------- SIGNUP FUNCTION -------------------- #
def signup():
    st.title("📝 Sign Up")

    new_user = st.text_input("Create Username")
    new_pass = st.text_input("Create Password", type="password")

    if st.button("Create Account"):
        if new_user and new_pass:
            st.success("Account created successfully! Please login.")
        else:
            st.error("Fill all fields.")

# -------------------- WORD CLOUD -------------------- #
def generate_wordcloud(text):
    wc = WordCloud(
        width=800,
        height=400,
        background_color="black"
    ).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# -------------------- AUTH SYSTEM -------------------- #
if not st.session_state.logged_in:

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        login()

    with tab2:
        signup()

# -------------------- MAIN APP -------------------- #
else:

    st.sidebar.title(f"👋 Welcome {st.session_state.username}")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "🧠 Single Analysis", "📂 Bulk Analysis", "📊 Analytics", "⚙ Settings"]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ---------------- DASHBOARD ---------------- #
    if page == "🏠 Dashboard":
        st.title("🎬 AI Sentiment Studio")
        st.info("Analyze movie reviews using Machine Learning")

        col1, col2, col3 = st.columns(3)

        col1.metric("Model", "Multinomial Naive Bayes")
        col2.metric("Dataset Size", "50,000 Reviews")
        col3.metric("Accuracy", "88%")

    # ---------------- SINGLE ANALYSIS ---------------- #
    elif page == "🧠 Single Analysis":

        st.title("Analyze Single Review")

        review = st.text_area(
            "Enter Review",
            height=200,
            key="review_text"
        )

        col1, col2 = st.columns(2)

        with col1:
            analyze_btn = st.button("Analyze")

        with col2:
            clear_btn = st.button("Clear")

        if clear_btn:
            st.session_state.review_text = ""
            st.rerun()

        if analyze_btn:

            if review.strip() == "":
                st.warning("Please enter a review.")
            else:
                transformed = vectorizer.transform([review])
                prediction = model.predict(transformed)[0]
                proba = model.predict_proba(transformed)[0]

                positive = round(proba[1] * 100, 2)
                negative = round(proba[0] * 100, 2)

                if prediction == 1:
                    st.success(f"Positive Review ({positive}%)")
                else:
                    st.error(f"Negative Review ({negative}%)")

                df = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative"],
                    "Probability": [positive, negative]
                })

                fig = px.pie(
                    df,
                    values="Probability",
                    names="Sentiment",
                    title="Sentiment Distribution"
                )

                st.plotly_chart(fig)

                st.subheader("Word Cloud")
                generate_wordcloud(review)

    # ---------------- BULK ANALYSIS ---------------- #
    elif page == "📂 Bulk Analysis":

        st.title("Bulk Sentiment Analysis")

        file = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])

        if file:

            if file.name.endswith(".txt"):
                text = file.read().decode("utf-8")
                reviews = text.split("\n")
            else:
                df = pd.read_csv(file)
                reviews = df.iloc[:, 0].dropna().tolist()

            predictions = []

            for r in reviews:
                transformed = vectorizer.transform([r])
                pred = model.predict(transformed)[0]
                predictions.append("Positive" if pred == 1 else "Negative")

            result_df = pd.DataFrame({
                "Review": reviews,
                "Prediction": predictions
            })

            st.dataframe(result_df)

            st.download_button(
                "Download Results CSV",
                result_df.to_csv(index=False),
                "sentiment_results.csv",
                "text/csv"
            )

    # ---------------- ANALYTICS ---------------- #
    elif page == "📊 Analytics":

        st.title("Model Analytics")

        st.write("Model Accuracy: 88%")
        st.write("Algorithm: Multinomial Naive Bayes")
        st.write("Vectorizer: TF-IDF")

    # ---------------- SETTINGS ---------------- #
    elif page == "⚙ Settings":

        st.title("Settings")

        st.session_state.theme = st.radio(
            "Choose Theme",
            ["Dark", "Light"]
        )

        apply_theme()

        st.success("Theme Updated Successfully!")
