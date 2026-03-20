# 🎬 AI Sentiment Studio — Movie Review Analysis

A Streamlit-based web application for analyzing the sentiment of movie reviews using a pre-trained Machine Learning model (Multinomial Naive Bayes + TF-IDF).

---

## 📌 Features

- 🔐 **Login / Sign Up** — Simple session-based authentication
- 🧠 **Single Review Analysis** — Paste any movie review and get instant sentiment prediction with probability scores and a word cloud
- 📂 **Bulk Analysis** — Upload a `.txt` or `.csv` file to analyze multiple reviews at once and download results
- 📊 **Analytics Dashboard** — View model metadata (accuracy, algorithm, vectorizer)
- 🌗 **Theme Toggle** — Switch between Dark and Light UI themes

---

## 🗂️ Project Structure

```
Movie_Review_Analysis_Project/
├── app.py                          # Main Streamlit application
├── sentiment_pipeline.pkl          # Pre-trained model (Naive Bayes + TF-IDF vectorizer)
├── IMDB_Dataset_sample.xlsx        # Sample IMDB movie review dataset
├── Notebook_for_model_Creation.ipynb  # Jupyter notebook for model training
├── .vscode/
│   └── settings.json               # VS Code workspace settings
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Movie_Review_Analysis_Project.git
cd Movie_Review_Analysis_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Required packages:**
> `streamlit`, `joblib`, `pandas`, `plotly`, `matplotlib`, `wordcloud`, `scikit-learn`, `openpyxl`

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 Model Details

| Property    | Value                   |
|-------------|-------------------------|
| Algorithm   | Multinomial Naive Bayes |
| Vectorizer  | TF-IDF                  |
| Dataset     | IMDB Movie Reviews      |
| Dataset Size| 50,000 Reviews          |
| Accuracy    | 88%                     |

The trained model and vectorizer are bundled together in `sentiment_pipeline.pkl` and loaded via `joblib`.

---

## 📁 Dataset

The `IMDB_Dataset_sample.xlsx` file contains a sample of the IMDB movie review dataset used for training. Each record includes a review text and a binary sentiment label (`positive` / `negative`).

---

## 📓 Model Training

Open `Notebook_for_model_Creation.ipynb` in Jupyter to see the full model training pipeline — data preprocessing, TF-IDF vectorization, Naive Bayes training, and evaluation.

---

## 📄 License

This project is for educational purposes. Feel free to use and modify it.
