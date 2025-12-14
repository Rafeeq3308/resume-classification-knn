import streamlit as st
import re
import joblib
import docx
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords

# Download stopwords (runs once)
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# -----------------------------
# LOAD SAVED MODELS
# -----------------------------
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
knn_model = joblib.load(os.path.join(BASE_DIR, "knn_grid_model.pkl"))


# -----------------------------
# TEXT EXTRACTION FUNCTIONS
# -----------------------------
def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return " ".join(p.text for p in doc.paragraphs)

    elif filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return " ".join(
            page.extract_text()
            for page in reader.pages
            if page.extract_text()
        )

    return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="Resume Classification (KNN)",
    layout="centered"
)

st.title("📄 Resume Classification using KNN")
st.write("Upload a resume file (DOCX or PDF) to predict the category")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["docx", "pdf"]
)

if uploaded_file is not None:
    raw_text = extract_text(uploaded_file)
    cleaned_text = clean_text(raw_text)

    if len(cleaned_text) < 50:
        st.error(" Resume text is too short or unreadable.")
    else:
        vector = tfidf.transform([cleaned_text])
        prediction = knn_model.predict(vector)

        st.success(f" Predicted Class (Numeric Label): **{prediction[0]}**")

