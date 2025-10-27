# The Reviewer Puzzle: NLP-Powered Reviewer Recommendation System

## 📝 Introduction

Finding suitable reviewers for academic papers is challenging due to the increasing specialization and interdisciplinary nature of research. This project, "The Reviewer Puzzle," tackles this problem by using Natural Language Processing (NLP) to recommend potential reviewers based on their publication history.

## 🎯 Objective

To design and build a system that accepts a research paper (PDF) as input and recommends a ranked list of the "Top k" authors from a provided dataset who are most likely to possess the expertise needed to review the paper.

## ✨ Features

* **PDF Upload:** Users can upload a research paper in PDF format.
* **Top-k Recommendations:** Generates a ranked list of the most suitable `k` reviewers (user-selectable).
* **Dual NLP Models:** Provides recommendations using two distinct methods for comparison:
    * **TF-IDF:** Matches reviewers based on shared **keywords** and phrases.
    * **Sentence Transformers (Embeddings):** Matches reviewers based on **semantic meaning** and conceptual similarity, leveraging a pre-trained language model (`all-MiniLM-L6-v2`).
* **Reviewer Similarity:** Includes an optional feature to find authors with similar overall research expertise to a selected author from the database, based on averaged paper embeddings.
* **Optimized Performance:** Utilizes pre-processed data and models (`.parquet`, `.npy`, `.joblib`) for fast application startup.

## 💻 Technology Stack

* **Language:** Python 3.11
* **Web Framework:** Streamlit
* **Core NLP/ML Libraries:**
    * `scikit-learn` (TF-IDF, Cosine Similarity)
    * `sentence-transformers` (Embeddings)
    * `nltk` (Text Preprocessing)
* **PDF Processing:**
    * `PyMuPDF` (fitz) - Primary text extraction
    * `pdf2image` - PDF-to-image conversion for OCR
    * `pytesseract` - Optical Character Recognition (OCR)
* **Data Handling:** `pandas`, `numpy`, `joblib`, `pyarrow`
* **System Dependencies (Handled by `packages.txt` on Streamlit Cloud):**
    * `poppler-utils` (Required by `pdf2image`)
    * `tesseract-ocr` & `tesseract-ocr-eng` (Required by `pytesseract`)

## 🚀 Accessing the App

This application is deployed and publicly accessible via Streamlit Community Cloud:

**➡️ [Access the Live App Here](https://your-app-name.streamlit.app/)** ⬅️
*(Replace this link with your actual Streamlit Cloud URL)*

No installation is required to use the deployed app. Just visit the link above.

## 📂 File Structure

* `app.py`: The main Streamlit application script.
* `requirements.txt`: Python package dependencies (installed via `pip`).
* `packages.txt`: System-level dependencies (installed via `apt-get` on Streamlit Cloud).
* `author_database.parquet`: Pre-processed DataFrame containing author names and cleaned text.
* `paper_embeddings.npy`: Pre-computed Sentence Transformer embeddings for all papers.
* `tfidf_vectorizer.joblib`, `tfidf_matrix.joblib`: Pre-computed TF-IDF model components.
* `model_cache/`: Directory containing the locally saved Sentence Transformer model files (loaded by `app.py`).

## ⚙️ Data Preprocessing Note

The core data files (`.parquet`, `.npy`, `.joblib`) were generated offline using a comprehensive pre-processing script (not included in the deployment). This script employed a robust 3-stage text extraction pipeline (PyMuPDF -> Tika -> Tesseract OCR) to handle the diverse and sometimes low-quality PDFs in the original dataset. The deployed application relies on these pre-processed files for its speed and efficiency.
